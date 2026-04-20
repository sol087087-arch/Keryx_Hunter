# keryx/tools/rag_search.py
# RAG Search Tool for KeryxHunter — semantic search over code, hypotheses, crashes and evidence.
# Uses LocalEmbedder + SharedContext with parallel async search and intelligent merging.
# Sovereign, air-gapped, fast, privacy-safe.

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any

from ..core.shared_context import SharedContext
from ..embeddings import SimilarityHit, create_embedder
from .Toolbox import BaseTool, ToolResult

logger = logging.getLogger("keryx.tools.rag_search")

# Minimum text length to be a useful hit for the agent
_MIN_TEXT_LENGTH = 20

# Context hits get a small boost to prioritise what the agent already knows
_CONTEXT_SCORE_BOOST = 0.05


class RagSearchTool(BaseTool):
    """
    Semantic search tool.

    Searches over:
    - Source code / functions (from embedder FAISS/numpy index)
    - Past hypotheses
    - Confirmed vulnerabilities
    - Evidence stored in SharedContext

    All vectors are L2-normalised before comparison so dot product == cosine similarity.
    """

    name        = "rag_search"
    description = (
        "Semantic search over codebase, hypotheses, crashes and evidence. "
        "Returns most relevant snippets with similarity scores."
    )

    def __init__(
        self,
        embedder=None,
        top_k: int = 8,
        min_score: float = 0.65,
    ) -> None:
        super().__init__()
        self.embedder  = embedder or create_embedder()
        self.top_k     = top_k
        self.min_score = min_score

        logger.info(
            "[RagSearchTool] Initialized | top_k=%d | min_score=%.2f",
            top_k, min_score,
        )

    # ------------------------------------------------------------------
    # Public execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        action_input: dict[str, Any],
        context: SharedContext | None = None,
    ) -> ToolResult:
        """
        Expected action_input keys:
            query     (str, required)  — what to search for
            top_k     (int, optional)  — override instance default
            min_score (float, optional)— override instance default
        """
        start = time.time()

        query = action_input.get("query", "").strip()
        if not query:
            self._record_call(success=False, duration_ms=0.0)
            return ToolResult(
                success=False,
                output="Missing 'query' in action_input.",
                error="query_missing",
            )

        top_k     = int(action_input.get("top_k",     self.top_k))
        min_score = float(action_input.get("min_score", self.min_score))

        # ── Parallel search: main index + dynamic context ────────────────────
        index_task   = self.embedder.similarity_search(query, top_k=top_k * 3)
        context_task = self._semantic_context_hits(query, context, top_k=5, min_score=min_score)

        index_hits, ctx_hits = await asyncio.gather(index_task, context_task)

        # ── Merge, boost context hits, deduplicate, rank ─────────────────────
        merged = self._merge_and_rank(index_hits, ctx_hits, top_k)

        # ── Final filter: score threshold + minimum text length ──────────────
        final = [
            h for h in merged
            if h.score >= min_score and len(h.text) >= _MIN_TEXT_LENGTH
        ]

        duration_ms = (time.time() - start) * 1000
        success     = True  # even zero results is a successful search
        self._record_call(success=success, duration_ms=duration_ms)

        output = self._format_for_llm(final, query)

        logger.info(
            "[RagSearchTool] query='%s...' hits=%d elapsed=%.0fms",
            query[:60], len(final), duration_ms,
        )

        return ToolResult(
            success=success,
            output=output,
            data={
                "query":            query,
                "hits_count":       len(final),
                "hits": [
                    {
                        "id":     h.id,
                        "text":   h.text[:500],
                        "score":  round(h.score, 4),
                        "source": h.metadata.get("source", "embedder"),
                    }
                    for h in final
                ],
                "execution_time_ms": round(duration_ms, 1),
            },
            metadata={
                "tool":  self.name,
                "query": query[:100],
            },
        )

    # ------------------------------------------------------------------
    # Semantic search over SharedContext
    # ------------------------------------------------------------------

    async def _semantic_context_hits(
        self,
        query: str,
        context: SharedContext | None,
        top_k: int,
        min_score: float,
    ) -> list[SimilarityHit]:
        """
        Embed hypotheses and confirmed vulns from SharedContext and rank them
        against the query.  Uses the same embedder so scores are comparable.

        BUG-FIX: embed_batch returns results[i] aligned with texts[i] even on
        partial failures (LocalEmbedder fills the slot with success=False).
        We iterate by index — never filter before zipping — so alignment holds.
        """
        if context is None:
            return []

        candidates: list[dict[str, str]] = []

        for hyp in getattr(context, "hypotheses", []):
            candidates.append({"text": hyp, "src": "hypothesis"})

        for vuln in getattr(context, "confirmed_vulns", []):
            desc = vuln.get("hypothesis") or vuln.get("description", "")
            if desc:
                candidates.append({"text": desc, "src": "confirmed_vuln"})

        if not candidates:
            return []

        # Embed query and all candidates in one shot
        texts        = [c["text"] for c in candidates]
        query_result = await self.embedder.embed(query)
        batch_results = await self.embedder.embed_batch(texts)

        if not query_result.success or not query_result.vector:
            return []

        # Normalise query vector for true cosine similarity
        qvec = _normalize(query_result.vector)

        hits: list[SimilarityHit] = []

        # BUG-FIX: iterate by index, never filter before zipping
        for i, res in enumerate(batch_results):
            if not res.success or not res.vector:
                continue  # slot failed — skip, index alignment unaffected
            cvec  = _normalize(res.vector)
            score = float(_dot(qvec, cvec))
            if score >= min_score:
                hits.append(SimilarityHit(
                    id=f"ctx_{i}",
                    text=candidates[i]["text"],
                    score=score,
                    metadata={"source": candidates[i]["src"]},
                ))

        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Merge + rank
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_and_rank(
        index_hits: list[SimilarityHit],
        ctx_hits: list[SimilarityHit],
        top_k: int,
    ) -> list[SimilarityHit]:
        """
        Merge index hits and context hits, deduplicate on first 100 chars,
        apply a small boost to context hits (they are immediately relevant).

        BUG-FIX: use dataclasses.replace() instead of mutating h.score in-place.
        Mutating would corrupt objects that may be cached/shared by the embedder.
        """
        seen:   set           = set()
        merged: list[SimilarityHit] = []

        # Context hits first (they get the boost)
        for h in ctx_hits:
            key = h.text[:100]
            if key not in seen:
                seen.add(key)
                # BUG-FIX: create a new instance, never mutate the original
                merged.append(replace(h, score=min(h.score + _CONTEXT_SCORE_BOOST, 1.0)))

        # Index hits — add only if not already represented
        for h in index_hits:
            key = h.text[:100]
            if key not in seen:
                seen.add(key)
                merged.append(h)

        return sorted(merged, key=lambda h: h.score, reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_for_llm(hits: list[SimilarityHit], query: str) -> str:
        if not hits:
            return f"No relevant results found for query: {query}"

        lines = [f"### RAG Search Results for: {query}", ""]
        for i, hit in enumerate(hits, 1):
            source = hit.metadata.get("source", "embedder")
            snippet = hit.text[:400] + ("..." if len(hit.text) > 400 else "")
            lines.append(f"{i}. [score={hit.score:.3f}] [id={hit.id}] [src={source}]")
            lines.append(f"   {snippet}")
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pure-Python math helpers (no numpy import at module level — stays optional)
# ---------------------------------------------------------------------------

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalise so dot product equals cosine similarity."""
    norm = _dot(vec, vec) ** 0.5
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_rag_search_tool(
    embedder=None,
    top_k: int = 8,
    min_score: float = 0.65,
) -> RagSearchTool:
    return RagSearchTool(embedder=embedder, top_k=top_k, min_score=min_score)


# ---------------------------------------------------------------------------
# Standalone one-shot helper (for scripts / tests)
# ---------------------------------------------------------------------------

async def rag_search(
    query: str,
    top_k: int = 8,
    min_score: float = 0.65,
    embedder=None,
) -> list[dict[str, Any]]:
    """One-shot search — returns the structured hits list, not a ToolResult."""
    tool   = create_rag_search_tool(embedder=embedder, top_k=top_k, min_score=min_score)
    result = await tool.execute({"query": query, "top_k": top_k, "min_score": min_score})
    if result.success:
        return result.data.get("hits", [])
    return []
