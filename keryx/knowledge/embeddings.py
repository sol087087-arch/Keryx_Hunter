# keryx/embeddings.py
# Local embeddings engine for KeryxHunter — sovereign, air-gapped, zero-cloud.
# sentence-transformers (all-MiniLM-L6-v2) or dummy hash vectors.
# In-memory vector index, cosine similarity, FAISS backend, LRU cache.

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor   # FIX 8: Thread not Process
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("keryx.embeddings")


# ---------------------------------------------------------------------------
# Optional heavy dependencies — ALL in try/except
# FIX 1: numpy was imported at module level without guard
# ---------------------------------------------------------------------------

_ST_AVAILABLE     = False
_FAISS_AVAILABLE  = False
_NUMPY_AVAILABLE  = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    pass

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore

try:
    import numpy as np   # FIX 1: guarded import
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore


# ---------------------------------------------------------------------------
# LRU cache with size limit
# FIX 9: unbounded _cache → LRU with configurable maxsize
# ---------------------------------------------------------------------------

class _LRUCache:
    def __init__(self, maxsize: int = 8192) -> None:
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[List[float]]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: List[float]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Callable wrapper for ThreadPoolExecutor (no lambda, no pickle issues)
# FIX 4: ProcessPoolExecutor can't pickle lambdas.
# FIX 2+3: run_in_executor extra args are passed to the callable, not to lambda.
#           Using a proper callable object avoids both issues cleanly.
# ---------------------------------------------------------------------------

class _EncodeCallable:
    """Picklable callable for model.encode — works with both Thread and Process pools."""

    def __init__(self, model: Any, batch_size: int) -> None:
        self._model      = model
        self._batch_size = batch_size

    def __call__(self, texts: Any) -> Any:
        if isinstance(texts, str):
            return self._model.encode(texts, convert_to_numpy=True)
        return self._model.encode(
            texts, batch_size=self._batch_size, convert_to_numpy=True
        )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingResult:
    success:           bool
    vector:            Optional[List[float]] = None
    embedding_time_ms: float                 = 0.0
    error:             Optional[str]         = None
    data:              Dict[str, Any]        = field(default_factory=dict)


@dataclass
class SimilarityHit:
    id:       str
    text:     str
    score:    float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LocalEmbedder
# ---------------------------------------------------------------------------

class LocalEmbedder:
    """
    Sovereign local embeddings engine.

    Default: all-MiniLM-L6-v2 (384-dim, ~90MB).
    Fully air-gapped — no network calls ever.
    Falls back to deterministic hash vectors if sentence-transformers unavailable.
    """

    def __init__(
        self,
        model_name:    str           = "all-MiniLM-L6-v2",
        cache_dir:     Optional[str] = None,
        device:        str           = "cpu",
        use_faiss:     bool          = True,
        batch_size:    int           = 32,
        cache_maxsize: int           = 8192,
    ) -> None:
        self.model_name = model_name
        self.device     = device
        self.use_faiss  = use_faiss and _FAISS_AVAILABLE and _NUMPY_AVAILABLE
        self.batch_size = batch_size
        self.cache_dir  = Path(cache_dir or Path.home() / ".cache" / "keryx" / "embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model:     Any           = None
        self._encode_fn: Optional[_EncodeCallable] = None
        self._dimension: int           = 384
        self._cache     = _LRUCache(maxsize=cache_maxsize)   # FIX 9

        self._index           = None
        self._id_to_text:     Dict[str, str]  = {}
        self._id_to_metadata: Dict[str, Dict] = {}

        # numpy fallback storage
        self._vectors: List[Any]  = []   # List[np.ndarray]
        self._ids:     List[int]  = []

        # FIX 8: ThreadPoolExecutor — model lives in main process, thread shares it.
        # ProcessPoolExecutor would spawn child processes that need to reload the model.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed")
        self._call_count = 0

        logger.info(
            f"[LocalEmbedder] Initialized | model={model_name} | "
            f"faiss={self.use_faiss} | device={device} | cache_max={cache_maxsize}"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not _ST_AVAILABLE:
            logger.warning("[LocalEmbedder] sentence-transformers not installed — using hash vectors")
            self._model = "dummy"
            return
        if not _NUMPY_AVAILABLE:
            logger.warning("[LocalEmbedder] numpy not installed — using hash vectors")
            self._model = "dummy"
            return
        start = time.time()
        try:
            self._model     = SentenceTransformer(self.model_name, device=self.device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._encode_fn = _EncodeCallable(self._model, self.batch_size)
            logger.info(
                f"[LocalEmbedder] Loaded {self.model_name} in "
                f"{(time.time()-start)*1000:.0f}ms | dim={self._dimension}"
            )
        except Exception as exc:
            logger.error(f"[LocalEmbedder] Model load failed: {exc}")
            self._model = "dummy"

    def _get_cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def _get_dummy_vector(self, text: str) -> List[float]:
        """Deterministic hash-based vector — reproducible across runs."""
        h = hashlib.sha256(text.encode()).hexdigest()
        return [int(h[i % len(h)], 16) / 15.0 - 0.5 for i in range(self._dimension)]

    def _normalize_1d(self, vec: Any) -> Any:
        """
        Normalize a 1D numpy vector in-place.
        FIX 6: only operates on 1D — caller must not pass 2D arrays.
        """
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _get_faiss_index(self) -> Any:
        if self._index is None and faiss is not None:
            sub   = faiss.IndexFlatIP(self._dimension)
            self._index = faiss.IndexIDMap(sub)
        return self._index

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> EmbeddingResult:
        self._load_model()
        start        = time.time()
        self._call_count += 1
        cache_key    = self._get_cache_key(text)

        cached = self._cache.get(cache_key)
        if cached is not None:
            return EmbeddingResult(
                success=True, vector=cached,
                embedding_time_ms=(time.time() - start) * 1000,
            )

        try:
            if self._model == "dummy" or not _NUMPY_AVAILABLE:
                vector = self._get_dummy_vector(text)
            else:
                # FIX 2+4: _EncodeCallable is picklable and takes one positional arg
                loop   = asyncio.get_running_loop()
                arr    = await loop.run_in_executor(self._executor, self._encode_fn, text)
                vector = arr.tolist()

            self._cache.put(cache_key, vector)
            return EmbeddingResult(
                success=True, vector=vector,
                embedding_time_ms=(time.time() - start) * 1000,
            )
        except Exception as exc:
            logger.error(f"[LocalEmbedder] embed failed: {exc}")
            return EmbeddingResult(success=False, error=str(exc))

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Embed multiple texts in one model call (much faster than loop)."""
        self._load_model()
        self._call_count += 1

        if self._model == "dummy" or not _NUMPY_AVAILABLE:
            # FIX 5: fall back to per-item dummy, not blanket failures
            return [
                EmbeddingResult(success=True, vector=self._get_dummy_vector(t))
                for t in texts
            ]

        # Check cache first — only encode cache misses
        results:    List[Optional[EmbeddingResult]] = [None] * len(texts)
        miss_idx:   List[int]  = []
        miss_texts: List[str]  = []

        for i, text in enumerate(texts):
            cached = self._cache.get(self._get_cache_key(text))
            if cached is not None:
                results[i] = EmbeddingResult(success=True, vector=cached)
            else:
                miss_idx.append(i)
                miss_texts.append(text)

        if miss_texts:
            try:
                # FIX 3: _EncodeCallable takes the list as single positional arg
                loop    = asyncio.get_running_loop()
                all_arr = await loop.run_in_executor(self._executor, self._encode_fn, miss_texts)
                for j, arr in enumerate(all_arr):
                    vec = arr.tolist()
                    self._cache.put(self._get_cache_key(miss_texts[j]), vec)
                    results[miss_idx[j]] = EmbeddingResult(success=True, vector=vec)
            except Exception as exc:
                logger.error(f"[LocalEmbedder] batch embed failed: {exc}")
                for j in miss_idx:
                    if results[j] is None:
                        results[j] = EmbeddingResult(success=False, error=str(exc))

        return [r or EmbeddingResult(success=False, error="unknown") for r in results]

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    async def add_batch_to_index(
        self,
        items: List[Tuple[int, str, Optional[Dict]]],
    ) -> int:
        """
        Add items to FAISS or numpy fallback index.

        items: [(int_id, text, metadata), ...]

        FIX 7: valid_ids was a Python list — .tolist() doesn't exist on list.
        Fixed by keeping it as a plain Python list throughout.
        """
        texts = [it[1] for it in items]
        int_ids = [it[0] for it in items]

        embed_results = await self.embed_batch(texts)

        valid_vecs: List[Any] = []
        valid_ids:  List[int] = []

        for i, res in enumerate(embed_results):
            if res.success and res.vector is not None and _NUMPY_AVAILABLE:
                vec = np.array(res.vector, dtype=np.float32)
                vec = self._normalize_1d(vec)
                valid_vecs.append(vec)
                valid_ids.append(int_ids[i])
                tid = str(int_ids[i])
                self._id_to_text[tid]     = texts[i]
                self._id_to_metadata[tid] = items[i][2] or {}

        if valid_vecs:
            vecs_arr = np.array(valid_vecs, dtype=np.float32)
            ids_arr  = np.array(valid_ids,  dtype=np.int64)

            if self.use_faiss:
                idx = self._get_faiss_index()
                idx.add_with_ids(vecs_arr, ids_arr)
            else:
                self._vectors.extend(valid_vecs)
                self._ids.extend(valid_ids)   # plain Python list — no .tolist() needed

        logger.info(f"[LocalEmbedder] Batch indexed {len(valid_vecs)}/{len(items)} items")
        return len(valid_vecs)

    async def similarity_search(self, query: str, top_k: int = 5) -> List[SimilarityHit]:
        result = await self.embed(query)
        if not result.success or result.vector is None or not _NUMPY_AVAILABLE:
            return []

        # FIX 6: normalize as 1D, then reshape for FAISS
        qvec_1d = self._normalize_1d(np.array(result.vector, dtype=np.float32))
        qvec    = qvec_1d.reshape(1, -1)

        if self.use_faiss and self._index is not None and self._index.ntotal > 0:
            k = min(top_k, self._index.ntotal)
            scores, indices = self._index.search(qvec, k)
            hits = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    tid = str(idx)
                    hits.append(SimilarityHit(
                        id=tid,
                        text=self._id_to_text.get(tid, ""),
                        score=float(score),
                        metadata=self._id_to_metadata.get(tid, {}),
                    ))
            return hits

        # Numpy fallback
        if not self._vectors:
            return []
        vecs    = np.array(self._vectors, dtype=np.float32)
        sims    = np.dot(vecs, qvec_1d).flatten()
        k       = min(top_k, len(sims))
        top_idx = np.argsort(sims)[-k:][::-1]
        return [
            SimilarityHit(
                id=str(self._ids[i]),
                text=self._id_to_text.get(str(self._ids[i]), ""),
                score=float(sims[i]),
                metadata=self._id_to_metadata.get(str(self._ids[i]), {}),
            )
            for i in top_idx
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def save_index(self, path: str) -> bool:
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "id_to_text":     self._id_to_text,
                    "id_to_metadata": self._id_to_metadata,
                    "model_name":     self.model_name,
                    "dimension":      self._dimension,
                }, f, ensure_ascii=False)
            if self.use_faiss and self._index is not None and faiss is not None:
                faiss.write_index(self._index, str(save_path) + ".faiss")
            elif self._vectors and _NUMPY_AVAILABLE:
                np.save(str(save_path) + "_vectors.npy", np.array(self._vectors))
                with open(str(save_path) + "_ids.json", "w") as f:
                    json.dump(self._ids, f)
            logger.info(f"[LocalEmbedder] Saved to {path}")
            return True
        except Exception as exc:
            logger.error(f"[LocalEmbedder] Save failed: {exc}")
            return False

    async def load_index(self, path: str) -> bool:
        try:
            load_path = Path(path)
            if not load_path.exists():
                return False
            with open(load_path, encoding="utf-8") as f:
                data = json.load(f)
            self._id_to_text     = data["id_to_text"]
            self._id_to_metadata = data.get("id_to_metadata", {})
            self.model_name      = data.get("model_name", self.model_name)
            self._dimension      = data.get("dimension", self._dimension)

            faiss_path = str(load_path) + ".faiss"
            if self.use_faiss and Path(faiss_path).exists() and faiss is not None:
                self._index = faiss.read_index(faiss_path)

            vec_path = str(load_path) + "_vectors.npy"
            ids_path = str(load_path) + "_ids.json"
            if Path(vec_path).exists() and _NUMPY_AVAILABLE:
                self._vectors = list(np.load(vec_path))
                with open(ids_path) as f:
                    self._ids = json.load(f)

            logger.info(f"[LocalEmbedder] Loaded from {path} | size={len(self._id_to_text)}")
            return True
        except Exception as exc:
            logger.error(f"[LocalEmbedder] Load failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # Metrics and lifecycle
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "name":          "local_embeddings",
            "model":         self.model_name,
            "dimension":     self._dimension,
            "cache_size":    len(self._cache),
            "index_size":    len(self._id_to_text),
            "faiss_enabled": self.use_faiss and self._index is not None,
            "calls":         self._call_count,
            "numpy":         _NUMPY_AVAILABLE,
            "sentence_transformers": _ST_AVAILABLE,
        }

    def shutdown(self) -> None:
        """Clean up thread pool."""
        self._executor.shutdown(wait=False)

    def __repr__(self) -> str:
        return (
            f"LocalEmbedder(model={self.model_name!r}, "
            f"cache={len(self._cache)}, index={len(self._id_to_text)}, "
            f"faiss={self.use_faiss})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_embedder(
    model_name: str  = "all-MiniLM-L6-v2",
    use_faiss:  bool = True,
    **kwargs,
) -> LocalEmbedder:
    return LocalEmbedder(model_name=model_name, use_faiss=use_faiss, **kwargs)


async def embed_text(text: str) -> EmbeddingResult:
    """One-shot helper — creates a fresh embedder each call (prefer create_embedder for reuse)."""
    embedder = create_embedder()
    return await embedder.embed(text)
