# keryx/graph.py
# Vulnerability / Dependency Graph for KeryxHunter — production edition.
# Directed graph of files, functions, call sites, hypotheses, crashes.
# Personalized PageRank, attack-path discovery, JSON/GraphML export.
# Sovereign, air-gapped fallback, SharedContext evidence integration.

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from .core.shared_context import SharedContext
except ImportError:
    SharedContext = Any  # type: ignore

try:
    import networkx as nx
    _NETWORKX_AVAILABLE = True
except ImportError:
    _NETWORKX_AVAILABLE = False
    nx = None  # type: ignore

logger = logging.getLogger("keryx.graph")


# ---------------------------------------------------------------------------
# C++ keyword blacklist for function detection
# FIX 4: extended list
# ---------------------------------------------------------------------------
_CPP_KEYWORDS = frozenset({
    "if", "else", "while", "for", "switch", "case", "return",
    "try", "catch", "throw", "new", "delete", "do", "goto",
    "static_assert", "sizeof", "alignof", "decltype", "auto",
})

_JS_KEYWORDS = frozenset({
    "if", "else", "while", "for", "switch", "return",
    "try", "catch", "throw", "new", "delete", "do", "await",
})


# ---------------------------------------------------------------------------
# JSON-safe serialisation helper
# FIX 8: metadata may contain non-serializable objects
# ---------------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively make an object JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (obj != obj):   # NaN
        return None
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    id:         str
    type:       str    # file | function | hypothesis | crash | evidence
    label:      str
    file:       Optional[str]        = None
    line:       Optional[int]        = None
    risk_score: float                = 0.0
    metadata:   Dict[str, Any]       = field(default_factory=dict)


@dataclass
class GraphEdge:
    source:   str
    target:   str
    type:     str    # contains | calls | dataflow | hypothesis_link | crash_link
    weight:   float              = 1.0
    metadata: Dict[str, Any]    = field(default_factory=dict)


@dataclass
class GraphAnalysisResult:
    success:          bool
    nodes:            int                   = 0
    edges:            int                   = 0
    top_risky:        List[str]             = field(default_factory=list)
    attack_paths:     List[List[str]]       = field(default_factory=list)
    execution_time_ms: float               = 0.0
    data:             Dict[str, Any]        = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Language parsers
# FIX 4: extended keyword blacklist, tighter patterns
# ---------------------------------------------------------------------------

class CodeParser:

    @staticmethod
    def extract_functions_cpp(content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract function definitions from C/C++ — excludes control flow keywords."""
        functions = []
        # Match: optional_return_type function_name(...)
        pattern = re.compile(
            r'^\s*(?:[\w:<>*&]+\s+)+(\w+)\s*\([^;)]{0,200}\)\s*(?:const\s*)?(?:override\s*)?[{;]'
        )
        for line_num, line in enumerate(content.splitlines(), 1):
            m = pattern.match(line)
            if not m:
                continue
            func_name = m.group(1)
            if func_name in _CPP_KEYWORDS or func_name[0].isupper() and not func_name[1:2].islower():
                # Skip constructors/destructors heuristically if desired
                pass
            if func_name not in _CPP_KEYWORDS:
                functions.append({
                    "name":        func_name,
                    "line":        line_num,
                    "signature":   line.strip()[:200],
                    "return_type": "",
                })
        return functions

    @staticmethod
    def extract_functions_js(content: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract function definitions from JavaScript/TypeScript."""
        functions = []
        patterns = [
            re.compile(r'\bfunction\s+(\w+)\s*\('),
            re.compile(r'\b(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>'),
            re.compile(r'\b(?:async\s+)?(\w+)\s*\([^)]{0,100}\)\s*\{'),
        ]
        for line_num, line in enumerate(content.splitlines(), 1):
            for pat in patterns:
                for m in pat.finditer(line):
                    name = m.group(1)
                    if name and name not in _JS_KEYWORDS:
                        functions.append({
                            "name":      name,
                            "line":      line_num,
                            "signature": line.strip()[:200],
                        })
                        break  # one match per line per pattern is enough


        return functions


# ---------------------------------------------------------------------------
# VulnerabilityGraph
# ---------------------------------------------------------------------------

class VulnerabilityGraph:
    """
    Main graph engine.

    NetworkX DiGraph when available (fast PageRank, shortest paths).
    Pure-Python BFS/risk-sort fallback for air-gapped environments.
    """

    SUPPORTED_EXTENSIONS = {
        '.cpp', '.c', '.h', '.hpp', '.cc', '.cxx',
        '.js', '.mjs', '.cjs',
        '.rs',
        '.py',
        '.idl', '.ipdl', '.ipdlh',
    }

    def __init__(self, enable_networkx: bool = True) -> None:
        self.enable_networkx = enable_networkx and _NETWORKX_AVAILABLE
        self.nodes:      Dict[str, GraphNode] = {}
        self.edges:      Dict[str, GraphEdge] = {}
        self.hypotheses: Dict[str, Dict]      = {}
        self._call_count = 0
        self._parser     = CodeParser()

        self._nx_graph = nx.DiGraph() if (self.enable_networkx and nx is not None) else None

        # FIX 7: adjacency list for O(1) edge lookup in BFS (no networkx dependency)
        self._adj: Dict[str, List[str]] = {}   # source_id → [target_id, ...]

        logger.info(f"[VulnerabilityGraph] Initialized | networkx={self.enable_networkx}")

    # ------------------------------------------------------------------
    # Core graph operations
    # ------------------------------------------------------------------

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node
        if node.id not in self._adj:
            self._adj[node.id] = []
        if self.enable_networkx and self._nx_graph is not None:
            # FIX 6: pass only scalar attributes to nx — no nested dicts/objects
            self._nx_graph.add_node(
                node.id,
                label=node.label,
                type=node.type,
                risk=node.risk_score,
            )

    def add_edge(self, edge: GraphEdge) -> None:
        # FIX 3: use | as separator — avoids unicode arrow collision in node IDs
        edge_key = f"{edge.source}|{edge.target}|{edge.type}"
        self.edges[edge_key] = edge

        # Maintain adjacency list for BFS
        if edge.source not in self._adj:
            self._adj[edge.source] = []
        if edge.target not in self._adj[edge.source]:
            self._adj[edge.source].append(edge.target)

        if self.enable_networkx and self._nx_graph is not None:
            self._nx_graph.add_edge(
                edge.source, edge.target,
                type=edge.type,
                weight=edge.weight,
            )

    # FIX 1: define get_edges_from — was called in _bfs_find_paths but missing
    def get_edges_from(self, node_id: str) -> List[GraphEdge]:
        """Return all outgoing edges from a node."""
        return [e for e in self.edges.values() if e.source == node_id]

    def get_neighbors(self, node_id: str) -> List[str]:
        """O(1) neighbor lookup via adjacency list."""
        return self._adj.get(node_id, [])

    # ------------------------------------------------------------------
    # Graph building
    # FIX 5: offload blocking directory scan to run_in_executor
    # ------------------------------------------------------------------

    def build_from_directory(self, root: str) -> GraphAnalysisResult:
        """
        Build graph from source directory.
        NOTE: call via build_graph_from_target() for async-safe execution.
        This method is synchronous and should not be called directly from
        an async event loop on large codebases.
        """
        start     = time.time()
        root_path = Path(root)

        if not root_path.exists():
            return GraphAnalysisResult(
                success=False,
                data={"error": f"Directory not found: {root}"},
            )

        files_scanned    = 0
        functions_found  = 0

        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in root_path.rglob(f"*{ext}"):
                try:
                    rel     = str(file_path.relative_to(root_path))
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    file_id = f"file:{rel}"

                    self.add_node(GraphNode(
                        id=file_id, type="file", label=rel,
                        file=str(file_path), risk_score=0.3,
                        metadata={"size": len(content), "ext": ext},
                    ))
                    files_scanned += 1

                    if ext in {'.cpp', '.c', '.h', '.hpp', '.cc', '.cxx'}:
                        funcs = self._parser.extract_functions_cpp(content, rel)
                    elif ext in {'.js', '.mjs', '.cjs'}:
                        funcs = self._parser.extract_functions_js(content, rel)
                    else:
                        funcs = []

                    for func in funcs:
                        func_id = f"func:{rel}:{func['name']}"
                        self.add_node(GraphNode(
                            id=func_id, type="function",
                            label=f"{func['name']}()",
                            file=rel, line=func.get("line"),
                            risk_score=0.5,
                            metadata={"signature": func.get("signature", "")},
                        ))
                        self.add_edge(GraphEdge(
                            source=file_id, target=func_id, type="contains",
                        ))
                        functions_found += 1

                except Exception as exc:
                    logger.debug(f"[Graph] Failed to parse {file_path}: {exc}")

        self._call_count += 1
        duration_ms = (time.time() - start) * 1000
        logger.info(
            f"[Graph] Built from {root}: "
            f"{files_scanned} files, {functions_found} functions | {duration_ms:.0f}ms"
        )
        return GraphAnalysisResult(
            success=True,
            nodes=len(self.nodes),
            edges=len(self.edges),
            execution_time_ms=duration_ms,
            data={"files_scanned": files_scanned, "functions_found": functions_found},
        )

    # ------------------------------------------------------------------
    # Hypothesis and crash injection
    # ------------------------------------------------------------------

    def add_hypothesis(self, hyp_id: str, nodes: List[str], risk: float = 0.8) -> None:
        """Inject hypothesis as PageRank seed — boosts connected node scores."""
        self.hypotheses[hyp_id] = {"nodes": nodes, "base_risk": risk}
        self.add_node(GraphNode(
            id=hyp_id, type="hypothesis",
            label=f"Hypothesis {hyp_id[:8]}",
            risk_score=risk,
            metadata={"timestamp": time.time()},
        ))
        for nid in nodes:
            if nid in self.nodes:
                self.add_edge(GraphEdge(
                    source=hyp_id, target=nid,
                    type="hypothesis_link", weight=risk,
                ))
                self.nodes[nid].risk_score = max(self.nodes[nid].risk_score, risk * 0.7)

    def add_crash(
        self,
        crash_id:          str,
        crash_info:        Dict[str, Any],
        related_functions: List[str],
    ) -> None:
        self.add_node(GraphNode(
            id=crash_id, type="crash",
            label=f"Crash: {crash_info.get('crash_type', 'unknown')}",
            risk_score=0.9,
            metadata={
                "crash_type":    crash_info.get("crash_type", "unknown"),
                "stack_trace":   crash_info.get("stack_trace", [])[:10],
                "fault_address": crash_info.get("fault_address"),
                "timestamp":     time.time(),
            },
        ))
        for func_name in related_functions:
            for nid, node in self.nodes.items():
                if node.type == "function" and func_name in node.label:
                    self.add_edge(GraphEdge(
                        source=nid, target=crash_id, type="crash_link",
                    ))
                    self.nodes[nid].risk_score = max(self.nodes[nid].risk_score, 0.85)
                    break

    # ------------------------------------------------------------------
    # Ranking and path finding
    # ------------------------------------------------------------------

    def rank_nodes_by_risk(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """Personalized PageRank with hypothesis risk as seeds, BFS fallback."""
        if self.enable_networkx and self._nx_graph is not None and len(self.nodes) > 1:
            try:
                personalization = {nid: n.risk_score for nid, n in self.nodes.items()}
                scores = nx.pagerank(
                    self._nx_graph, personalization=personalization, weight="weight"
                )
                return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
            except Exception as exc:
                # FIX 9: log fallback reason
                logger.debug(f"[Graph] PageRank failed ({exc}) — using risk-score sort")

        return sorted(
            ((nid, n.risk_score) for nid, n in self.nodes.items()),
            key=lambda x: x[1], reverse=True,
        )[:top_n]

    def find_attack_paths(
        self,
        source_node: str,
        target_node: str,
        max_paths:   int = 5,
    ) -> List[List[str]]:
        if source_node not in self.nodes or target_node not in self.nodes:
            return []
        if self.enable_networkx and self._nx_graph is not None:
            try:
                return list(nx.all_shortest_paths(
                    self._nx_graph, source_node, target_node
                ))[:max_paths]
            except (Exception,):
                pass
        return self._bfs_find_paths(source_node, target_node, max_paths)

    def _bfs_find_paths(
        self, source: str, target: str, max_paths: int
    ) -> List[List[str]]:
        """Pure-Python BFS using adjacency list for air-gapped mode."""
        paths: List[List[str]] = []
        queue: deque = deque([(source, [source])])

        while queue and len(paths) < max_paths:
            node, path = queue.popleft()
            if node == target:
                paths.append(path)
                continue
            # FIX 1: use get_neighbors (adjacency list) not get_edges_from
            for neighbor in self.get_neighbors(node):
                if neighbor not in path:   # cycle guard
                    queue.append((neighbor, path + [neighbor]))

        return paths

    def find_high_risk_paths(
        self, min_risk: float = 0.7, max_length: int = 5
    ) -> List[List[str]]:
        """
        Find paths where all nodes are high-risk.
        FIX 2: limit source+target pairs to avoid O(N²) explosion.
        """
        high_risk = [
            nid for nid, n in self.nodes.items()
            if n.risk_score >= min_risk
        ][:50]   # hard cap on candidates

        paths: List[List[str]] = []
        seen: set = set()

        for i, source in enumerate(high_risk):
            for target in high_risk[i + 1:]:   # upper triangle only
                found = self.find_attack_paths(source, target, max_paths=1)
                for p in found:
                    key = tuple(p)
                    if key not in seen and len(p) <= max_length:
                        seen.add(key)
                        paths.append(p)

        return paths[:20]

    # ------------------------------------------------------------------
    # Serialisation
    # FIX 8: _json_safe applied to metadata before dump
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "nodes": {
                nid: {
                    "type":     n.type,
                    "label":    n.label,
                    "file":     n.file,
                    "line":     n.line,
                    "risk":     n.risk_score,
                    "metadata": _json_safe(n.metadata),
                }
                for nid, n in self.nodes.items()
            },
            "edges": [
                {
                    "source":   e.source,
                    "target":   e.target,
                    "type":     e.type,
                    "weight":   e.weight,
                    "metadata": _json_safe(e.metadata),
                }
                for e in self.edges.values()
            ],
            "metrics": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "calls_made": self._call_count,
            },
        }

    def save(self, path: str, format: str = "json") -> bool:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if format == "graphml" and self.enable_networkx and self._nx_graph is not None:
                nx.write_graphml(self._nx_graph, path)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as exc:
            logger.error(f"[Graph] Save failed: {exc}")
            return False

    def load(self, path: str) -> bool:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            self.nodes.clear()
            self.edges.clear()
            self._adj.clear()
            if self._nx_graph is not None:
                self._nx_graph.clear()

            for nid, nd in data.get("nodes", {}).items():
                self.add_node(GraphNode(
                    id=nid, type=nd["type"], label=nd["label"],
                    file=nd.get("file"), line=nd.get("line"),
                    risk_score=nd.get("risk", 0.0),
                    metadata=nd.get("metadata", {}),
                ))
            for ed in data.get("edges", []):
                self.add_edge(GraphEdge(
                    source=ed["source"], target=ed["target"],
                    type=ed["type"], weight=ed.get("weight", 1.0),
                    metadata=ed.get("metadata", {}),
                ))
            return True
        except Exception as exc:
            logger.error(f"[Graph] Load failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # SharedContext integration
    # FIX 7: store graph findings as evidence fragments
    # ------------------------------------------------------------------

    def sync_to_context(self, context: Any) -> None:
        """
        Push high-risk nodes and confirmed crashes into SharedContext
        evidence memory so advisors can recall them via keyword search.
        """
        if not hasattr(context, "add_evidence"):
            return

        for nid, node in self.nodes.items():
            if node.risk_score >= 0.75:
                keywords = [node.type]
                if node.file:
                    keywords.append(Path(node.file).stem)
                context.add_evidence(
                    pointer=f"graph:{nid}",
                    observation_fragment=(
                        f"{node.type} {node.label!r} "
                        f"risk={node.risk_score:.2f} file={node.file}:{node.line}"
                    ),
                    keywords=keywords,
                )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "nodes":            len(self.nodes),
            "edges":            len(self.edges),
            "networkx_enabled": self.enable_networkx,
            "calls":            self._call_count,
            "node_types":       self._count_node_types(),
        }

    def _count_node_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n in self.nodes.values():
            counts[n.type] = counts.get(n.type, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"VulnerabilityGraph("
            f"nodes={len(self.nodes)}, "
            f"edges={len(self.edges)}, "
            f"networkx={self.enable_networkx})"
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def create_vulnerability_graph(enable_networkx: bool = True) -> VulnerabilityGraph:
    return VulnerabilityGraph(enable_networkx=enable_networkx)


async def build_graph_from_target(
    target_path: str,
    context:     Optional[Any] = None,
) -> GraphAnalysisResult:
    """
    Async-safe graph build — offloads blocking I/O to thread pool.
    FIX 5: build_from_directory is sync; calling it directly from async
    would block the event loop for large codebases.
    """
    import asyncio
    graph = create_vulnerability_graph()

    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, graph.build_from_directory, target_path
    )

    # FIX 7: sync high-risk findings to SharedContext evidence memory
    if context is not None:
        graph.sync_to_context(context)

    if context is not None and hasattr(context, "add_step"):
        class _Action:
            action       = "build_graph"
            action_input = {"target": target_path}
            thought      = ""
            confidence   = 1.0

        context.add_step(
            _Action(),
            f"Graph built: {result.nodes} nodes, {result.edges} edges, "
            f"{result.data.get('files_scanned', 0)} files",
        )

    return result
  
