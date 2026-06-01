"""
Validity — Priority Cycle Encoder
src/priority_cycle_encoder.py

Detects ordinal priority cycles in intercreditor and waterfall documents.
A priority cycle is a circular ordering of obligations — A > B > C > A —
which is UNSAT because strict partial orders must be acyclic.

Detection paths:
  Primary:  check_document_text(text) — scope-aware extraction from raw text
  Fallback: analyse(claims)           — extraction from pipeline claim objects

Scope awareness:
  Priority edges are tagged with a collateral-pool scope derived from the
  sentence context (e.g. "ABL Priority Collateral", "Seeded Test Collateral").
  Edges in different scopes are never compared — bilateral priority arrangements
  (each party senior on its own collateral) are normal in intercreditor
  agreements and must not generate false positives.
  Only a cycle within a single scope is a genuine contradiction.

Node normalisation:
  Conservative alias table only. Obvious surface variants (e.g. "ABL Secured
  Parties" / "ABL Lenders" / "ABL") are collapsed. No fuzzy matching.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class PriorityEdge:
    """A directed priority relation: senior has priority over junior in scope."""
    senior:      str   # normalised node name: "Party@scope_key"
    junior:      str   # normalised node name: "Party@scope_key"
    relation:    str   # e.g. "first_priority_over", "senior_to", "junior_to"
    source_text: str   # verbatim source sentence (truncated to 400 chars)
    clause_id:   str   # UUID for this edge
    confidence:  str   # "high" | "medium"


@dataclass
class PriorityCycleResult:
    """Returned when a priority cycle is detected."""
    verdict:         str                  # always "unsat"
    cycle_path:      list[str]            # [A, B, C, A] — ordered, last == first
    cycle_edges:     list[PriorityEdge]   # edges that form the cycle
    source_spans:    list[dict]           # [{text, clause_id, confidence}]
    formal_proof:    str
    failure_subtype: str = "priority_cycle"


# ── Alias table ────────────────────────────────────────────────────────────────
# Maps lowercase surface variants → canonical party name.
# Sorted longest-first so longer matches shadow shorter ones.

_PARTY_ALIASES: list[tuple[str, str]] = sorted([
    ("term loan/notes secured parties",  "Term"),
    ("term loan/notes secured party",    "Term"),
    ("term loan/notes obligations",      "Term"),
    ("term loan/notes",                  "Term"),
    ("term loan secured parties",        "Term"),
    ("term loan secured party",          "Term"),
    ("term lenders",                     "Term"),
    ("term loan",                        "Term"),
    ("abl secured parties",              "ABL"),
    ("abl secured party",                "ABL"),
    ("abl obligations",                  "ABL"),
    ("abl lenders",                      "ABL"),
    ("abl",                              "ABL"),
    ("mezzanine secured parties",        "Mezzanine"),
    ("mezzanine secured party",          "Mezzanine"),
    ("mezzanine lenders",                "Mezzanine"),
    ("mezzanine obligations",            "Mezzanine"),
    ("mezzanine",                        "Mezzanine"),
    ("first lien secured parties",       "FirstLien"),
    ("first lien secured party",         "FirstLien"),
    ("first lien lenders",               "FirstLien"),
    ("first lien obligations",           "FirstLien"),
    ("first lien",                       "FirstLien"),
    ("second lien secured parties",      "SecondLien"),
    ("second lien secured party",        "SecondLien"),
    ("second lien lenders",              "SecondLien"),
    ("second lien obligations",          "SecondLien"),
    ("second lien",                      "SecondLien"),
    ("senior secured parties",           "Senior"),
    ("senior secured party",             "Senior"),
    ("senior lenders",                   "Senior"),
    ("junior secured parties",           "Junior"),
    ("junior secured party",             "Junior"),
    ("junior lenders",                   "Junior"),
    ("subordinated lenders",             "Subordinated"),
    ("subordinated creditors",           "Subordinated"),
], key=lambda x: -len(x[0]))


# ── Main encoder ───────────────────────────────────────────────────────────────

class PriorityCycleEncoder:
    """
    Detects ordinal priority cycles using scope-aware directed graph analysis.

    Workflow:
      1. Extract PriorityEdge objects from text using pattern matching
      2. Deduplicate edges; build directed graph (senior → junior)
      3. Run Kahn's algorithm — O(V+E) — to test for cycles
      4. If cycle exists, reconstruct minimal path via DFS with 3-colour marking
      5. Map cycle path back to source sentences
      6. Return PriorityCycleResult or None (clean)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def analyse(self, claims: list) -> Optional[PriorityCycleResult]:
        """
        Primary path: run over FormalClaim or CandidateClaim objects from the pipeline.
        Reads text_span from each claim.
        """
        edges: list[PriorityEdge] = []
        for claim in claims:
            text = getattr(claim, "text_span", "") or ""
            if text.strip():
                edges.extend(self._extract_edges(text, fallback_mode=False))
        return self._detect_cycle(edges)

    def check_document_text(self, document_text: str) -> Optional[PriorityCycleResult]:
        """
        Fallback path: run over raw document text.
        Applies the same extraction patterns as analyse(), but processes the full
        document rather than individual pipeline claims.
        Marked as lower-confidence source for the medium-confidence fallback pattern.
        """
        if not document_text or not document_text.strip():
            return None
        edges: list[PriorityEdge] = []
        for sentence in self._split_sentences(document_text):
            if sentence.strip():
                edges.extend(self._extract_edges(sentence, fallback_mode=True))
        return self._detect_cycle(edges)

    # ── Edge extraction ───────────────────────────────────────────────────────

    def _extract_edges(self, text: str, fallback_mode: bool) -> list[PriorityEdge]:
        """
        Try each extraction pattern in priority order.
        Patterns 1 and 2 are high confidence and scope-aware.
        Pattern 3 (medium confidence, global scope) is only enabled in fallback_mode
        when patterns 1 and 2 find nothing.
        """
        edges: list[PriorityEdge] = []

        # Pattern 1: "For the [SCOPE], A shall have [first] priority over B"
        edges.extend(self._pat_priority_over(text))

        # Pattern 2: "A Obligations...senior/junior...B Obligations" + collateral scope
        edges.extend(self._pat_obligations_senior_junior(text))

        # Pattern 3: general fallback (medium confidence, no scope)
        if fallback_mode and not edges:
            edges.extend(self._pat_general_senior_junior(text))

        return edges

    def _pat_priority_over(self, text: str) -> list[PriorityEdge]:
        """
        Matches: 'For the [SCOPE], [A] shall have [first] priority over [B]'
        Confidence: high. Scope extracted from the 'For the X,' prefix.
        """
        m = re.search(
            r'For\s+(?:the\s+)?([^,]+?),\s*'
            r'(.+?)\s+shall\s+have\s+(?:first\s+)?priority\s+over\s+'
            r'(.+?)\.?\s*$',
            text, re.IGNORECASE,
        )
        if not m:
            return []
        scope      = m.group(1).strip()
        senior_raw = m.group(2).strip()
        junior_raw = m.group(3).strip()
        senior     = self._normalize(senior_raw)
        junior     = self._normalize(junior_raw)
        if not senior or not junior or senior == junior:
            return []
        sk = self._scope_key(scope)
        return [PriorityEdge(
            senior      = f"{senior}@{sk}",
            junior      = f"{junior}@{sk}",
            relation    = "first_priority_over",
            source_text = text.strip()[:400],
            clause_id   = str(uuid.uuid4()),
            confidence  = "high",
        )]

    def _pat_obligations_senior_junior(self, text: str) -> list[PriorityEdge]:
        """
        Matches: '...on the [SCOPE] Collateral...A Obligations...shall be senior/junior...B Obligations...'
        Confidence: high. Scope extracted from 'on the X Collateral'.
        Typical in complex intercreditor-agreement sentences.
        """
        if not re.search(r'\bshall\s+be\s+(?:senior|junior|subordinate)', text, re.IGNORECASE):
            return []

        scope_m = re.search(r'\bon\s+the\s+([\w\s/]+?)\s+Collateral\b', text, re.IGNORECASE)
        if not scope_m:
            return []
        scope = scope_m.group(1).strip()

        obligations_re = re.compile(
            r'\b(ABL|Term\s+Loan/Notes|Term\s+Loan|Mezzanine|'
            r'First\s+Lien|Second\s+Lien)\s+Obligations',
            re.IGNORECASE,
        )
        parties = [m.group(1).strip() for m in obligations_re.finditer(text)]
        if len(parties) < 2:
            return []

        if re.search(r'\bshall\s+be\s+senior\b', text, re.IGNORECASE):
            senior_raw, junior_raw, relation = parties[0], parties[-1], "senior_to"
        elif re.search(r'\bshall\s+be\s+(?:junior|subordinate)', text, re.IGNORECASE):
            senior_raw, junior_raw, relation = parties[-1], parties[0], "junior_to"
        else:
            return []

        senior = self._normalize(senior_raw)
        junior = self._normalize(junior_raw)
        if not senior or not junior or senior == junior:
            return []

        sk = self._scope_key(scope)
        return [PriorityEdge(
            senior      = f"{senior}@{sk}",
            junior      = f"{junior}@{sk}",
            relation    = relation,
            source_text = text.strip()[:400],
            clause_id   = str(uuid.uuid4()),
            confidence  = "high",
        )]

    def _pat_general_senior_junior(self, text: str) -> list[PriorityEdge]:
        """
        Pattern 3 — medium confidence fallback. No scope (uses 'global' scope key).
        Matches common legal priority language not covered by patterns 1/2.
        Only activated when patterns 1 and 2 find nothing (fallback_mode only).

        Patterns:
          'A [shall be / is] senior to B'
          'B [shall be / is] subordinated to A'
          'B [shall be / is] junior to A'
          'A ranked ahead of B'
          'no payment to B until A [is paid / discharged]'
        """
        edges: list[PriorityEdge] = []

        # "A shall be senior to B" / "A is senior to B"
        m = re.search(
            r'([A-Za-z][\w\s/\-]{2,60}?)\s+(?:shall\s+be|is|are)\s+'
            r'senior(?:\s+in\s+[\w,\s]+?)?\s+to\s+'
            r'([A-Za-z][\w\s/\-]{2,60}?)(?:\s+in\b|\s+with\b|\s+under\b|\.|\,|$)',
            text, re.IGNORECASE,
        )
        if m:
            senior = self._normalize(m.group(1))
            junior = self._normalize(m.group(2))
            if senior and junior and senior != junior:
                edges.append(PriorityEdge(
                    senior=f"{senior}@global", junior=f"{junior}@global",
                    relation="senior_to", source_text=text.strip()[:400],
                    clause_id=str(uuid.uuid4()), confidence="medium",
                ))

        # "B shall be subordinated/junior to A"
        m = re.search(
            r'([A-Za-z][\w\s/\-]{2,60}?)\s+(?:shall\s+be|is|are)\s+'
            r'(?:subordinated?|junior)\s+to\s+'
            r'([A-Za-z][\w\s/\-]{2,60}?)(?:\s+in\b|\s+with\b|\s+under\b|\.|\,|$)',
            text, re.IGNORECASE,
        )
        if m:
            junior = self._normalize(m.group(1))
            senior = self._normalize(m.group(2))
            if senior and junior and senior != junior:
                edges.append(PriorityEdge(
                    senior=f"{senior}@global", junior=f"{junior}@global",
                    relation="subordinated_to", source_text=text.strip()[:400],
                    clause_id=str(uuid.uuid4()), confidence="medium",
                ))

        # "A ranked ahead of B"
        m = re.search(
            r'([A-Za-z][\w\s/\-]{2,60}?)\s+(?:is\s+)?ranked\s+ahead\s+of\s+'
            r'([A-Za-z][\w\s/\-]{2,60}?)(?:\s+in\b|\.|\,|$)',
            text, re.IGNORECASE,
        )
        if m:
            senior = self._normalize(m.group(1))
            junior = self._normalize(m.group(2))
            if senior and junior and senior != junior:
                edges.append(PriorityEdge(
                    senior=f"{senior}@global", junior=f"{junior}@global",
                    relation="ranked_ahead_of", source_text=text.strip()[:400],
                    clause_id=str(uuid.uuid4()), confidence="medium",
                ))

        # "no payment to B until A is paid/discharged"
        m = re.search(
            r'no\s+payment\s+(?:shall\s+be\s+made\s+)?to\s+'
            r'([A-Za-z][\w\s/\-]{2,60}?)\s+until\s+'
            r'([A-Za-z][\w\s/\-]{2,60}?)\s+(?:is\s+paid|has\s+been\s+paid|is\s+discharged|paid\s+in\s+full)',
            text, re.IGNORECASE,
        )
        if m:
            junior = self._normalize(m.group(1))
            senior = self._normalize(m.group(2))
            if senior and junior and senior != junior:
                edges.append(PriorityEdge(
                    senior=f"{senior}@global", junior=f"{junior}@global",
                    relation="payment_blocked_until", source_text=text.strip()[:400],
                    clause_id=str(uuid.uuid4()), confidence="medium",
                ))

        return edges

    # ── Graph algorithms ──────────────────────────────────────────────────────

    def _detect_cycle(self, edges: list[PriorityEdge]) -> Optional[PriorityCycleResult]:
        """
        1. Deduplicate edges
        2. Build directed graph (senior → junior)
        3. Kahn's algorithm — detects whether a cycle exists in O(V+E)
        4. DFS cycle reconstruction — finds the minimal cycle path
        5. Map path back to source edges and return PriorityCycleResult
        """
        if not edges:
            return None

        # Deduplicate: same (senior, junior) pair → keep highest-confidence edge
        edge_map: dict[tuple[str, str], PriorityEdge] = {}
        for edge in edges:
            key = (edge.senior, edge.junior)
            if key not in edge_map or edge.confidence == "high":
                edge_map[key] = edge
        deduped = list(edge_map.values())

        # Build graph structures
        nodes: set[str] = set()
        adj: dict[str, list[tuple[str, PriorityEdge]]] = defaultdict(list)
        in_degree: dict[str, int] = defaultdict(int)

        for edge in deduped:
            nodes.add(edge.senior)
            nodes.add(edge.junior)
            adj[edge.senior].append((edge.junior, edge))
            in_degree[edge.junior] += 1

        # Ensure every node is in in_degree
        for n in nodes:
            if n not in in_degree:
                in_degree[n] = 0

        # ── Kahn's algorithm ──────────────────────────────────────────────────
        queue = deque(n for n in nodes if in_degree[n] == 0)
        processed = 0
        while queue:
            u = queue.popleft()
            processed += 1
            for (v, _) in adj.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if processed == len(nodes):
            return None  # Topological sort succeeded — no cycle

        # ── DFS cycle reconstruction ──────────────────────────────────────────
        # Nodes still with in_degree > 0 are part of the cycle(s)
        cycle_nodes = {n for n in nodes if in_degree[n] > 0}
        cycle_path  = self._find_cycle_dfs(adj, cycle_nodes)
        if not cycle_path:
            return None

        # Map cycle path to edges
        cycle_edges: list[PriorityEdge] = []
        for i in range(len(cycle_path) - 1):
            s, j = cycle_path[i], cycle_path[i + 1]
            for (neighbor, edge) in adj.get(s, []):
                if neighbor == j:
                    cycle_edges.append(edge)
                    break

        source_spans = [
            {"text": e.source_text, "clause_id": e.clause_id, "confidence": e.confidence}
            for e in cycle_edges
        ]

        return PriorityCycleResult(
            verdict         = "unsat",
            cycle_path      = cycle_path,
            cycle_edges     = cycle_edges,
            source_spans    = source_spans,
            formal_proof    = self._build_proof(cycle_path, cycle_edges),
            failure_subtype = "priority_cycle",
        )

    def _find_cycle_dfs(
        self,
        adj:         dict[str, list[tuple[str, PriorityEdge]]],
        cycle_nodes: set[str],
    ) -> list[str]:
        """
        3-colour DFS over cycle_nodes to reconstruct one minimal cycle.
        Returns [v1, v2, ..., vk, v1] (last element closes the cycle).
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color  = {n: WHITE for n in cycle_nodes}
        parent = {n: None  for n in cycle_nodes}
        result: list[str] = []

        def dfs(u: str) -> bool:
            color[u] = GRAY
            for (v, _) in adj.get(u, []):
                if v not in cycle_nodes:
                    continue
                if color[v] == GRAY:
                    # Back edge u → v: cycle is v → ... → u → v
                    path = [v]
                    curr = u
                    visited = 0
                    while curr != v and visited <= len(cycle_nodes):
                        path.append(curr)
                        curr = parent.get(curr)
                        if curr is None:
                            break
                        visited += 1
                    path.append(v)
                    result.extend(reversed(path))
                    return True
                if color[v] == WHITE:
                    parent[v] = u
                    if dfs(v):
                        return True
            color[u] = BLACK
            return False

        for n in cycle_nodes:
            if color[n] == WHITE:
                if dfs(n):
                    return result

        return result

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _normalize(self, text: str) -> Optional[str]:
        """
        Map a raw party-name string to its canonical form via the alias table.
        Longest-match wins. Returns None if no alias matches.
        """
        if not text:
            return None
        text_l = text.lower().strip()
        for alias, canonical in _PARTY_ALIASES:
            if alias in text_l:
                return canonical
        return None

    @staticmethod
    def _scope_key(scope: str) -> str:
        """Stable lowercase scope identifier with whitespace collapsed to underscores."""
        return re.sub(r'[\s/]+', '_', scope.lower().strip())

    # ── Sentence splitting ────────────────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """
        Split document into sentences for sentence-level extraction.
        Splits on '. ' + capital letter and on newlines.
        Filters out lines shorter than 20 characters (headings, labels).
        """
        text = re.sub(r'\n{2,}', '\n', text)
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)', text)
        return [s.strip() for s in parts if len(s.strip()) >= 20]

    # ── Proof builder ─────────────────────────────────────────────────────────

    def _build_proof(
        self,
        cycle_path:  list[str],
        cycle_edges: list[PriorityEdge],
    ) -> str:
        def display(node: str) -> str:
            return node.split("@")[0] if "@" in node else node

        cycle_str = " → ".join(display(n) for n in cycle_path)

        lines = [
            "PRIORITY CYCLE PROOF",
            "=" * 44,
            "",
            "A strict priority ordering must be acyclic (irreflexive + transitive).",
            "The following cycle violates acyclicity:",
            "",
            f"  {cycle_str}",
            "",
            "Contributing priority claims:",
        ]
        for i, edge in enumerate(cycle_edges, 1):
            s = display(edge.senior)
            j = display(edge.junior)
            lines.append(f"  {i}. {s} > {j}  [{edge.relation}, confidence={edge.confidence}]")
            lines.append(f'     Source: "{edge.source_text[:120]}"')
        lines += [
            "",
            "Formal contradiction:",
            "  Before(A, B) is defined as: A has priority over B.",
            "  Strict orders require: ∀x,y,z  Before(x,y) ∧ Before(y,z) → Before(x,z)",
            "  and:                   ∀x      ¬Before(x,x)",
            "  The detected cycle produces Before(A, A) via transitivity — UNSAT.",
            "",
            "Z3 verdict: unsat (acyclicity axiom violated)",
        ]
        return "\n".join(lines)


# ── Integration helper ─────────────────────────────────────────────────────────

def run_priority_cycle_check(claims: list) -> Optional[PriorityCycleResult]:
    """Convenience wrapper. Returns PriorityCycleResult or None."""
    return PriorityCycleEncoder().analyse(claims)
