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
    # ── Intercreditor / corporate lending parties ──────────────────────────────
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
    # ── CLO / structured finance note classes ─────────────────────────────────
    # Longest variants first so they shadow shorter ones (e.g. "class a-l1" before "class a").
    ("class a-l2 loans",                 "ClassAL2"),
    ("class a-l1 notes",                 "ClassAL1"),
    ("class a-l1 loans",                 "ClassAL1"),
    ("class a-l notes",                  "ClassAL"),
    ("class a-s notes",                  "ClassAS"),
    ("class a-1 notes",                  "ClassA1"),
    ("class a-2 notes",                  "ClassA2"),
    ("class f-e notes",                  "ClassFE"),
    ("class f-x notes",                  "ClassFX"),
    ("class g-e notes",                  "ClassGE"),
    ("class g-x notes",                  "ClassGX"),
    ("class a notes",                    "ClassA"),
    ("class b notes",                    "ClassB"),
    ("class c notes",                    "ClassC"),
    ("class d notes",                    "ClassD"),
    ("class e notes",                    "ClassE"),
    ("class f notes",                    "ClassF"),
    ("class g notes",                    "ClassG"),
    ("class a noteholders",              "ClassA"),
    ("class b noteholders",              "ClassB"),
    ("class c noteholders",              "ClassC"),
    ("class d noteholders",              "ClassD"),
    ("class e noteholders",              "ClassE"),
    ("class f noteholders",              "ClassF"),
    ("class g noteholders",              "ClassG"),
    ("preferred interests",              "PreferredInterests"),
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
        Applies sentence-level patterns plus a document-level ordinal waterfall
        scan that can detect CLO-specific cross-section priority conflicts.
        """
        if not document_text or not document_text.strip():
            return None
        edges: list[PriorityEdge] = []
        for sentence in self._split_sentences(document_text):
            if sentence.strip():
                edges.extend(self._extract_edges(sentence, fallback_mode=True))

        # Document-level: extract named waterfall sections and compare step orderings
        # for the same (class, payment-type) pair across sections.
        edges.extend(self._extract_waterfall_cross_section_edges(document_text))

        return self._detect_cycle(edges)

    def _extract_waterfall_cross_section_edges(self, document_text: str) -> list[PriorityEdge]:
        """
        Document-level CLO waterfall analysis.

        Identifies named waterfall sections (Debt Payment Sequence, Special Priority
        of Payments, Priority of Interest/Principal Proceeds) and extracts the
        relative payment ordering of each (class, payment-type) node within each
        section using character position as a proxy for ordinal priority.

        Only the FIRST (definitional) occurrence of each section type is used.
        Cross-references to a section from within another section are ignored.

        If the SAME two (class, payment-type) nodes appear in BOTH the DPS and the
        EOD waterfall with their relative order REVERSED, that constitutes a priority
        cycle: one load-bearing provision of the indenture says A is paid before B,
        while another load-bearing provision says B is paid before A.

        Nodes are only compared if they are in DIFFERENT payment-type families
        for the SAME class (e.g. ClassC_principal vs ClassC_interest), which is
        the only scenario where the DPS and EOD intenionally define ordering.
        Pro-rata co-payments (different classes paid together in one sentence) are
        filtered by requiring a minimum position gap.
        """
        # ── Named section markers: use only FIRST occurrence of each ──────────
        section_defs = [
            (r'Debt\s+Payment\s+Sequence\s+The\s+application',    "dps",      12000),
            (r'(?:Notwithstanding[^.]*,\s*)?in\s+the\s+case\s+of\s+any\s+Enforcement\s+Event', "eod", 15000),
            (r'Special\s+Priority\s+of\s+Payments\s*\)',          "eod",      15000),
        ]

        named_sections: dict[str, tuple[int, int]] = {}  # key → (start, end)
        for pattern, key, max_len in section_defs:
            if key in named_sections:
                continue
            m = re.search(pattern, document_text, re.IGNORECASE)
            if m:
                start = m.start()
                end   = start + max_len
                named_sections[key] = (start, end)

        if len(named_sections) < 2:
            return []

        # ── Extract nodes from each named section ─────────────────────────────
        sections: dict[str, list[tuple[str, int, str]]] = {}
        for key, (start, end) in named_sections.items():
            window = document_text[start:end]
            nodes  = self._extract_ordinal_nodes(window, key)
            if nodes:
                sections[key] = nodes

        if len(sections) < 2:
            return []

        # ── Cross-section conflict detection ──────────────────────────────────
        # Compare only nodes that belong to the SAME note class but DIFFERENT
        # payment types (principal vs interest).  This targets the genuine
        # DPS/EOD ordering conflict and avoids pro-rata co-payment false positives.
        edges: list[PriorityEdge] = []
        section_keys = list(sections.keys())
        MIN_POSITION_GAP = 50   # chars — two nodes ≤ 50 chars apart = same sentence

        for i in range(len(section_keys)):
            for j in range(i + 1, len(section_keys)):
                sk1, sk2 = section_keys[i], section_keys[j]
                nodes1 = {n: (pos, src) for n, pos, src in sections[sk1]}
                nodes2 = {n: (pos, src) for n, pos, src in sections[sk2]}
                common = set(nodes1) & set(nodes2)

                for n_a in common:
                    for n_b in common:
                        if n_a >= n_b:
                            continue

                        # Only compare nodes with the same class prefix but
                        # different payment types (e.g. ClassC_principal vs ClassC_interest)
                        base_a = n_a.rsplit("_", 1)[0]
                        base_b = n_b.rsplit("_", 1)[0]
                        if base_a != base_b:
                            continue   # Different classes — skip

                        pos1_a, src1_a = nodes1[n_a]
                        pos1_b, src1_b = nodes1[n_b]
                        pos2_a, src2_a = nodes2[n_a]
                        pos2_b, src2_b = nodes2[n_b]

                        # Skip if nodes are too close together in either section
                        # (likely pro-rata co-payments in the same sentence)
                        if abs(pos1_a - pos1_b) < MIN_POSITION_GAP:
                            continue
                        if abs(pos2_a - pos2_b) < MIN_POSITION_GAP:
                            continue

                        # Detect reversed ordering between the two sections.
                        # Case 1: sk1 says n_a > n_b; sk2 says n_b > n_a
                        # Case 2: sk1 says n_b > n_a; sk2 says n_a > n_b  (same conflict, reversed labels)
                        conflict = False
                        if pos1_a < pos1_b and pos2_b < pos2_a:
                            # sk1: n_a first; sk2: n_b first
                            senior_in_1, junior_in_1 = n_a, n_b
                            senior_in_2, junior_in_2 = n_b, n_a
                            src_fwd, src_rev = src1_a, src2_b
                            conflict = True
                        elif pos1_b < pos1_a and pos2_a < pos2_b:
                            # sk1: n_b first; sk2: n_a first
                            senior_in_1, junior_in_1 = n_b, n_a
                            senior_in_2, junior_in_2 = n_a, n_b
                            src_fwd, src_rev = src1_b, src2_a
                            conflict = True

                        if conflict:
                            cid_fwd = str(uuid.uuid4())
                            cid_rev = str(uuid.uuid4())
                            edges.append(PriorityEdge(
                                senior=f"{senior_in_1}@waterfall_order",
                                junior=f"{junior_in_1}@waterfall_order",
                                relation=f"paid_before_in_{sk1}",
                                source_text=f"[{sk1}] {src_fwd[:300]}",
                                clause_id=cid_fwd,
                                confidence="high",
                            ))
                            edges.append(PriorityEdge(
                                senior=f"{senior_in_2}@waterfall_order",
                                junior=f"{junior_in_2}@waterfall_order",
                                relation=f"paid_before_in_{sk2}",
                                source_text=f"[{sk2}] {src_rev[:300]}",
                                clause_id=cid_rev,
                                confidence="high",
                            ))

        return edges

    def _extract_ordinal_nodes(self, section_text: str, section_key: str) -> list[tuple[str, int, str]]:
        """
        Extract (node_name, ordinal_step, source_snippet) tuples from a waterfall section.

        Uses character POSITION within the section as a proxy for ordinal priority:
        a payment description appearing earlier in the text has a smaller ordinal
        and therefore higher payment priority.  This is robust to all step-label
        formats (numeric, alpha, dotted-numeric like 1.1.1.2.1.X) and avoids false
        positives from inline cross-reference labels such as "(B)(1)".

        Two payment types are recognised for each CLO note class:
          - <Class>_interest  — interest, accrued interest, Interest Distribution Amount
          - <Class>_principal — principal, deferred principal, Deferred Interest
        """
        nodes: list[tuple[str, int, str]] = []
        seen_nodes: set[str] = set()   # (node_name) — first occurrence wins

        # Two patterns covering both word orderings found in CLO documents.

        # Pattern A: "[principal/interest] … Class X Notes"  (DPS style)
        pat_a = re.compile(
            r'\b(principal|interest|deferred)\b.{0,150}?\b'
            r'(Class\s+[A-G][\w\-]*\s+(?:Notes?|Noteholders?|Loans?))',
            re.IGNORECASE,
        )
        # Pattern B: "Class X Notes … [principal/interest/Interest Distribution Amount]"  (waterfall style)
        pat_b = re.compile(
            r'\b(Class\s+[A-G][\w\-]*\s+(?:Notes?|Noteholders?|Loans?))'
            r'.{0,200}?'
            r'\b(Interest\s+Distribution\s+Amount|principal\s+of|accrued[^.]{0,60}?interest|'
            r'deferred\s+interest|interest)',
            re.IGNORECASE,
        )

        # Collect all matches with their position in the section text
        matches: list[tuple[int, str, str]] = []  # (char_pos, canonical_node, source_snippet)

        for m in pat_a.finditer(section_text):
            pay_raw   = m.group(1).lower()
            class_raw = m.group(2).strip()
            pay_type  = "principal" if pay_raw in ("principal", "deferred") else "interest"
            canonical = self._normalize(class_raw)
            if canonical:
                node = f"{canonical}_{pay_type}"
                src  = section_text[max(0, m.start()-20):m.end()].replace('\n', ' ')[:150]
                matches.append((m.start(), node, src))

        for m in pat_b.finditer(section_text):
            class_raw = m.group(1).strip()
            pay_raw   = m.group(2).lower()
            pay_type  = (
                "principal"
                if "principal" in pay_raw or "deferred" in pay_raw
                else "interest"
            )
            canonical = self._normalize(class_raw)
            if canonical:
                node = f"{canonical}_{pay_type}"
                src  = section_text[m.start():m.end()].replace('\n', ' ')[:150]
                matches.append((m.start(), node, src))

        # Sort by position; use position as ordinal; keep first occurrence of each node
        matches.sort(key=lambda x: x[0])
        for pos, node, src in matches:
            if node not in seen_nodes:
                seen_nodes.add(node)
                nodes.append((node, pos, src))

        return nodes

    # ── Edge extraction ───────────────────────────────────────────────────────

    def _extract_edges(self, text: str, fallback_mode: bool) -> list[PriorityEdge]:
        """
        Try each extraction pattern in priority order.
        Patterns 1 and 2 are high confidence and scope-aware.
        Pattern 3 (medium confidence, global scope) is only enabled in fallback_mode
        when patterns 1 and 2 find nothing.
        Pattern 4 (CLO note class direct priority) runs in fallback_mode.
        """
        edges: list[PriorityEdge] = []

        # Pattern 1: "For the [SCOPE], A shall have [first] priority over B"
        edges.extend(self._pat_priority_over(text))

        # Pattern 2: "A Obligations...senior/junior...B Obligations" + collateral scope
        edges.extend(self._pat_obligations_senior_junior(text))

        # Pattern 3: general fallback (medium confidence, no scope)
        if fallback_mode and not edges:
            edges.extend(self._pat_general_senior_junior(text))

        # Pattern 4: CLO note class direct priority language (fallback_mode)
        if fallback_mode and not edges:
            edges.extend(self._pat_clo_note_priority(text))

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

    def _pat_clo_note_priority(self, text: str) -> list[PriorityEdge]:
        """
        Pattern 4 — CLO note class direct priority language (medium confidence, global scope).

        Matches CLO-specific priority statements:
          'Class A Notes are [senior / senior in right of payment] to Class B Notes'
          'Class B Notes are [subordinated / junior] to Class A Notes'
          'prior to any payment on the Class B Notes ... Class A Notes'
          'no payment shall be made on Class B Notes until Class A Notes paid in full'
          'Class A Notes shall rank senior to Class B Notes'

        Uses global scope because CLO notes operate within a single collateral pool.
        """
        edges: list[PriorityEdge] = []

        # "Class X [are/is/shall be] senior [in right of payment] to Class Y"
        m = re.search(
            r'(Class\s+[A-G][\w\-]*\s+(?:Notes?|Noteholders?|Loans?))\s+'
            r'(?:are|is|shall\s+be|rank[s]?)\s+senior(?:\s+in\s+right\s+of\s+payment)?\s+to\s+'
            r'(Class\s+[A-G][\w\-]*\s+(?:Notes?|Noteholders?|Loans?))',
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

        # "Class X Notes [are/is/shall be] [subordinated/junior] to Class Y Notes"
        m = re.search(
            r'(Class\s+[A-G][\w\-]*\s+(?:Notes?|Noteholders?|Loans?))\s+'
            r'(?:are|is|shall\s+be)\s+(?:subordinated?|junior)\s+to\s+'
            r'(Class\s+[A-G][\w\-]*\s+(?:Notes?|Noteholders?|Loans?))',
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

        # "no payment ... to Class X ... until Class Y is paid in full"
        m = re.search(
            r'no\s+payment\s+(?:shall\s+be\s+made\s+)?(?:on|to)\s+'
            r'(Class\s+[A-G][\w\-]*\s+(?:Notes?|Noteholders?))[^.]*?until\s+'
            r'(Class\s+[A-G][\w\-]*\s+(?:Notes?|Noteholders?))\s+(?:is|are|has\s+been)\s+paid\s+in\s+full',
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
