"""
Validity — Fragment Validator
Component 1 of 7

The Fragment Validator is the trust anchor of the system.
It functions as a type checker, not a permissive parser.
Its job is not to make things work. Its job is to reject
anything that cannot be represented with deterministic
semantic precision inside the Logical Fragment Specification (LFS v2).

Input:  A candidate FormalClaim AST
Output: validated | outside_fragment
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import uuid


# ── Enums ────────────────────────────────────────────────────────────────────

class ClaimType(Enum):
    GUARANTEE            = "guarantee"
    CONDITION            = "condition"
    DISCLAIMER           = "disclaimer"
    OBLIGATION           = "obligation"
    QUANTIFIED_ASSURANCE = "quantified_assurance"


class ValidationStatus(Enum):
    VALIDATED        = "validated"
    OUTSIDE_FRAGMENT = "outside_fragment"


# ── AST Node Types ───────────────────────────────────────────────────────────
# These mirror the LFS v2 AST specification exactly.
# Any node type not listed here is automatically outside the fragment.

class NodeType(Enum):
    # Atomic
    PREDICATE  = "Predicate"

    # Boolean
    AND        = "And"
    OR         = "Or"
    NOT        = "Not"
    IMPLIES    = "Implies"

    # Quantified (bounded only)
    FORALL     = "ForAll"
    EXISTS     = "Exists"

    # Deontic (structural only)
    OBLIGATED  = "Obligated"
    FORBIDDEN  = "Forbidden"
    PERMITTED  = "Permitted"

    # Temporal (discrete scope)
    AT         = "At"


# ── Forbidden modal terms ─────────────────────────────────────────────────────
# Per LFS v2 Section 3.3: unresolved modal terms cause immediate rejection
# unless formally bounded by the translator.

FORBIDDEN_MODAL_TERMS = {
    "reasonable", "reasonably",
    "material", "materially",
    "significant", "significantly",
    "promptly",
    "appropriate", "appropriately",
    "substantial", "substantially",
    "adequate", "adequately",
    "sufficient", "sufficiently",
    "necessary",
    "proper", "properly",
    "timely",
    "good faith",
}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class Provenance:
    text_span:    str
    page:         Optional[int]  = None
    section:      Optional[str]  = None
    char_offset:  Optional[int]  = None
    rule_id:      Optional[str]  = None
    lfs_version:  str            = "2.0"


@dataclass
class RejectionRecord:
    reason:        str
    rule_violated: str
    node_path:     str = ""


@dataclass
class FormalClaim:
    """
    Output of Stage 2 translation, input to the Fragment Validator.
    The validator populates status and rejection fields.
    """
    id:          str
    formula:     dict            # AST as nested dict
    claim_type:  ClaimType
    confidence:  float
    provenance:  Provenance
    text_span:   str             # verbatim source text

    # Populated by validator
    status:      ValidationStatus = ValidationStatus.OUTSIDE_FRAGMENT
    rejection:   Optional[RejectionRecord] = None


@dataclass
class ValidationResult:
    claim_id:  str
    status:    ValidationStatus
    rejection: Optional[RejectionRecord] = None


# ── Fragment Validator ────────────────────────────────────────────────────────

class FragmentValidator:
    """
    Deterministic boundary enforcement for LFS v2.

    Validates a FormalClaim AST against the Logical Fragment Specification.
    Returns validated or outside_fragment — never approximates, never coerces.

    Rules applied (in order):
      R01 — Claim type must be a recognised ClaimType
      R02 — Formula must be a non-empty dict with a 'type' key
      R03 — All AST node types must be within the supported set
      R04 — Predicates must have a name and args list
      R05 — Boolean operators must have correct arity
      R06 — Quantifiers must be bounded (finite domain required)
      R07 — Quantifier alternation depth must not exceed 1
      R08 — Deontic operators must not nest inside quantifiers
      R09 — Temporal nodes must carry a time_label
      R10 — No forbidden modal terms in any text field
      R11 — Confidence must be a float in [0, 1]
      R12 — Provenance text_span must be non-empty
    """

    MAX_QUANTIFIER_DEPTH = 1

    def validate(self, claim: FormalClaim) -> ValidationResult:
        """
        Main entry point. Validates a single FormalClaim.
        Returns a ValidationResult. Mutates claim.status and claim.rejection.
        """
        rejection = self._run_rules(claim)

        if rejection:
            claim.status    = ValidationStatus.OUTSIDE_FRAGMENT
            claim.rejection = rejection
            return ValidationResult(
                claim_id  = claim.id,
                status    = ValidationStatus.OUTSIDE_FRAGMENT,
                rejection = rejection,
            )

        claim.status    = ValidationStatus.VALIDATED
        claim.rejection = None
        return ValidationResult(
            claim_id = claim.id,
            status   = ValidationStatus.VALIDATED,
        )

    def _run_rules(self, claim: FormalClaim) -> Optional[RejectionRecord]:
        """
        Run all validation rules in order.
        Returns the first RejectionRecord encountered, or None if clean.
        """

        # R01 — Claim type
        if not isinstance(claim.claim_type, ClaimType):
            return RejectionRecord(
                reason        = f"Unrecognised claim type: {claim.claim_type!r}",
                rule_violated = "R01",
            )

        # R11 — Confidence range
        if not isinstance(claim.confidence, (int, float)) or not (0.0 <= claim.confidence <= 1.0):
            return RejectionRecord(
                reason        = f"Confidence must be float in [0,1], got: {claim.confidence!r}",
                rule_violated = "R11",
            )

        # R12 — Provenance
        if not claim.provenance.text_span or not claim.provenance.text_span.strip():
            return RejectionRecord(
                reason        = "Provenance text_span is empty",
                rule_violated = "R12",
            )

        # R10 — Modal terms in text_span
        modal_hit = self._check_modal_terms(claim.text_span)
        if modal_hit:
            return RejectionRecord(
                reason        = f"Unresolved modal term in source text: '{modal_hit}'",
                rule_violated = "R10",
                node_path     = "text_span",
            )

        # R02 — Formula structure
        if not claim.formula or not isinstance(claim.formula, dict) or "type" not in claim.formula:
            return RejectionRecord(
                reason        = "Formula must be a non-empty dict with a 'type' key",
                rule_violated = "R02",
            )

        # Recursively validate AST
        return self._validate_node(claim.formula, path="root", quantifier_depth=0)

    def _validate_node(
        self,
        node: Any,
        path: str,
        quantifier_depth: int,
        inside_deontic: bool = False,
    ) -> Optional[RejectionRecord]:
        """
        Recursively validate an AST node.
        """
        if not isinstance(node, dict) or "type" not in node:
            return RejectionRecord(
                reason        = f"AST node at '{path}' is not a dict with a 'type' key",
                rule_violated = "R02",
                node_path     = path,
            )

        raw_type = node.get("type")

        # R03 — Node type must be in supported set
        try:
            node_type = NodeType(raw_type)
        except ValueError:
            return RejectionRecord(
                reason        = f"Unsupported AST node type '{raw_type}' at '{path}'",
                rule_violated = "R03",
                node_path     = path,
            )

        # Dispatch to type-specific validators
        if node_type == NodeType.PREDICATE:
            return self._validate_predicate(node, path)

        elif node_type in (NodeType.AND, NodeType.OR, NodeType.IMPLIES):
            return self._validate_binary(node, node_type, path, quantifier_depth, inside_deontic)

        elif node_type == NodeType.NOT:
            return self._validate_unary(node, path, quantifier_depth, inside_deontic)

        elif node_type in (NodeType.FORALL, NodeType.EXISTS):
            return self._validate_quantifier(node, node_type, path, quantifier_depth, inside_deontic)

        elif node_type in (NodeType.OBLIGATED, NodeType.FORBIDDEN, NodeType.PERMITTED):
            return self._validate_deontic(node, node_type, path, quantifier_depth)

        elif node_type == NodeType.AT:
            return self._validate_temporal(node, path, quantifier_depth, inside_deontic)

        return None

    def _validate_predicate(self, node: dict, path: str) -> Optional[RejectionRecord]:
        # R04 — Predicates must have name and args
        if "name" not in node or not isinstance(node["name"], str) or not node["name"].strip():
            return RejectionRecord(
                reason        = f"Predicate at '{path}' missing or empty 'name'",
                rule_violated = "R04",
                node_path     = path,
            )
        if "args" not in node or not isinstance(node["args"], list):
            return RejectionRecord(
                reason        = f"Predicate at '{path}' missing 'args' list",
                rule_violated = "R04",
                node_path     = path,
            )
        # Check modal terms in predicate name
        modal_hit = self._check_modal_terms(node["name"])
        if modal_hit:
            return RejectionRecord(
                reason        = f"Unresolved modal term '{modal_hit}' in predicate name at '{path}'",
                rule_violated = "R10",
                node_path     = path,
            )
        return None

    def _validate_binary(
        self,
        node: dict,
        node_type: NodeType,
        path: str,
        quantifier_depth: int,
        inside_deontic: bool,
    ) -> Optional[RejectionRecord]:
        # R05 — Binary operators require left and right children
        for child_key in ("left", "right"):
            if child_key not in node:
                return RejectionRecord(
                    reason        = f"{node_type.value} at '{path}' missing '{child_key}'",
                    rule_violated = "R05",
                    node_path     = path,
                )
            result = self._validate_node(
                node[child_key],
                path=f"{path}.{child_key}",
                quantifier_depth=quantifier_depth,
                inside_deontic=inside_deontic,
            )
            if result:
                return result
        return None

    def _validate_unary(
        self,
        node: dict,
        path: str,
        quantifier_depth: int,
        inside_deontic: bool,
    ) -> Optional[RejectionRecord]:
        # R05 — Not requires a single child
        if "operand" not in node:
            return RejectionRecord(
                reason        = f"Not at '{path}' missing 'operand'",
                rule_violated = "R05",
                node_path     = path,
            )
        return self._validate_node(
            node["operand"],
            path=f"{path}.operand",
            quantifier_depth=quantifier_depth,
            inside_deontic=inside_deontic,
        )

    def _validate_quantifier(
        self,
        node: dict,
        node_type: NodeType,
        path: str,
        quantifier_depth: int,
        inside_deontic: bool,
    ) -> Optional[RejectionRecord]:
        # R08 — Deontic operators must not contain quantifiers
        if inside_deontic:
            return RejectionRecord(
                reason        = f"Quantifier '{node_type.value}' at '{path}' nested inside deontic operator — forbidden by LFS v2",
                rule_violated = "R08",
                node_path     = path,
            )

        # R07 — Quantifier alternation depth
        new_depth = quantifier_depth + 1
        if new_depth > self.MAX_QUANTIFIER_DEPTH:
            return RejectionRecord(
                reason        = f"Quantifier nesting depth {new_depth} exceeds maximum of {self.MAX_QUANTIFIER_DEPTH} at '{path}'",
                rule_violated = "R07",
                node_path     = path,
            )

        # R06 — Domain must be present and finite/enumerated
        if "domain" not in node:
            return RejectionRecord(
                reason        = f"{node_type.value} at '{path}' missing 'domain' — unbounded quantifier",
                rule_violated = "R06",
                node_path     = path,
            )
        domain = node["domain"]
        if not isinstance(domain, (list, set)) or len(domain) == 0:
            return RejectionRecord(
                reason        = f"{node_type.value} at '{path}' domain must be a non-empty finite list or set",
                rule_violated = "R06",
                node_path     = path,
            )

        if "var" not in node or not isinstance(node["var"], str):
            return RejectionRecord(
                reason        = f"{node_type.value} at '{path}' missing 'var'",
                rule_violated = "R06",
                node_path     = path,
            )

        if "body" not in node:
            return RejectionRecord(
                reason        = f"{node_type.value} at '{path}' missing 'body'",
                rule_violated = "R06",
                node_path     = path,
            )

        return self._validate_node(
            node["body"],
            path=f"{path}.body",
            quantifier_depth=new_depth,
            inside_deontic=inside_deontic,
        )

    def _validate_deontic(
        self,
        node: dict,
        node_type: NodeType,
        path: str,
        quantifier_depth: int,
    ) -> Optional[RejectionRecord]:
        if "operand" not in node:
            return RejectionRecord(
                reason        = f"{node_type.value} at '{path}' missing 'operand'",
                rule_violated = "R05",
                node_path     = path,
            )
        return self._validate_node(
            node["operand"],
            path=f"{path}.operand",
            quantifier_depth=quantifier_depth,
            inside_deontic=True,  # flag for R08
        )

    def _validate_temporal(
        self,
        node: dict,
        path: str,
        quantifier_depth: int,
        inside_deontic: bool,
    ) -> Optional[RejectionRecord]:
        # R09 — Temporal nodes must have a time_label
        if "time_label" not in node or not isinstance(node["time_label"], str) or not node["time_label"].strip():
            return RejectionRecord(
                reason        = f"At node at '{path}' missing or empty 'time_label'",
                rule_violated = "R09",
                node_path     = path,
            )
        if "body" not in node:
            return RejectionRecord(
                reason        = f"At node at '{path}' missing 'body'",
                rule_violated = "R09",
                node_path     = path,
            )
        return self._validate_node(
            node["body"],
            path=f"{path}.body",
            quantifier_depth=quantifier_depth,
            inside_deontic=inside_deontic,
        )

    def _check_modal_terms(self, text: str) -> Optional[str]:
        """
        Check for forbidden modal terms in a text string.
        Returns the first hit, or None if clean.
        """
        if not text:
            return None
        lower = text.lower()
        for term in FORBIDDEN_MODAL_TERMS:
            if term in lower:
                return term
        return None


# ── Convenience factory ───────────────────────────────────────────────────────

def make_claim(
    text_span:  str,
    formula:    dict,
    claim_type: ClaimType = ClaimType.OBLIGATION,
    confidence: float     = 1.0,
    page:       int       = 1,
    section:    str       = "",
) -> FormalClaim:
    """
    Convenience factory for constructing FormalClaim objects in tests.
    """
    return FormalClaim(
        id         = str(uuid.uuid4()),
        formula    = formula,
        claim_type = claim_type,
        confidence = confidence,
        provenance = Provenance(
            text_span = text_span,
            page      = page,
            section   = section,
        ),
        text_span  = text_span,
    )


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    validator = FragmentValidator()

    # Should pass: simple obligation with a valid predicate
    claim_valid = make_claim(
        text_span  = "The borrower shall repay the principal by maturity date.",
        formula    = {
            "type":    "Obligated",
            "operand": {
                "type": "Predicate",
                "name": "repay_principal",
                "args": ["borrower", "maturity_date"],
            }
        },
        claim_type = ClaimType.OBLIGATION,
        confidence = 0.95,
    )

    result = validator.validate(claim_valid)
    print(f"[VALID CLAIM]    status={result.status.value}")

    # Should fail: modal term 'reasonable' in text_span
    claim_modal = make_claim(
        text_span  = "The borrower shall act in a reasonable manner.",
        formula    = {
            "type":    "Obligated",
            "operand": {
                "type": "Predicate",
                "name": "act",
                "args": ["borrower"],
            }
        },
        claim_type = ClaimType.OBLIGATION,
        confidence = 0.90,
    )

    result = validator.validate(claim_modal)
    print(f"[MODAL TERM]     status={result.status.value}  rule={result.rejection.rule_violated}  reason={result.rejection.reason}")

    # Should fail: unbounded quantifier
    claim_unbounded = make_claim(
        text_span  = "All investors will receive distributions.",
        formula    = {
            "type":   "ForAll",
            "var":    "x",
            "body":   {
                "type": "Predicate",
                "name": "receives_distribution",
                "args": ["x"],
            }
        },
        claim_type = ClaimType.QUANTIFIED_ASSURANCE,
        confidence = 0.88,
    )

    result = validator.validate(claim_unbounded)
    print(f"[UNBOUND QUANT]  status={result.status.value}  rule={result.rejection.rule_violated}  reason={result.rejection.reason}")

    # Should fail: unsupported node type
    claim_bad_node = make_claim(
        text_span  = "The fund will likely outperform.",
        formula    = {
            "type": "Probabilistic",
            "p":    0.8,
        },
        claim_type = ClaimType.GUARANTEE,
        confidence = 0.70,
    )

    result = validator.validate(claim_bad_node)
    print(f"[BAD NODE TYPE]  status={result.status.value}  rule={result.rejection.rule_violated}  reason={result.rejection.reason}")

    print("\nFragment Validator smoke test complete.")
