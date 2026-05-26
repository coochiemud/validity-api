"""
Validity — Z3 Encoder
Component 2 of 7

Takes a validated FormalClaim AST and encodes it as Z3 constraints.
Deterministic. No probabilistic element. No LLM involvement.

Input:  A validated FormalClaim (status == VALIDATED)
Output: An EncodedClaim containing Z3 expressions, or an EncodingError

Only accepts claims that have passed the Fragment Validator.
Raises ValueError if a non-validated claim is submitted.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import uuid

import z3

from fragment_validator import (
    FormalClaim,
    ValidationStatus,
    ClaimType,
)


# ── Output types ──────────────────────────────────────────────────────────────

@dataclass
class EncodedClaim:
    """
    A validated claim encoded as Z3 expressions.
    Ready for submission to the Solver Interface (Component 3).
    """
    claim_id:    str
    constraints: list          # List of z3.ExprRef
    declarations: dict         # name -> z3.ExprRef (all declared symbols)
    provenance:  str           # verbatim source text span
    claim_type:  ClaimType


@dataclass
class EncodingError:
    claim_id: str
    reason:   str


# ── Z3 Encoder ────────────────────────────────────────────────────────────────

class Z3Encoder:
    """
    Encodes a validated FormalClaim AST into Z3 Boolean constraints.

    Design principles:
    - Every supported AST node maps to exactly one Z3 construct.
    - Symbols are declared once per encoding context and reused.
    - Deontic operators (Obligated, Forbidden, Permitted) are encoded
      structurally as Boolean constraints on their operands.
    - Temporal nodes (At) scope their body under a Boolean predicate
      keyed by time_label.
    - Quantifiers are unrolled over their finite domain.
    """

    def encode(self, claim: FormalClaim) -> EncodedClaim | EncodingError:
        """
        Main entry point. Encodes a single validated FormalClaim.
        Returns EncodedClaim on success, EncodingError on failure.
        """
        if claim.status != ValidationStatus.VALIDATED:
            raise ValueError(
                f"Claim {claim.id} is not validated "
                f"(status={claim.status.value}). "
                "Only validated claims may be encoded."
            )

        # Fresh symbol table per claim
        self._symbols: dict[str, z3.ExprRef] = {}

        try:
            constraint = self._encode_node(claim.formula)
            return EncodedClaim(
                claim_id     = claim.id,
                constraints  = [constraint],
                declarations = dict(self._symbols),
                provenance   = claim.text_span,
                claim_type   = claim.claim_type,
            )
        except Exception as e:
            return EncodingError(
                claim_id = claim.id,
                reason   = str(e),
            )

    # ── Node dispatch ─────────────────────────────────────────────────────────

    def _encode_node(self, node: dict) -> z3.ExprRef:
        node_type = node["type"]

        dispatch = {
            "Predicate": self._encode_predicate,
            "And":       self._encode_and,
            "Or":        self._encode_or,
            "Not":       self._encode_not,
            "Implies":   self._encode_implies,
            "ForAll":    self._encode_forall,
            "Exists":    self._encode_exists,
            "Obligated": self._encode_obligated,
            "Forbidden": self._encode_forbidden,
            "Permitted": self._encode_permitted,
            "At":        self._encode_at,
        }

        handler = dispatch.get(node_type)
        if handler is None:
            raise ValueError(f"No encoder for node type '{node_type}'")

        return handler(node)

    # ── Atomic ────────────────────────────────────────────────────────────────

    def _encode_predicate(self, node: dict) -> z3.ExprRef:
        """
        Predicates become Boolean constants.
        Predicate(name, args) → Bool(name__arg1__arg2)

        All arguments are folded into the symbol name to produce
        a unique, stable Boolean variable per predicate application.
        """
        name = node["name"]
        args = node.get("args", [])
        symbol_name = self._symbol_name(name, args)
        return self._declare_bool(symbol_name)

    # ── Boolean ───────────────────────────────────────────────────────────────

    def _encode_and(self, node: dict) -> z3.ExprRef:
        left  = self._encode_node(node["left"])
        right = self._encode_node(node["right"])
        return z3.And(left, right)

    def _encode_or(self, node: dict) -> z3.ExprRef:
        left  = self._encode_node(node["left"])
        right = self._encode_node(node["right"])
        return z3.Or(left, right)

    def _encode_not(self, node: dict) -> z3.ExprRef:
        operand = self._encode_node(node["operand"])
        return z3.Not(operand)

    def _encode_implies(self, node: dict) -> z3.ExprRef:
        left  = self._encode_node(node["left"])
        right = self._encode_node(node["right"])
        return z3.Implies(left, right)

    # ── Quantifiers (unrolled over finite domain) ─────────────────────────────

    def _encode_forall(self, node: dict) -> z3.ExprRef:
        """
        ForAll(var, domain, body) → And(body[var/d] for d in domain)

        Unrolled because the Fragment Validator guarantees the domain
        is finite and non-empty. No symbolic quantification needed.
        """
        var    = node["var"]
        domain = node["domain"]
        body   = node["body"]

        instances = []
        for value in domain:
            instantiated = self._substitute(body, var, value)
            instances.append(self._encode_node(instantiated))

        return z3.And(*instances) if len(instances) > 1 else instances[0]

    def _encode_exists(self, node: dict) -> z3.ExprRef:
        """
        Exists(var, domain, body) → Or(body[var/d] for d in domain)
        """
        var    = node["var"]
        domain = node["domain"]
        body   = node["body"]

        instances = []
        for value in domain:
            instantiated = self._substitute(body, var, value)
            instances.append(self._encode_node(instantiated))

        return z3.Or(*instances) if len(instances) > 1 else instances[0]

    # ── Deontic ───────────────────────────────────────────────────────────────

    def _encode_obligated(self, node: dict) -> z3.ExprRef:
        """
        Obligated(φ) → φ must hold.
        Encoded directly as the operand constraint.
        Deontic force is represented by assertion, not modal wrapping.
        """
        return self._encode_node(node["operand"])

    def _encode_forbidden(self, node: dict) -> z3.ExprRef:
        """
        Forbidden(φ) → ¬φ must hold.
        """
        return z3.Not(self._encode_node(node["operand"]))

    def _encode_permitted(self, node: dict) -> z3.ExprRef:
        """
        Permitted(φ) → φ is consistent (does not assert φ, only allows it).
        Encoded as a named Boolean to track permissibility without asserting.
        """
        operand     = self._encode_node(node["operand"])
        permit_name = f"permitted__{id(node)}"
        permit_var  = self._declare_bool(permit_name)
        # Permitted means: if the permission holds, φ is allowed.
        # We assert: permit_var → φ (i.e. permission is consistent with φ)
        return z3.Implies(permit_var, operand)

    # ── Temporal ──────────────────────────────────────────────────────────────

    def _encode_at(self, node: dict) -> z3.ExprRef:
        """
        At(time_label, φ) → Bool(at__time_label) → φ

        The time_label is encoded as a Boolean sentinel that, when true,
        implies the body constraint holds at that time point.
        """
        time_label  = node["time_label"]
        body        = self._encode_node(node["body"])
        time_symbol = self._declare_bool(f"at__{time_label}")
        return z3.Implies(time_symbol, body)

    # ── Symbol management ─────────────────────────────────────────────────────

    def _declare_bool(self, name: str) -> z3.ExprRef:
        """
        Declare a Boolean variable. Returns existing declaration if already seen.
        Ensures each symbol name maps to exactly one Z3 variable.
        """
        if name not in self._symbols:
            self._symbols[name] = z3.Bool(name)
        return self._symbols[name]

    def _symbol_name(self, predicate: str, args: list) -> str:
        """
        Construct a stable symbol name from a predicate and its arguments.
        Example: repay(borrower, principal) → repay__borrower__principal
        """
        parts = [predicate] + [str(a) for a in args]
        return "__".join(parts)

    # ── Substitution ──────────────────────────────────────────────────────────

    def _substitute(self, node: Any, var: str, value: Any) -> Any:
        """
        Recursively substitute all occurrences of var with value in an AST node.
        Returns a new node dict with the substitution applied.
        Used to unroll quantifiers over finite domains.
        """
        if isinstance(node, str):
            return value if node == var else node

        if isinstance(node, list):
            return [self._substitute(item, var, value) for item in node]

        if isinstance(node, dict):
            return {
                k: self._substitute(v, var, value)
                for k, v in node.items()
            }

        return node


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from fragment_validator import FragmentValidator, make_claim, ClaimType

    validator = FragmentValidator()
    encoder   = Z3Encoder()

    def run(label, claim):
        validator.validate(claim)
        if claim.status != ValidationStatus.VALIDATED:
            print(f"[{label}]  SKIP (outside fragment): {claim.rejection.reason}")
            return
        result = encoder.encode(claim)
        if isinstance(result, EncodingError):
            print(f"[{label}]  ENCODING ERROR: {result.reason}")
        else:
            print(f"[{label}]  OK  constraints={result.constraints}")

    # 1. Simple obligation
    run("OBLIGATION", make_claim(
        text_span  = "The borrower shall repay the principal by maturity date.",
        formula    = {
            "type":    "Obligated",
            "operand": {"type": "Predicate", "name": "repay_principal", "args": ["borrower", "maturity_date"]},
        },
        claim_type = ClaimType.OBLIGATION,
    ))

    # 2. Contradiction pair — Obligated(P) and Forbidden(P)
    #    These should encode to P and Not(P) respectively — provably unsat together.
    c1 = make_claim(
        text_span  = "The borrower shall maintain a reserve fund.",
        formula    = {"type": "Obligated", "operand": {"type": "Predicate", "name": "maintain_reserve", "args": ["borrower"]}},
        claim_type = ClaimType.OBLIGATION,
    )
    c2 = make_claim(
        text_span  = "The borrower is prohibited from maintaining a reserve fund.",
        formula    = {"type": "Forbidden", "operand": {"type": "Predicate", "name": "maintain_reserve", "args": ["borrower"]}},
        claim_type = ClaimType.OBLIGATION,
    )
    validator.validate(c1)
    validator.validate(c2)
    e1 = encoder.encode(c1)
    e2 = encoder.encode(c2)

    solver = z3.Solver()
    solver.add(*e1.constraints)
    solver.add(*e2.constraints)
    result = solver.check()
    print(f"\n[CONTRADICTION CHECK]  Z3 result: {result}")
    print(f"  Claim 1: {e1.constraints}")
    print(f"  Claim 2: {e2.constraints}")
    print(f"  Expected: unsat  Got: {result}  {'✓' if str(result) == 'unsat' else '✗'}")

    # 3. Bounded ForAll unrolling
    run("FORALL UNROLL", make_claim(
        text_span  = "All class A investors receive distributions.",
        formula    = {
            "type":   "ForAll",
            "var":    "x",
            "domain": ["investor_A", "investor_B", "investor_C"],
            "body":   {"type": "Predicate", "name": "receives_distribution", "args": ["x"]},
        },
        claim_type = ClaimType.QUANTIFIED_ASSURANCE,
    ))

    # 4. Temporal constraint
    run("TEMPORAL", make_claim(
        text_span  = "At maturity, the principal is repaid.",
        formula    = {
            "type":       "At",
            "time_label": "maturity",
            "body":       {"type": "Predicate", "name": "principal_repaid", "args": []},
        },
        claim_type = ClaimType.GUARANTEE,
    ))

    print("\nZ3 Encoder smoke test complete.")
