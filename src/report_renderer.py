"""
Validity — Report Renderer
Component 8 of 8

Generates a deployable HTML verification report from a pipeline AnalysisOutput.
Accepts the output of ProofMapper.map() and produces a self-contained HTML file
matching the Validity report design system.

Input:  AnalysisOutput (from ProofMapper)
        document_name: str
        output_path: str
Output: HTML file written to output_path
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import html
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from proof_mapper import AnalysisOutput, ProofObject, CleanVerdict, FailureClass


# ── Tag classification ─────────────────────────────────────────────────────────

def _classify_refusal(reason: str, rule: str) -> tuple[str, str]:
    """
    Returns (tag_label, tag_class) for a refused claim.
    """
    reason_lower = reason.lower()
    rule_upper   = rule.upper()

    if "modal" in reason_lower or "reasonable" in reason_lower or "material" in reason_lower or "substantial" in reason_lower:
        return "Modal", "modal"
    if "temporal" in reason_lower or "period" in reason_lower or "r09" in rule_upper:
        return "Temporal", "temporal"
    if "quantifier" in reason_lower or "nested" in reason_lower or "r08" in rule_upper:
        return "Structural", "structural"
    if "reference" in reason_lower or "glossary" in reason_lower or "cross-reference" in reason_lower or "trans" in rule_upper:
        return "Reference", "reference"
    if "r10" in rule_upper:
        return "Modal", "modal"
    if "r07" in rule_upper or "r06" in rule_upper:
        return "Structural", "structural"
    return "Refused", "refused"


def _frag_summary(text: str) -> str:
    """Produce a short summary for the collapsed fragment row."""
    words = text.split()
    short = " ".join(words[:12])
    return short + "…" if len(words) > 12 else short


# ── HTML template ──────────────────────────────────────────────────────────────

STYLES = """
        * { margin:0; padding:0; box-sizing:border-box; }
        :root {
            --bg:          #09090F;
            --surface:     #0E0E16;
            --surface2:    #131320;
            --surface3:    #181828;
            --border:      rgba(255,255,255,0.055);
            --border2:     rgba(255,255,255,0.09);
            --border3:     rgba(255,255,255,0.14);
            --text:        #D8DCE8;
            --text-mid:    #7B8499;
            --text-dim:    #3C4257;
            --text-faint:  #252A3A;
            --proof-blue:  #4A7FA5;
            --proof-blue-dim: rgba(74,127,165,0.10);
            --ver-green:   #3D7A6A;
            --ver-green-dim: rgba(61,122,106,0.08);
            --frag-brass:  #8A7040;
            --frag-brass-dim: rgba(138,112,64,0.08);
            --red:         #8A4040;
            --red-dim:     rgba(138,64,64,0.08);
            --mono: 'IBM Plex Mono', monospace;
            --serif: 'IBM Plex Serif', Georgia, serif;
            --sans: 'IBM Plex Sans', system-ui, sans-serif;
        }
        html { scroll-behavior: smooth; }
        body { font-family:var(--sans); background:var(--bg); color:var(--text); min-height:100vh; -webkit-font-smoothing:antialiased; overflow-x:hidden; }
        body::before { content:''; position:fixed; inset:0; opacity:0.018; pointer-events:none; z-index:0; background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
        header { position:sticky; top:0; z-index:100; display:flex; align-items:center; justify-content:space-between; padding:0 48px; height:50px; background:rgba(9,9,15,0.92); backdrop-filter:blur(16px); border-bottom:1px solid var(--border); }
        .nav-logo img { height:22px; display:block; opacity:0.9; }
        .nav-label { font-family:var(--mono); font-size:10px; letter-spacing:0.12em; text-transform:uppercase; color:var(--text-dim); }
        main { position:relative; z-index:1; max-width:660px; margin:0 auto; padding:80px 48px 140px; }
        .doc-eyebrow { font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.18em; color:var(--text-dim); margin-bottom:12px; display:block; opacity:0; animation:fadeUp 0.7s ease forwards 0.05s; }
        .doc-title { font-family:var(--serif); font-size:clamp(18px,2.8vw,26px); font-weight:400; color:var(--text); line-height:1.3; letter-spacing:-0.01em; margin-bottom:40px; opacity:0; animation:fadeUp 0.7s ease forwards 0.10s; }
        .result-seal { border:1px solid var(--border2); border-radius:6px; overflow:hidden; margin-bottom:52px; opacity:0; animation:fadeUp 0.7s ease forwards 0.16s; }
        .seal-body { padding:44px 40px 36px; background:var(--surface); border-bottom:1px solid var(--border); }
        .seal-outcome { font-family:var(--sans); font-size:clamp(22px,3.6vw,32px); font-weight:300; color:var(--text); letter-spacing:-0.025em; margin-bottom:16px; line-height:1.1; }
        .seal-outcome.contradiction { color:#C47070; }
        .seal-counts { font-family:var(--mono); font-size:11px; color:var(--text-mid); letter-spacing:0.03em; margin-bottom:24px; }
        .seal-state-row { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
        .seal-pill { font-family:var(--mono); font-size:9px; font-weight:500; letter-spacing:0.12em; text-transform:uppercase; padding:4px 11px; border-radius:3px; }
        .seal-pill.sat { background:var(--ver-green-dim); color:var(--ver-green); border:1px solid rgba(61,122,106,0.22); }
        .seal-pill.unsat { background:var(--red-dim); color:var(--red); border:1px solid rgba(138,64,64,0.22); }
        .seal-meta-item { font-family:var(--mono); font-size:10px; color:var(--text-dim); letter-spacing:0.05em; }
        .seal-sep { color:var(--text-faint); }
        .seal-footer { display:grid; grid-template-columns:1fr 1fr; background:var(--surface2); }
        .seal-field { padding:14px 20px; border-right:1px solid var(--border); border-bottom:1px solid var(--border); }
        .seal-field:nth-child(2n) { border-right:none; }
        .seal-field:nth-last-child(-n+2) { border-bottom:none; }
        .seal-field-label { font-family:var(--mono); font-size:8px; text-transform:uppercase; letter-spacing:0.14em; color:var(--text-dim); margin-bottom:5px; }
        .seal-field-value { font-family:var(--mono); font-size:10px; color:var(--text-mid); word-break:break-all; line-height:1.4; }
        .seal-field-value.proof { color:var(--proof-blue); }
        .phil { font-family:var(--serif); font-size:14px; font-style:italic; font-weight:300; color:var(--text-dim); line-height:1.8; padding-bottom:44px; border-bottom:1px solid var(--border); margin-bottom:52px; opacity:0; animation:fadeUp 0.7s ease forwards 0.22s; }
        .sec-head { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:16px; }
        .sec-label { font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.16em; color:var(--text-dim); }
        .sec-count { font-family:var(--mono); font-size:9px; color:var(--text-dim); letter-spacing:0.06em; }

        /* Contradiction core */
        .core-section { margin-bottom:52px; opacity:0; animation:fadeUp 0.7s ease forwards 0.25s; }
        .core-item { border:1px solid rgba(138,64,64,0.20); border-radius:5px; padding:18px 20px; margin-bottom:4px; background:rgba(138,64,64,0.04); }
        .core-item-id { font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.10em; color:var(--text-dim); margin-bottom:10px; }
        .core-item-text { font-family:var(--sans); font-size:13px; font-weight:300; color:var(--text); line-height:1.65; font-style:italic; margin-bottom:8px; }
        .core-item-loc { font-family:var(--mono); font-size:9px; color:var(--text-dim); }
        .proof-block { background:#060608; border:1px solid var(--border); border-radius:4px; padding:16px 18px; margin-top:4px; }
        .proof-block-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
        .proof-block-label { font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.14em; color:var(--text-dim); }
        .proof-block-state { font-family:var(--mono); font-size:9px; color:var(--red); }
        .proof-line { font-family:var(--mono); font-size:10px; color:var(--text-dim); line-height:1.85; }
        .proof-line .c { color:var(--text-mid); }
        .proof-line .bot { color:var(--red); }
        .proof-sep { border:none; border-top:1px solid var(--border); margin:10px 0; }

        /* Verified */
        .verified-section { margin-bottom:52px; opacity:0; animation:fadeUp 0.7s ease forwards 0.27s; }
        .verified-list { border:1px solid var(--border); border-radius:5px; overflow:hidden; }
        .verified-item { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; padding:13px 18px; border-bottom:1px solid var(--border); background:var(--surface); transition:background 0.15s; }
        .verified-item:last-child { border-bottom:none; }
        .verified-item:hover { background:var(--surface2); }
        .verified-item-name { font-family:var(--sans); font-size:12px; font-weight:400; color:var(--text); margin-bottom:3px; }
        .verified-item-desc { font-family:var(--sans); font-size:11px; font-weight:300; color:var(--text-mid); line-height:1.55; }
        .verified-dot { width:5px; height:5px; border-radius:50%; background:var(--ver-green); flex-shrink:0; margin-top:6px; opacity:0.6; }

        /* Outside fragment */
        .fragment-section { margin-bottom:52px; opacity:0; animation:fadeUp 0.7s ease forwards 0.32s; }
        .fragment-intro { font-family:var(--sans); font-size:12px; font-weight:300; color:var(--text-mid); line-height:1.75; margin-bottom:16px; }
        .fragment-row { border:1px solid var(--border); border-radius:4px; margin-bottom:3px; overflow:hidden; background:var(--surface); }
        .fragment-row-head { display:flex; align-items:center; gap:10px; padding:12px 16px; cursor:pointer; transition:background 0.15s; user-select:none; }
        .fragment-row-head:hover { background:var(--surface2); }
        .frag-tag { font-family:var(--mono); font-size:8px; letter-spacing:0.10em; text-transform:uppercase; padding:2px 7px; border-radius:2px; color:var(--frag-brass); border:1px solid rgba(138,112,64,0.28); background:var(--frag-brass-dim); flex-shrink:0; }
        .frag-rule { font-family:var(--mono); font-size:8px; letter-spacing:0.08em; color:var(--text-dim); border:1px solid var(--border2); padding:2px 7px; border-radius:2px; flex-shrink:0; }
        .frag-summary { font-family:var(--sans); font-size:11px; font-weight:300; color:var(--text-mid); flex:1; line-height:1.4; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
        .frag-chevron { font-family:var(--mono); font-size:9px; color:var(--text-dim); transition:transform 0.2s; flex-shrink:0; }
        .frag-chevron.open { transform:rotate(180deg); }
        .fragment-row-body { display:none; padding:0 16px 16px; border-top:1px solid var(--border); }
        .fragment-row-body.open { display:block; }
        .frag-clause { font-family:var(--mono); font-size:10px; color:var(--text-mid); line-height:1.8; padding:12px 14px; background:var(--surface2); border-radius:3px; margin-top:12px; margin-bottom:10px; border-left:2px solid var(--border3); }
        .frag-reason { font-family:var(--sans); font-size:11px; font-weight:300; color:var(--text-mid); line-height:1.65; }

        /* Audit */
        .audit-section { opacity:0; animation:fadeUp 0.7s ease forwards 0.37s; }
        .audit-btn { display:flex; align-items:center; gap:10px; font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.14em; color:var(--text-dim); background:none; border:1px solid var(--border2); padding:10px 16px; border-radius:4px; cursor:pointer; transition:color 0.2s,border-color 0.2s; }
        .audit-btn:hover { color:var(--text-mid); border-color:var(--border3); }
        .audit-chev { transition:transform 0.25s; font-size:9px; }
        .audit-chev.open { transform:rotate(180deg); }
        .audit-panel { overflow:hidden; max-height:0; transition:max-height 0.5s ease; margin-top:10px; }
        .audit-panel.open { max-height:8000px; }
        .a-stage { margin-bottom:3px; }
        .a-stage-head { display:flex; align-items:center; gap:12px; padding:11px 16px; background:var(--surface); border:1px solid var(--border); border-radius:4px 4px 0 0; cursor:pointer; transition:background 0.15s; }
        .a-stage-head:hover { background:var(--surface2); }
        .a-stage-head.collapsed { border-radius:4px; margin-bottom:3px; }
        .a-stage-head.collapsed .a-chev { transform:rotate(-90deg); }
        .a-num { font-family:var(--mono); font-size:9px; color:var(--text-dim); min-width:14px; }
        .a-title { font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.10em; color:var(--text-mid); flex:1; }
        .a-badge { font-family:var(--mono); font-size:8px; padding:2px 7px; border-radius:2px; background:var(--ver-green-dim); color:var(--ver-green); border:1px solid rgba(61,122,106,0.2); }
        .a-badge.brass { background:var(--frag-brass-dim); color:var(--frag-brass); border-color:rgba(138,112,64,0.2); }
        .a-chev { font-family:var(--mono); font-size:9px; color:var(--text-dim); transition:transform 0.2s; }
        .a-body { background:var(--surface); border:1px solid var(--border); border-top:none; border-radius:0 0 4px 4px; overflow:hidden; max-height:2000px; transition:max-height 0.4s ease; margin-bottom:3px; }
        .a-body.hidden { max-height:0; }
        .a-inner { padding:16px 18px; }
        .a-desc { font-family:var(--sans); font-size:11px; font-weight:300; color:var(--text-mid); line-height:1.7; }
        footer { border-top:1px solid var(--border); padding:20px 48px; display:flex; align-items:center; justify-content:space-between; position:relative; z-index:1; }
        .footer-note { font-family:var(--mono); font-size:9px; color:var(--text-dim); letter-spacing:0.08em; }
        .footer-link { font-family:var(--mono); font-size:9px; color:var(--text-dim); text-decoration:none; letter-spacing:0.08em; transition:color 0.2s; }
        .footer-link:hover { color:var(--text-mid); }
        @keyframes fadeUp { from{opacity:0;transform:translateY(7px);} to{opacity:1;transform:translateY(0);} }
        @media(max-width:640px) {
            header,footer{padding:0 20px;}
            main{padding:56px 24px 100px;}
            .seal-body{padding:32px 24px 28px;}
            .seal-footer{grid-template-columns:1fr;}
            .seal-field{border-right:none;}
            .seal-field:nth-last-child(-n+2){border-bottom:1px solid var(--border);}
            .seal-field:last-child{border-bottom:none;}
        }
"""

CARD_STYLES = """
        * { margin:0; padding:0; box-sizing:border-box; }
        :root {
            --bg:           #F8F8F6;
            --surface:      #FFFFFF;
            --surface2:     #F2F2EF;
            --border:       rgba(0,0,0,0.08);
            --border2:      rgba(0,0,0,0.13);
            --border3:      rgba(0,0,0,0.20);
            --text:         #111111;
            --text-mid:     #444444;
            --text-dim:     #888888;
            --blue:         #1A3CC2;
            --green:        #0F5C2E;
            --green-bg:     #F2FAF5;
            --green-border: rgba(15,92,46,0.22);
            --red:          #C0111F;
            --red-bg:       #FFF4F4;
            --red-border:   rgba(192,17,31,0.22);
            --mono: 'IBM Plex Mono', monospace;
            --serif: 'GT Super', Georgia, 'Times New Roman', serif;
            --sans: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        html { scroll-behavior: smooth; }
        body { font-family: var(--sans); background: var(--bg); color: var(--text); min-height: 100vh; -webkit-font-smoothing: antialiased; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }

        header { display:flex; align-items:center; justify-content:space-between; padding:0 48px; height:56px; background:var(--bg); border-bottom:1px solid var(--border2); }
        .nav-brand { font-family:var(--serif); font-size:16px; font-weight:400; color:var(--blue); letter-spacing:-0.01em; }
        .nav-label { font-family:var(--mono); font-size:10px; letter-spacing:0.12em; text-transform:uppercase; color:var(--text-dim); }

        main { max-width:680px; margin:0 auto; padding:72px 48px 120px; }

        .doc-eyebrow { font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.18em; color:var(--text-dim); margin-bottom:10px; display:block; opacity:0; animation:fadeUp 0.6s ease forwards 0.05s; }
        .doc-title { font-family:var(--serif); font-size:clamp(20px,3vw,28px); font-weight:400; color:var(--text); line-height:1.25; letter-spacing:-0.02em; margin-bottom:48px; opacity:0; animation:fadeUp 0.6s ease forwards 0.10s; }

        .verdict-block { background:var(--surface); border:1px solid var(--border2); border-radius:8px; overflow:hidden; margin-bottom:48px; opacity:0; animation:fadeUp 0.6s ease forwards 0.15s; }
        .verdict-header { padding:36px 40px 28px; border-bottom:1px solid var(--border); }
        .verdict-label { font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.16em; color:var(--text-dim); margin-bottom:14px; }
        .verdict-text { font-family:var(--serif); font-size:clamp(24px,3.6vw,36px); font-weight:400; line-height:1.1; letter-spacing:-0.025em; margin-bottom:18px; }
        .verdict-text.clean { color:var(--green); }
        .verdict-text.unsat { color:var(--red); }
        .verdict-pill { display:inline-block; font-family:var(--mono); font-size:9px; font-weight:500; letter-spacing:0.12em; text-transform:uppercase; padding:4px 12px; border-radius:3px; }
        .verdict-pill.clean { background:var(--green-bg); color:var(--green); border:1px solid var(--green-border); }
        .verdict-pill.unsat { background:var(--red-bg); color:var(--red); border:1px solid var(--red-border); }
        .verdict-body { padding:24px 40px; }
        .verdict-desc { font-family:var(--sans); font-size:14px; color:var(--text-mid); line-height:1.8; }

        .section { margin-bottom:44px; opacity:0; animation:fadeUp 0.6s ease forwards 0.20s; }
        .section:nth-child(5) { animation-delay:0.25s; }
        .section:nth-child(6) { animation-delay:0.30s; }
        .section:nth-child(7) { animation-delay:0.35s; }
        .section:nth-child(8) { animation-delay:0.40s; }
        .section-label { font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.16em; color:var(--text-dim); margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid var(--border); }
        .section-body { font-family:var(--sans); font-size:14px; color:var(--text-mid); line-height:1.8; }

        .clause-item { border:1px solid var(--border2); border-radius:6px; padding:18px 22px; margin-bottom:8px; background:var(--surface2); }
        .clause-ref { font-family:var(--mono); font-size:9px; text-transform:uppercase; letter-spacing:0.10em; color:var(--text-dim); margin-bottom:10px; }
        .clause-text { font-family:var(--serif); font-size:14px; font-weight:400; color:var(--text); line-height:1.65; font-style:italic; }

        .cert-grid { display:grid; grid-template-columns:1fr 1fr; border:1px solid var(--border2); border-radius:6px; overflow:hidden; background:var(--surface); }
        .cert-field { padding:16px 20px; border-right:1px solid var(--border); border-bottom:1px solid var(--border); }
        .cert-field:nth-child(2n) { border-right:none; }
        .cert-field:nth-last-child(-n+2) { border-bottom:none; }
        .cert-field-label { font-family:var(--mono); font-size:8px; text-transform:uppercase; letter-spacing:0.14em; color:var(--text-dim); margin-bottom:6px; }
        .cert-field-value { font-family:var(--mono); font-size:11px; color:var(--text-mid); word-break:break-all; line-height:1.4; }
        .cert-field-value.blue { color:var(--blue); }

        .caveat { font-family:var(--mono); font-size:10px; color:var(--text-dim); line-height:1.85; padding:20px 24px; background:var(--surface2); border:1px solid var(--border2); border-radius:6px; letter-spacing:0.01em; opacity:0; animation:fadeUp 0.6s ease forwards 0.45s; }

        footer { border-top:1px solid var(--border2); padding:20px 48px; display:flex; align-items:center; justify-content:space-between; }
        .footer-brand { font-family:var(--serif); font-size:13px; color:var(--blue); }
        .footer-note { font-family:var(--mono); font-size:9px; color:var(--text-dim); letter-spacing:0.08em; }

        @media(max-width:640px) {
            header, footer { padding:0 20px; }
            main { padding:48px 24px 80px; }
            .verdict-header { padding:24px; }
            .verdict-body { padding:16px 24px; }
            .cert-grid { grid-template-columns:1fr; }
            .cert-field { border-right:none; }
            .cert-field:nth-last-child(-n+2) { border-bottom:1px solid var(--border); }
            .cert-field:last-child { border-bottom:none; }
        }
"""

_CARD_FINDING = {
    FailureClass.DEONTIC_CONFLICT: (
        "A deontic conflict has been formally proven: the document simultaneously obligates and forbids the same action. "
        "The Validity system has mathematically established that compliance with both commitments is impossible.",
        "No party can comply with all verified commitments simultaneously. "
        "The conflict is formally proven and independently verifiable using the certificate below. "
        "Immediate legal review of the identified clauses is recommended before relying on this document."
    ),
    FailureClass.CONTRADICTION: (
        "A formal contradiction has been proven between two or more commitments in this document. "
        "The Validity system has established that these commitments cannot all be true at the same time.",
        "Parties cannot simultaneously satisfy all verified commitments. "
        "The contradiction is mathematically proven and independently verifiable. "
        "Legal review of the conflicting clauses is recommended."
    ),
    FailureClass.STRESS_EXPOSURE: (
        "A conditional stress exposure has been identified. The document's waterfall or priority structure "
        "contains competing claims that may appear consistent under normal conditions but create an unresolvable "
        "conflict under stress scenarios.",
        "Under stressed conditions, the identified commitments cannot all be honoured simultaneously. "
        "Parties should assess their exposure under downside scenarios and consider whether structural protections are adequate. "
        "Legal review of the waterfall mechanics is recommended."
    ),
    FailureClass.PRIORITY_CYCLE: (
        "A circular priority ordering has been proven: the priority hierarchy defined in this document forms a cycle "
        "where no consistent resolution exists. This is a formal impossibility.",
        "The priority ordering cannot be satisfied in any scenario. "
        "Parties relying on this document's priority structure cannot determine which claims take precedence. "
        "Legal review and restructuring of the priority provisions is required."
    ),
    FailureClass.NUMERIC_IMPOSSIBILITY: (
        "A numeric impossibility has been formally proven: the minimum allocation commitments in this document "
        "sum to more than the available resource pool, making simultaneous satisfaction mathematically impossible.",
        "The minimum commitments defined cannot all be met simultaneously under any distribution of the available resource. "
        "This represents a structural defect that will manifest as a shortfall under normal operating conditions. "
        "Amendment of the minimum allocation provisions is required."
    ),
    FailureClass.TEMPORAL_CONFLICT: (
        "A temporal conflict has been proven: two or more commitments in this document make incompatible claims "
        "about timing or ordering that cannot both be satisfied.",
        "The conflicting timing commitments cannot both be satisfied simultaneously. "
        "This creates a structural impossibility that will manifest on the relevant date. "
        "Legal review of the deadline and timing provisions is recommended."
    ),
    FailureClass.TEMPORAL_INCONSISTENCY: (
        "A temporal inconsistency has been proven: two or more commitments assert contradictory facts about the same point in time.",
        "The document's temporal commitments are internally inconsistent. "
        "Parties relying on these provisions cannot determine the governing rule at the relevant time. "
        "Legal review is recommended."
    ),
    FailureClass.RESOURCE_CONFLICT: (
        "A resource conflict has been proven: competing priority claims on a shared pool cannot all be satisfied simultaneously. "
        "The Validity system has established that the combined obligations exceed what is available.",
        "The resource commitments in this document cannot all be honoured simultaneously. "
        "This will manifest as a shortfall when the relevant obligations fall due. "
        "Legal review of the priority and allocation provisions is recommended."
    ),
    FailureClass.IMPLICATION_LOOP: (
        "An implication loop has been proven: the conditional commitments in this document form a circular dependency "
        "with no consistent base case. The logical chain has no ground truth.",
        "The document's conditional logic cannot be resolved to a consistent state. "
        "Parties relying on these conditional provisions face an unresolvable interpretation problem. "
        "Legal review of the conditional structure is recommended."
    ),
    FailureClass.GENERAL: (
        "A formal contradiction has been proven between commitments in this document. "
        "The Validity system has established that the identified commitments cannot all be true at the same time.",
        "Parties cannot simultaneously satisfy all verified commitments. "
        "The contradiction is formally proven and independently verifiable. "
        "Legal review of the conflicting clauses is recommended."
    ),
}

SCRIPT = """
    function toggleAudit() {
        const p=document.getElementById('auditPanel');
        const c=document.getElementById('auditChev');
        const l=document.getElementById('auditLabel');
        const o=p.classList.toggle('open');
        c.classList.toggle('open',o);
        l.textContent=o?'Close audit trail':'Open audit trail';
    }
    function toggleAS(h) {
        h.classList.toggle('collapsed');
        h.nextElementSibling.classList.toggle('hidden');
    }
    function toggleFrag(h) {
        const body=h.nextElementSibling;
        const chev=h.querySelector('.frag-chevron');
        const open=body.classList.toggle('open');
        chev.classList.toggle('open',open);
    }
"""


# ── Report Renderer ────────────────────────────────────────────────────────────

class ReportRenderer:
    """
    Generates a deployable HTML verification report from a pipeline AnalysisOutput.
    Matches the Validity report design system exactly.
    """

    def render(
        self,
        output:        AnalysisOutput,
        document_name: str,
        document_type: str = "Document",
        permalink:     Optional[str] = None,
    ) -> str:
        """
        Render a complete HTML report string.

        output:        AnalysisOutput from ProofMapper.map()
        document_name: Human-readable document name for the title
        document_type: e.g. "Facility agreement", "Structured finance prospectus"
        permalink:     Optional Jekyll permalink front matter
        """
        primary   = output.primary
        is_unsat  = isinstance(primary, ProofObject)
        timestamp = getattr(primary, 'timestamp', datetime.now(timezone.utc).isoformat())
        doc_hash  = getattr(primary, 'document_hash', '')

        front_matter = f"---\npermalink: {permalink}\n---\n" if permalink else ""

        return f"""{front_matter}<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validity — {html.escape(document_name)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Serif:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Sans:wght@300;400;500&display=swap" rel="stylesheet">
    <style>{STYLES}</style>
</head>
<body>

<header>
    <a href="/"><img src="validity-logo.png" alt="Validity" /></a>
    <span class="nav-label">Verification Report</span>
</header>

<main>

    <span class="doc-eyebrow">Formal Verification</span>
    <h1 class="doc-title">{html.escape(document_name)}</h1>

    {self._render_seal(primary, output, is_unsat, timestamp, doc_hash, document_type)}

    <p class="phil">Formal verification does not determine whether a document is legally sound. It determines whether the commitments it contains can all be true at the same time. That is a different — and prior — question.</p>

    {self._render_contradiction(primary) if is_unsat else ''}

    {self._render_verified(output)}

    {self._render_fragment(output)}

    {self._render_audit(output)}

</main>

<footer>
    <span class="footer-note">validity.live</span>
    <a href="/verify" class="footer-link">← Run verification</a>
</footer>

<script>{SCRIPT}</script>

</body>
</html>"""

    # ── Seal ──────────────────────────────────────────────────────────────────

    def _render_seal(self, primary, output, is_unsat, timestamp, doc_hash, document_type) -> str:
        if is_unsat:
            outcome_text  = primary.failure_class.value
            outcome_class = "contradiction"
            pill_class    = "unsat"
            pill_text     = "UNSAT"
        else:
            outcome_text  = "No contradiction proven."
            outcome_class = ""
            pill_class    = "sat"
            pill_text     = "SAT"

        analysed = output.claims_analysed
        refused  = output.claims_refused
        proof_id = doc_hash[:16] + "…" if doc_hash else "—"

        ts_display = timestamp.replace("T", " ").replace("+00:00", " UTC").split(".")[0] + " UTC"

        return f"""
    <div class="result-seal">
        <div class="seal-body">
            <div class="seal-outcome {outcome_class}">{html.escape(outcome_text)}</div>
            <div class="seal-counts">{analysed} commitments verified &nbsp;·&nbsp; {refused} outside fragment</div>
            <div class="seal-state-row">
                <span class="seal-pill {pill_class}">{pill_text}</span>
                <span class="seal-meta-item">LFS v2.0</span>
                <span class="seal-sep">·</span>
                <span class="seal-meta-item">Z3 4.15.x</span>
                <span class="seal-sep">·</span>
                <span class="seal-meta-item">Proof ID: {html.escape(proof_id)}</span>
            </div>
        </div>
        <div class="seal-footer">
            <div class="seal-field">
                <div class="seal-field-label">Document type</div>
                <div class="seal-field-value">{html.escape(document_type)}</div>
            </div>
            <div class="seal-field">
                <div class="seal-field-label">Timestamp</div>
                <div class="seal-field-value">{html.escape(ts_display)}</div>
            </div>
            <div class="seal-field">
                <div class="seal-field-label">Commitments verified</div>
                <div class="seal-field-value">{analysed}</div>
            </div>
            <div class="seal-field">
                <div class="seal-field-label">Document hash</div>
                <div class="seal-field-value proof">{html.escape(doc_hash[:24] + '…' if doc_hash else '—')}</div>
            </div>
        </div>
    </div>"""

    # ── Contradiction core ────────────────────────────────────────────────────

    def _render_contradiction(self, primary: ProofObject) -> str:
        if not primary.source_spans:
            return ""

        items = ""
        for i, span in enumerate(primary.source_spans, 1):
            location = ""
            if span.page:
                location = f"Page {span.page}"
                if span.section:
                    location += f" &nbsp;·&nbsp; {html.escape(span.section)}"

            items += f"""
            <div class="core-item">
                <div class="core-item-id">Commitment {i}{(' &nbsp;·&nbsp; ' + html.escape(span.section)) if span.section else ''}</div>
                <div class="core-item-text">"{html.escape(span.text)}"</div>
                {f'<div class="core-item-loc">{location}</div>' if location else ''}
            </div>"""

        proof_lines = ""
        for i, span in enumerate(primary.source_spans, 1):
            constraint = f"claim_{i}"
            proof_lines += f'<div class="proof-line">{i}. &nbsp;<span class="c">{html.escape(constraint)}</span></div>\n'

        formal_proof = primary.formal_proof or ""
        constraint_lines = ""
        for line in formal_proof.split("\n"):
            if "Constraint:" in line:
                constraint = line.split("Constraint:")[-1].strip()
                idx = formal_proof.split("\n").index(line)
                constraint_lines += f'<div class="proof-line">{html.escape(constraint)}</div>\n'

        return f"""
    <div class="core-section">
        <div class="sec-head">
            <span class="sec-label">Contradicting Commitments</span>
            <span class="sec-count">{len(primary.source_spans)} in minimal core</span>
        </div>
        {items}
        <div class="proof-block">
            <div class="proof-block-top">
                <span class="proof-block-label">Formal Proof</span>
                <span class="proof-block-state">Z3 · unsat</span>
            </div>
            {constraint_lines or proof_lines}
            <hr class="proof-sep">
            <div class="proof-line"><span class="bot">⊥ &nbsp;These constraints cannot simultaneously hold.</span></div>
        </div>
    </div>"""

    # ── Verified commitments ──────────────────────────────────────────────────

    def _render_verified(self, output: AnalysisOutput) -> str:
        count = output.claims_analysed
        if count == 0:
            return ""

        return f"""
    <div class="verified-section">
        <div class="sec-head">
            <span class="sec-label">Verified Commitments</span>
            <span class="sec-count">{count} passed</span>
        </div>
        <div class="verified-list">
            <div class="verified-item">
                <div>
                    <div class="verified-item-name">{count} commitments</div>
                    <div class="verified-item-desc">All formally verified within LFS v2. No logical contradiction proven among the analysed set.</div>
                </div>
                <div class="verified-dot"></div>
            </div>
        </div>
    </div>"""

    # ── Outside fragment ──────────────────────────────────────────────────────

    def _render_fragment(self, output: AnalysisOutput) -> str:
        if not output.outside_fragment:
            return ""

        count = len(output.outside_fragment)
        rows  = ""

        for i, record in enumerate(output.outside_fragment):
            text   = record.get("text_span", "")
            reason = record.get("reason", "")
            rule   = record.get("rule_violated", "")
            tag, _ = _classify_refusal(reason, rule)
            summary = _frag_summary(text)

            rows += f"""
        <div class="fragment-row">
            <div class="fragment-row-head" onclick="toggleFrag(this)">
                <span class="frag-tag">{html.escape(tag)}</span>
                <span class="frag-rule">{html.escape(rule)}</span>
                <span class="frag-summary">{html.escape(summary)}</span>
                <span class="frag-chevron">▾</span>
            </div>
            <div class="fragment-row-body">
                <div class="frag-clause">{html.escape(text)}</div>
                <div class="frag-reason">{html.escape(reason)}</div>
            </div>
        </div>"""

        return f"""
    <div class="fragment-section">
        <div class="sec-head">
            <span class="sec-label">Outside Fragment</span>
            <span class="sec-count">{count} · human attestation required</span>
        </div>
        <p class="fragment-intro">The system does not refuse judgment arbitrarily. Each refusal has a precise reason. These commitments require human review before they can be relied upon with the same confidence as the verified set.</p>
        {rows}
    </div>"""

    # ── Audit trail ───────────────────────────────────────────────────────────

    def _render_audit(self, output: AnalysisOutput) -> str:
        analysed = output.claims_analysed
        refused  = output.claims_refused
        total    = analysed + refused

        return f"""
    <div class="audit-section">
        <button class="audit-btn" onclick="toggleAudit()">
            <span id="auditLabel">Open audit trail</span>
            <span class="audit-chev" id="auditChev">▾</span>
        </button>
        <div class="audit-panel" id="auditPanel">

            <div class="a-stage">
                <div class="a-stage-head" onclick="toggleAS(this)">
                    <span class="a-num">01</span>
                    <span class="a-title">Commitment Extraction</span>
                    <span class="a-badge">{total} extracted</span>
                    <span class="a-chev">▾</span>
                </div>
                <div class="a-body">
                    <div class="a-inner">
                        <p class="a-desc">{total} candidate commitments extracted via GPT-4o. All confirmed and submitted to Stage 2 translation.</p>
                    </div>
                </div>
            </div>

            <div class="a-stage">
                <div class="a-stage-head collapsed" onclick="toggleAS(this)">
                    <span class="a-num">02</span>
                    <span class="a-title">LFS Fragment Validation</span>
                    <span class="a-badge">{analysed} passed</span>
                    <span class="a-badge brass" style="margin-left:4px;">{refused} refused</span>
                    <span class="a-chev">▾</span>
                </div>
                <div class="a-body hidden">
                    <div class="a-inner">
                        <p class="a-desc">{analysed} commitments validated within LFS v2. {refused} refused outside fragment.</p>
                    </div>
                </div>
            </div>

            <div class="a-stage">
                <div class="a-stage-head collapsed" onclick="toggleAS(this)">
                    <span class="a-num">03</span>
                    <span class="a-title">SMT Encoding + Z3 Solver</span>
                    <span class="a-badge">{'UNSAT' if isinstance(output.primary, ProofObject) else 'SAT'}</span>
                    <span class="a-chev">▾</span>
                </div>
                <div class="a-body hidden">
                    <div class="a-inner">
                        <p class="a-desc">{analysed} validated commitments encoded as named Boolean assertions via assert_and_track. Z3 returned {'unsat — contradiction proven.' if isinstance(output.primary, ProofObject) else 'sat — no contradiction proven.'}</p>
                    </div>
                </div>
            </div>

        </div>
    </div>"""

    def write(
        self,
        output:        AnalysisOutput,
        document_name: str,
        output_path:   str,
        document_type: str = "Document",
        permalink:     Optional[str] = None,
    ) -> str:
        """
        Render and write the HTML report to output_path.
        Returns the output_path for chaining.
        """
        html_content = self.render(output, document_name, document_type, permalink)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path

    # ── Client report card ────────────────────────────────────────────────────

    def render_card(
        self,
        output:        AnalysisOutput,
        document_name: str,
        document_type: str = "Document",
    ) -> str:
        """
        Render a client-facing report card.  Light validity.live design:
        #F8F8F6 background, GT Super serif, IBM Plex Mono, #1A3CC2 blue.
        """
        primary      = output.primary
        is_proof     = isinstance(primary, ProofObject)
        is_cond      = is_proof and getattr(primary, 'verdict', '') == 'conditional'
        timestamp    = getattr(primary, 'timestamp', datetime.now(timezone.utc).isoformat())
        doc_hash     = getattr(primary, 'document_hash', '')

        if not is_proof:
            verdict_key   = "clean"
            verdict_text  = "No Contradiction Found"
            pill_class    = "clean"
            pill_text     = "CLEAN"
        elif is_cond:
            verdict_key   = "conditional"
            verdict_text  = "Stress Exposure Detected"
            pill_class    = "unsat"
            pill_text     = "CONDITIONAL"
        else:
            verdict_key   = "unsat"
            verdict_text  = "Contradiction Proven"
            pill_class    = "unsat"
            pill_text     = "UNSAT"

        failure_class = getattr(primary, 'failure_class', None)

        if is_proof and failure_class in _CARD_FINDING:
            finding_desc, implications = _CARD_FINDING[failure_class]
        elif is_proof:
            finding_desc  = (
                "A formal contradiction has been proven between commitments in this document. "
                "The Validity system has established that the identified commitments cannot all be true at the same time."
            )
            implications  = (
                "Parties cannot simultaneously satisfy all verified commitments. "
                "The contradiction is formally proven and independently verifiable. "
                "Legal review of the conflicting clauses is recommended."
            )
        else:
            finding_desc  = (
                "This document has been formally verified. All extractable commitments were analysed within the "
                "Validity Logical Framework. No commitments were found to be simultaneously impossible — "
                "the document is internally consistent within the verified scope."
            )
            implications  = (
                "Parties relying on this document can have formal assurance that its verified obligations are "
                "not self-contradictory. This verification establishes logical consistency, not legal validity. "
                "Commitments outside the supported fragment require separate human attestation."
            )

        ts_display = timestamp.replace("T", " ").replace("+00:00", " UTC").split(".")[0] + " UTC"
        proof_id   = doc_hash[:16] + "…" if doc_hash else "—"
        hash_disp  = doc_hash[:32] + "…" if len(doc_hash) > 32 else (doc_hash or "—")

        source_clauses_html = self._card_source_clauses(primary) if is_proof else ""

        scope_html = (
            f"{output.claims_analysed} commitment{'s' if output.claims_analysed != 1 else ''} "
            f"{'were' if output.claims_analysed != 1 else 'was'} formally analysed using the Validity Logical Framework "
            f"(LFS v2.0) with the Z3 SMT solver. "
        )
        if output.claims_refused:
            scope_html += (
                f"{output.claims_refused} additional commitment{'s' if output.claims_refused != 1 else ''} "
                f"{'were' if output.claims_refused != 1 else 'was'} identified as outside the supported logical "
                f"fragment — these require human attestation and are not covered by this certificate."
            )
        else:
            scope_html += "All extracted commitments were within the supported logical fragment."

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validity — {html.escape(document_name)} · Report Card</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>{CARD_STYLES}</style>
</head>
<body>

<header>
    <span class="nav-brand">Validity</span>
    <span class="nav-label">Verification Report Card</span>
</header>

<main>

    <span class="doc-eyebrow">Formal Verification · Report Card</span>
    <h1 class="doc-title">{html.escape(document_name)}</h1>

    <div class="verdict-block">
        <div class="verdict-header">
            <div class="verdict-label">Verdict</div>
            <div class="verdict-text {verdict_key}">{html.escape(verdict_text)}</div>
            <span class="verdict-pill {pill_class}">{html.escape(pill_text)}</span>
        </div>
        <div class="verdict-body">
            <p class="verdict-desc">{html.escape(finding_desc)}</p>
        </div>
    </div>

    {source_clauses_html}

    <div class="section">
        <div class="section-label">What This Means</div>
        <div class="section-body">{html.escape(implications)}</div>
    </div>

    <div class="section">
        <div class="section-label">Scope of Verification</div>
        <div class="section-body">{html.escape(scope_html)}</div>
    </div>

    <div class="section">
        <div class="section-label">Verification Certificate</div>
        <div class="cert-grid">
            <div class="cert-field">
                <div class="cert-field-label">Proof ID</div>
                <div class="cert-field-value blue">{html.escape(proof_id)}</div>
            </div>
            <div class="cert-field">
                <div class="cert-field-label">Verdict</div>
                <div class="cert-field-value">{html.escape(pill_text)}</div>
            </div>
            <div class="cert-field">
                <div class="cert-field-label">Document Hash (SHA-256)</div>
                <div class="cert-field-value">{html.escape(hash_disp)}</div>
            </div>
            <div class="cert-field">
                <div class="cert-field-label">Timestamp</div>
                <div class="cert-field-value">{html.escape(ts_display)}</div>
            </div>
            <div class="cert-field">
                <div class="cert-field-label">Framework</div>
                <div class="cert-field-value">LFS v{html.escape(getattr(primary, 'lfs_version', '2.0'))}</div>
            </div>
            <div class="cert-field">
                <div class="cert-field-label">Solver</div>
                <div class="cert-field-value">Z3 4.15.x</div>
            </div>
        </div>
    </div>

    <div class="caveat">
        This report card is produced by an automated formal verification system. It establishes logical consistency
        or inconsistency within the scope described above — it does not constitute legal advice, a legal opinion,
        or a determination of contractual enforceability. Findings should be reviewed by qualified legal counsel
        before any reliance is placed upon them. Validity's verification is limited to commitments expressible
        within the supported logical fragment; commitments outside that fragment are not covered by this certificate.
    </div>

</main>

<footer>
    <span class="footer-brand">Validity</span>
    <span class="footer-note">validity.live</span>
</footer>

</body>
</html>"""

    def _card_source_clauses(self, primary: ProofObject) -> str:
        if not primary.source_spans:
            return ""
        items = ""
        for i, span in enumerate(primary.source_spans, 1):
            ref_parts = [f"Clause {i}"]
            if span.section:
                ref_parts.append(html.escape(span.section))
            if span.page:
                ref_parts.append(f"Page {span.page}")
            ref_label = " &nbsp;·&nbsp; ".join(ref_parts)
            items += f"""
        <div class="clause-item">
            <div class="clause-ref">{ref_label}</div>
            <div class="clause-text">{html.escape(span.text)}</div>
        </div>"""

        return f"""
    <div class="section">
        <div class="section-label">Source Clauses</div>
        {items}
    </div>"""

    def write_card(
        self,
        output:        AnalysisOutput,
        document_name: str,
        output_path:   str,
        document_type: str = "Document",
    ) -> str:
        """
        Render and write the client report card to output_path.
        Returns the output_path for chaining.
        """
        html_content = self.render_card(output, document_name, document_type)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import uuid
    sys.path.insert(0, os.path.dirname(__file__))

    from fragment_validator import FragmentValidator, make_claim, ClaimType, ValidationStatus, RejectionRecord, Provenance, FormalClaim
    from z3_encoder import Z3Encoder
    from solver_interface import SolverInterface
    from proof_mapper import ProofMapper

    validator = FragmentValidator()
    encoder   = Z3Encoder()
    solver_if = SolverInterface()
    mapper    = ProofMapper()
    renderer  = ReportRenderer()

    # Build a contradiction
    c1 = make_claim(
        text_span  = "The Borrower shall not make any distributions to shareholders during the term of this Agreement.",
        formula    = {"type": "Forbidden", "operand": {"type": "Predicate", "name": "distribute", "args": ["borrower"]}},
        claim_type = ClaimType.OBLIGATION,
        page=4, section="Section 4.3",
    )
    c2 = make_claim(
        text_span  = "The Borrower shall make distributions to shareholders during the term of this Agreement.",
        formula    = {"type": "Obligated", "operand": {"type": "Predicate", "name": "distribute", "args": ["borrower"]}},
        claim_type = ClaimType.OBLIGATION,
        page=9, section="Section 6.1",
    )

    # Refused claim
    c3 = make_claim(
        text_span  = "The Borrower shall act in a reasonable manner.",
        formula    = {"type": "Obligated", "operand": {"type": "Predicate", "name": "act", "args": ["borrower"]}},
        claim_type = ClaimType.OBLIGATION,
    )

    validator.validate(c1)
    validator.validate(c2)
    validator.validate(c3)

    encoded = [encoder.encode(c) for c in [c1, c2]]
    result  = solver_if.solve(encoded)
    output  = mapper.map(result, [c1, c2, c3], document_text="sample facility agreement text")

    path = renderer.write(
        output        = output,
        document_name = "Sample Facility Agreement",
        output_path   = "/tmp/validity-test-report.html",
        document_type = "Facility agreement",
        permalink     = "/demo-sample",
    )

    print(f"Report written to: {path}")
    print(f"Verdict: {output.primary.verdict if hasattr(output.primary, 'verdict') else 'sat'}")
    print(f"Claims analysed: {output.claims_analysed}")
    print(f"Claims refused: {output.claims_refused}")
    print("Report renderer smoke test complete.")
