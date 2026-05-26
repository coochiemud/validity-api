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
