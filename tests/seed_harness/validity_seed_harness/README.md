# Validity Seed Harness

This package contains four controlled seed cases for the Validity formal-verification pipeline. Each case uses a public SEC filing extract as the clean control and a synthetic seeded variant as the test document. The source documents are not alleged to contain contradictions; contradictions are deliberately injected for testing.

## Cases

- T001_numeric_impossible_structured_finance: Numeric Impossibility / Resource Conflict (unsat)
- T002_ordinal_circular_intercreditor: Ordinal Circularity / Priority Cycle (unsat)
- T003_definitional_circular_transfer_terms: Definitional Circularity / Ungrounded Definition Graph (outside_fragment_or_definition_graph_failure)
- T004_temporal_overlap_post_closing: Temporal Conflict / Impossible Deadline (unsat)