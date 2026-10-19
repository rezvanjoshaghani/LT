# Phase 5 changes after evaluation

Changes made after the evaluation at commit E,
`acc86ba178958053dcf5d56e9050962d86e52528`. `reporting_rules.md` decision 1
requires every reporting change after E to be named here with its reason.
Acceptance requires the same of a test file whose results it reads. Each item
starts with the path in backticks, and its reason follows on the same line.

- `tests/test_phase5_train.py`: test_a_fit_masked_by_dropout_passes_the_amended_gate assumed that, in a 60-step run of the tiny model at seed 0, the best dropout-off score beats the best dropout-on score. That held by 0.0032 where the test was written, failed on the Borah CPU node where the first acceptance run executed the suite (0.43629 against 0.43768), and fails for two of the first eight seeds even where it was written, so it was never a property of the gate. The test now takes the first of 16 seeds whose run shows the effect by at least 0.001 and checks the amended overfit gate on that seed. Its assertions about the gate are unchanged, and no code under test changed.
- `tests/test_phase5_acceptance.py`: the repository its tests build to stand for E copied every Markdown file under validation/evidence/phase5 from the checkout, so this record, written after E, leaked into it, and test_the_suite_files_read_at_r_are_es_own saw a reason for an unchanged file. The copy now leaves this record out, and a test that needs one writes its own. Acceptance does not read this file's results, so it is listed for completeness.
