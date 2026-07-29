# Deterministic Benchmark Suite

This suite measures whether the reviewer finds real defects while minimizing fabricated findings.
It runs against isolated synthetic repositories under benchmarks/cases.

## Why cases are isolated

Benchmark defects are intentionally vulnerable or broken code snippets. They are kept isolated in
benchmark fixture repositories and never merged into production modules. This prevents training
or QA fixtures from contaminating runtime application code.

## Case layout

Each case is an independent mini repository:

- base/: baseline commit contents
- head/: PR commit contents
- expected.json: ground truth
- README.md: short case intent

## Running

Run full suite:

python benchmarks/run_benchmarks.py \
  --provider ollama \
  --model qwen2.5-coder:7b \
  --review-mode consolidated \
  --output benchmarks/results/qwen-7b.json

Run one case:

python benchmarks/run_benchmarks.py \
  --provider fake \
  --case clean_fake_llm_client \
  --output benchmarks/results/single-case.json

Useful options:

- --case CASE_ID
- --provider fake|ollama
- --model MODEL
- --review-mode consolidated|separate
- --llm-timeout SECONDS
- --max-review-seconds SECONDS
- --candidate-findings-output PATH
- --candidate-findings-input PATH
- --sampling-seed INT
- --sampling-temperature FLOAT
- --sampling-top-p FLOAT
- --keep-temp
- --verbose

## Candidate Replay For Verification

You can export candidate findings from an end-to-end run and later replay those exact
candidates to evaluate verification independently from reviewer variance.

Export candidate findings:

python benchmarks/run_benchmarks.py \
  --provider ollama \
  --model qwen2.5-coder:7b \
  --candidate-findings-output benchmarks/results/candidates.qwen-7b.json \
  --output benchmarks/results/end-to-end.qwen-7b.json

Run verification-only replay:

python benchmarks/run_benchmarks.py \
  --provider ollama \
  --model qwen2.5-coder:7b \
  --candidate-findings-input benchmarks/results/candidates.qwen-7b.json \
  --verify-findings \
  --output benchmarks/results/verification-only.qwen-7b.json

Replay outputs include:

- before verification TP/FP/FN (candidate stage)
- after verification TP/FP/FN (verified stage)
- after publishing TP/FP/FN (published stage)
- delta metrics (TP removed, FP removed, new FN, precision/recall change)
- per-finding verification impact

Fixed calibration candidate replay input:

- benchmarks/datasets/verification_calibration_qwen7b_candidates.json

Quick compare helper:

python benchmarks/compare_verification_runs.py \
  benchmarks/results/verification-only.qwen-7b.minconf-0.8.json \
  benchmarks/results/verification-only.qwen-7b.minconf-0.6.json

## Scoring

The scorecard includes:

- true positives (TP)
- false positives (FP)
- false negatives (FN)
- precision, recall, F1
- clean-case false-positive rate
- valid file and line rate
- parse success rate
- elapsed time
- correct findings per minute

Matching is deterministic and simple:

1. file equality
2. finding line within expected range plus/minus a tolerance (default 5 lines)
3. category equality
4. concept keyword overlap after normalization

One finding can match only one expected defect, and each expected defect can be matched once.
Unmatched findings are false positives. Unmatched expected defects are false negatives.

For clean cases, forbidden concepts in generated findings are flagged as hallucinations.

## Quality gate recommendations

These are recommendations, not universal truths:

- precision >= 0.80
- clean-case false-positive rate <= 0.10
- valid location rate >= 0.95
- parse success rate >= 0.99

## Compare two models

Run each model to separate output files and compare metrics side by side:

- benchmarks/results/model-a.json
- benchmarks/results/model-b.json

Start by optimizing precision and clean-case false-positive rate. Early on, high precision is more
important than high recall to avoid eroding trust with fabricated findings.

## Known limitations

- concept matching is keyword based and can miss semantic equivalents
- category matching is strict and does not use ontology mapping
- line-based localization with tolerance cannot validate deeper control-flow reasoning
