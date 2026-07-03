# Evaluation Workflow

This project separates tuning from final reporting so retrieval and citation fixes do not overfit the benchmark.

## 1. Dev Set

- Document: `PerezPerez2020.pdf`
- Questions: `evaluation/questions.csv`
- Role: development and hard stress-test
- Allowed: tune chunking, retrieval, reranker, prompt, citation, and profile config
- Not allowed: hardcode question-specific fixes or copy expected answers into the pipeline

Useful commands:

```powershell
python evaluation\debug_retrieval.py --questions evaluation\questions.csv --limit 10 --top-k 5
python evaluation\ab_test_retrieval.py --questions evaluation\questions.csv --limit 30 --top-k 5 --output evaluation\results\retrieval_ablation_perez.csv
python evaluation\run_custom_eval.py --questions evaluation\questions.csv --limit 30 --refresh-cache
```

## 2. Freeze Config

After choosing a retrieval/profile configuration, do not keep tuning from holdout failures. Record:

- `DOCUMENT_PROFILE`
- `EMBEDDING_MODEL`
- `RERANKER_MODEL`
- `USE_MULTI_QUERY`
- `MULTI_QUERY_COUNT`
- `TOP_K`
- `SIMILARITY_THRESHOLD`
- chunking and parent context settings

Recommended output: `evaluation/results/frozen_config.json`.

Command:

```powershell
python evaluation\freeze_config.py --notes "selected after Perez dev retrieval/citation debug"
```

## 3. Holdout Validation

- Document: `w18347.pdf`
- Questions: `evaluation/questions_w18347.csv`
- Role: behavior/retrieval holdout validation

Run custom eval first. RAGAS can be run here as a holdout check, but do not tune from these failures:

```powershell
python evaluation\run_custom_eval.py --questions evaluation\questions_w18347.csv --limit 30 --refresh-cache --output evaluation\results\custom_eval_w18347.csv --summary evaluation\results\custom_eval_w18347.md
python -m evaluation.evaluate --questions evaluation\questions_w18347.csv --judge-provider ollama --judge-model llama3.1:8b --sleep-seconds 0 --output evaluation\results\ragas_w18347.csv --report-output evaluation\results\ragas_w18347.md
```

## 4. Final RAGAS / Cross-Document Eval

- Document: `2302.06590v1.pdf`
- Questions: `evaluation/questions_2302_06590.csv`
- Role: final RAGAS report candidate and cross-document generalization check

Run custom eval first, then RAGAS. RAGAS averages are computed only on answerable, reasoning, and claim-verification rows; true out-of-scope and unsupported-claim rows remain custom behavior metrics.

```powershell
python evaluation\run_custom_eval.py --questions evaluation\questions_2302_06590.csv --limit 30 --refresh-cache --output evaluation\results\custom_eval_2302_06590.csv --summary evaluation\results\custom_eval_2302_06590.md
python -m evaluation.evaluate --questions evaluation\questions_2302_06590.csv --judge-provider ollama --judge-model llama3.1:8b --sleep-seconds 0 --output evaluation\results\ragas_2302_06590.csv --report-output evaluation\results\ragas_2302_06590.md
```

## 5. Report

Every report should say:

- Which profile was used
- Which document and question set were used
- Whether the result is dev, holdout, or cross-document
- Which metrics are retrieval-only proxies and which are answer-level metrics
- Any model/quota/latency limitations
