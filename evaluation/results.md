# Core RAG Evaluation Results

Framework: `ragas`
Judge model: `llama3.1:8b`
Total questions: 15
Successful generations: 15
Answerable questions: 15
Unanswerable questions: 0
Easy questions: 5
Medium questions: 9
Hard questions: 1

## Core Metrics

- Context Precision: 71.3%
- Context Recall: 69.6%
- Answer Relevancy: 69.0%
- Faithfulness: 46.5%
- Citation Accuracy: 53.3%
- Refusal Accuracy: 
- Average Latency: 9.54s

## Per-question Results

| # | Difficulty | Type | Latency | CtxPrec | CtxRecall | AnsRel | Faith | Cite | Refusal | Error | Question | Answer |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 | easy | answerable | 10.42s | 88.7% | 16.7% | 86.1% | 14.3% | 0.0% |  |  | What data source and sample period does the paper use? | The paper uses Colombia’s National Household Survey (ENH) as its primary data source, specifically for the years 1996-2000. The survey is a quarterly rotating cross-sectional su... |
| 2 | easy | answerable | 9.72s | 77.1% |  | 88.2% | 50.0% | 100.0% |  |  | How does the paper define informal workers? | According to the provided document [3], an informal worker is defined as someone who does not have access to health insurance provided by their employer. |
| 3 | easy | answerable | 7.61s | 20.0% | 71.4% | 0.0% | 50.0% | 0.0% |  |  | What institutional process determined Colombia's minimum wage around the study period? | [1] Arango et al. (2008b) [2] Maloney and Mendez (2004) |
| 4 | easy | answerable | 5.28s | 20.0% | 41.4% | 77.0% | 75.0% | 100.0% |  |  | What empirical shock does the paper use for identification? | The paper uses an inflation forecast error in 1999, which produced a large unexpected real minimum wage increase, as the empirical shock for identification [1]. This shock is co... |
| 5 | easy | answerable | 5.56s | 100.0% | 100.0% | 43.7% | 33.3% | 0.0% |  |  | What are the main outcomes studied in the paper? | The paper focuses on estimating the effects of minimum wages on the unconditional distribution of wages in both formal and informal sectors, which is a novel approach compared t... |
| 6 | medium | answerable | 17.36s | 100.0% |  | 94.8% | 70.0% | 0.0% |  |  | Why is Colombia a useful setting for studying minimum wage effects? | Colombia is a useful setting for studying minimum wage effects because it has several unique features. Firstly, Colombia has had high and stable minimum wages over many years, c... |
| 7 | medium | answerable | 13.17s | 26.7% |  | 64.2% | 40.0% | 100.0% |  |  | How did the financial crisis create both identification and a threat to identification? | The financial crisis created both a challenge and a threat to identification of the minimum wage's impact. Firstly, it caused an unexpected change in the real value of the minim... |
| 8 | medium | answerable | 6.45s | 100.0% |  | 80.0% | 60.0% | 100.0% |  |  | What methods does the paper combine to estimate effects across the wage distribution? | This methodology is used to estimate the effect of the change in the minimum wage on the unconditional wage distributions. The unconditional quantile regression method estimates... |
| 9 | medium | answerable | 10.95s | 91.7% | 100.0% | 90.3% | 33.3% | 100.0% |  |  | How is minimum wage incidence measured in the baseline design? | Minimum wage incidence is measured using two different variables: "fraction affected" and "fraction at". "Fraction affected" is the proportion of workers affected by the minimum... |
| 10 | medium | answerable | 8.88s | 100.0% |  | 70.6% | 0.0% | 100.0% |  |  | What does the paper find for formal wages? | The author finds evidence of wage increase in the lower part of the formal wage distribution. [3] |
| 11 | medium | answerable | 12.00s | 75.6% | 100.0% | 87.9% | 100.0% | 0.0% |  |  | How should the formal wage estimates be interpreted relative to full compliance? | The formal wage estimates suggest that employers partially comply with the minimum wage policy and use it as a reference point. This is because the estimated increases in wages... |
| 12 | medium | answerable | 8.96s | 100.0% | 0.0% | 74.7% | 87.5% | 0.0% |  |  | What does the paper find for informal wages? | The paper finds that higher minimum wage incidence leads to wage increases in the informal sector around median wages, around the minimum wage. These effects are smaller and les... |
| 13 | medium | answerable | 7.16s | 70.0% |  | 95.6% | 50.0% | 0.0% |  |  | Does the paper find evidence that informal wage effects are caused by formal-to-informal spillovers? | The paper does not find evidence that informal wage effects are caused by formal-to-informal spillovers. In fact, it suggests that the reaction of informal wages to the minimum... |
| 14 | medium | answerable | 7.58s | 0.0% | 96.7% | 0.0% | 0.0% | 100.0% |  |  | What does the paper find about employment effects in each sector? | [1], [2] |
| 15 | hard | answerable | 12.01s | 100.0% | 100.0% | 81.3% | 33.3% | 100.0% |  |  | How does the author interpret the informal employment result? | The author suggests that the informal employment result is not explained by spillover effects from the formal labor market. Instead, the author proposes that lower flexibility i... |
