from evaluation.debug_utils import DebugThresholds
from evaluation.run_custom_eval import summarize_rows


def test_summarize_rows_separates_keyword_page_and_evidence_metrics() -> None:
    rows = [
        {
            "is_answerable": True,
            "source_pages": "1",
            "keyword_hit_rate_at_5": "0.8000",
            "keyword_recall_at_5": True,
            "page_recall_at_5": False,
            "evidence_hit_at_5": True,
            "mrr": "0.0000",
            "citation_accuracy": "1.0000",
            "refusal_accuracy": "",
            "false_refusal": False,
            "latency_seconds": "2.0",
        },
        {
            "is_answerable": False,
            "source_pages": "",
            "keyword_hit_rate_at_5": "0.0000",
            "keyword_recall_at_5": False,
            "page_recall_at_5": False,
            "evidence_hit_at_5": False,
            "mrr": "0.0000",
            "citation_accuracy": "",
            "refusal_accuracy": "1.0000",
            "false_refusal": False,
            "latency_seconds": "1.0",
        },
    ]

    summary = summarize_rows(rows, DebugThresholds())

    assert summary["mean_keyword_hit_rate_at_5"] == 0.8
    assert summary["keyword_recall_at_5"] == 1.0
    assert summary["page_recall_at_5"] == 0.0
    assert summary["evidence_hit_at_5"] == 1.0
    assert summary["refusal_accuracy"] == 1.0


def test_summarize_rows_reports_missing_refusal_metric_as_none() -> None:
    rows = [
        {
            "is_answerable": True,
            "source_pages": "1",
            "keyword_hit_rate_at_5": "0.8000",
            "keyword_recall_at_5": True,
            "page_recall_at_5": True,
            "evidence_hit_at_5": True,
            "mrr": "1.0000",
            "citation_accuracy": "1.0000",
            "refusal_accuracy": "",
            "false_refusal": False,
            "latency_seconds": "2.0",
        }
    ]

    summary = summarize_rows(rows, DebugThresholds())

    assert summary["refusal_accuracy"] is None
