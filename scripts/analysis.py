"""
Comparative Analysis: Regular vs. Novel-View-Augmented CDViews
===============================================================

Reuses the error categorisation logic from wrong_answers.py and adds
a side-by-side comparison of the two pipelines, including tracking of
how many novel views were selected by the ViewSelector.

Public API
----------
    compare_and_report(predictions_regular, predictions_augmented,
                       ground_truth, novel_tracker, output_dir, tag)
"""

import json
import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
#  Answer matching  (same logic as wrong_answers.py)
# ---------------------------------------------------------------------------

def is_answer_correct(pred_answer: str, gt_answers: List[str]) -> bool:
    """Flexible match: exact, substring, or >50% word overlap."""
    pred = pred_answer.lower().strip()
    gt_lower = [a.lower().strip() for a in gt_answers]

    for gt_a in gt_lower:
        if pred == gt_a:
            return True
        if pred in gt_a or gt_a in pred:
            return True
        pred_words = set(pred.split())
        gt_words = set(gt_a.split())
        if len(pred_words) > 1 and len(gt_words) > 1:
            overlap = len(pred_words & gt_words) / max(len(pred_words), len(gt_words))
            if overlap > 0.5:
                return True
    return False


# ---------------------------------------------------------------------------
#  Error categorisation  (same logic as wrong_answers.py)
# ---------------------------------------------------------------------------

_COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "brown",
    "gray", "grey", "orange", "purple", "pink", "beige", "tan",
]
_NUMBERS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
]
_SPATIAL = [
    "left", "right", "front", "behind", "above", "below", "near",
    "next to", "between", "under", "over", "beside", "corner",
]
_OBJECTS = [
    "chair", "table", "door", "window", "bed", "desk", "sofa",
    "couch", "lamp", "shelf", "cabinet", "tv", "monitor", "computer",
    "refrigerator", "sink", "toilet", "shower", "bathtub", "mirror",
]
_STOPWORDS = {"the", "a", "an", "is", "are", "it", "in", "on", "of", "to"}


def categorize_error(pred: str, gt_answers: List[str], question: str) -> str:
    """Return an error-type string for a wrong answer."""
    pred = pred.lower()
    gt_lower = [g.lower() for g in gt_answers]
    question = question.lower()

    # Color
    pc = [c for c in _COLORS if c in pred]
    gc = [c for c in _COLORS if any(c in g for g in gt_lower)]
    if pc and gc and set(pc) != set(gc):
        return "wrong_color"
    if "color" in question and pc and not gc:
        return "hallucinated_color"

    # Count
    pn = [n for n in _NUMBERS if n in pred.split()]
    gn = [n for n in _NUMBERS if any(n in g.split() for g in gt_lower)]
    if pn and gn and set(pn) != set(gn):
        return "wrong_count"
    if ("how many" in question or "number" in question) and pn:
        return "wrong_count"

    # Spatial
    if any(s in question for s in _SPATIAL):
        return "wrong_spatial"

    # Object
    po = [o for o in _OBJECTS if o in pred]
    go = [o for o in _OBJECTS if any(o in g for g in gt_lower)]
    if po and go and set(po) != set(go):
        return "wrong_object"
    if po and not go:
        return "hallucinated_object"

    # Yes/No
    if pred in ("yes", "no") or any(g in ("yes", "no") for g in gt_lower):
        return "wrong_yes_no"

    # Unrelated
    pw = set(pred.split())
    gw = set(" ".join(gt_lower).split())
    if len(pw - gw - _STOPWORDS) == len(pw - _STOPWORDS) and len(pw) > 0:
        return "unrelated_hallucination"

    return "other"


# ---------------------------------------------------------------------------
#  Single-pipeline analysis
# ---------------------------------------------------------------------------

def _analyse_one_pipeline(
    predictions: List[Dict],
    ground_truth: List[Dict],
    label: str,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Analyse a single pipeline's predictions against ground truth.

    Returns (summary_dict, correct_list, wrong_list).
    """
    gt_dict = {str(g["question_id"]): g for g in ground_truth}
    pred_dict = {str(p["question_id"]): p for p in predictions}

    correct, wrong = [], []

    for qid, gt in gt_dict.items():
        if qid not in pred_dict:
            continue
        pred_answer = pred_dict[qid]["answer_top10"][0].lower().strip()
        gt_answers = gt["answers"]

        entry = {
            "question_id": qid,
            "scene_id": gt["scene_id"],
            "question": gt["question"],
            "situation": gt.get("situation", ""),
            "predicted": pred_answer,
            "ground_truth": gt_answers,
        }

        if is_answer_correct(pred_answer, gt_answers):
            correct.append(entry)
        else:
            entry["error_type"] = categorize_error(pred_answer, gt_answers, gt["question"])
            wrong.append(entry)

    n = len(correct) + len(wrong)
    accuracy = 100 * len(correct) / n if n > 0 else 0

    # Error distribution
    error_counts = defaultdict(int)
    for w in wrong:
        error_counts[w["error_type"]] += 1

    # Hallucination total
    hallucination_cats = {
        "hallucinated_color", "hallucinated_object", "unrelated_hallucination",
    }
    total_hallucination = sum(
        error_counts[c] for c in hallucination_cats if c in error_counts
    )

    summary = {
        "label": label,
        "total": n,
        "correct": len(correct),
        "wrong": len(wrong),
        "accuracy": round(accuracy, 2),
        "total_hallucination": total_hallucination,
        "error_distribution": dict(sorted(error_counts.items(), key=lambda x: -x[1])),
    }

    return summary, correct, wrong


# ---------------------------------------------------------------------------
#  Novel-view selection statistics
# ---------------------------------------------------------------------------

def _compute_novel_stats(tracker: List[Dict]) -> Dict:
    """Aggregate novel-view selection statistics from the per-question tracker."""
    if not tracker:
        return {}

    total_questions = len(tracker)
    questions_with_novel = sum(1 for t in tracker if t["novel_selected"] > 0)
    total_novel_selected = sum(t["novel_selected"] for t in tracker)
    total_novel_available = sum(t["total_novel_available"] for t in tracker)
    novel_counts = [t["novel_selected"] for t in tracker]

    # Distribution of how many novel views were selected per question
    count_dist = Counter(novel_counts)

    return {
        "total_questions": total_questions,
        "questions_with_novel_selected": questions_with_novel,
        "pct_questions_with_novel": round(100 * questions_with_novel / total_questions, 1)
            if total_questions > 0 else 0,
        "total_novel_selected": total_novel_selected,
        "total_novel_available": total_novel_available,
        "avg_novel_per_question": round(total_novel_selected / total_questions, 2)
            if total_questions > 0 else 0,
        "max_novel_per_question": max(novel_counts) if novel_counts else 0,
        "novel_count_distribution": dict(sorted(count_dist.items())),
    }


# ---------------------------------------------------------------------------
#  Comparative reporting
# ---------------------------------------------------------------------------

def _print_divider(char="─", width=60):
    print(char * width)


def _print_summary(summary: Dict):
    """Print a formatted summary for one pipeline."""
    print(f"\n  Pipeline : {summary['label']}")
    print(f"  Total    : {summary['total']}")
    print(f"  Correct  : {summary['correct']}")
    print(f"  Wrong    : {summary['wrong']}")
    print(f"  Accuracy : {summary['accuracy']:.2f}%")
    print(f"  Total hallucinations: {summary['total_hallucination']}")
    print()
    print(f"  {'Error Category':<30s} {'Count':>6s} {'%':>7s}")
    print(f"  {'─'*30} {'─'*6} {'─'*7}")
    for cat, cnt in summary["error_distribution"].items():
        pct = 100 * cnt / summary["wrong"] if summary["wrong"] > 0 else 0
        print(f"  {cat:<30s} {cnt:>6d} {pct:>6.1f}%")


def compare_and_report(
    predictions_regular: List[Dict],
    predictions_augmented: List[Dict],
    ground_truth: List[Dict],
    novel_tracker: Optional[List[Dict]] = None,
    output_dir: str = ".",
    tag: str = "eval",
):
    """Run analysis on both pipelines and print a comparative report.

    Parameters
    ----------
    novel_tracker : list of dicts, optional
        Per-question novel-view selection info from evaluate_pipeline.py.
    """

    print(f"\n{'='*60}")
    print("  COMPARATIVE ANALYSIS: Regular vs. Novel-View Augmented")
    print(f"{'='*60}")

    sum_reg, correct_reg, wrong_reg = _analyse_one_pipeline(
        predictions_regular, ground_truth, "Regular CDViews"
    )
    sum_aug, correct_aug, wrong_aug = _analyse_one_pipeline(
        predictions_augmented, ground_truth, "Augmented CDViews (+ novel views)"
    )

    _print_divider("═")
    _print_summary(sum_reg)
    _print_divider()
    _print_summary(sum_aug)
    _print_divider("═")

    # ── Delta table ──
    delta_acc = sum_aug["accuracy"] - sum_reg["accuracy"]
    delta_hall = sum_aug["total_hallucination"] - sum_reg["total_hallucination"]
    delta_wrong = sum_aug["wrong"] - sum_reg["wrong"]

    print(f"\n  {'Metric':<30s} {'Regular':>10s} {'Augmented':>10s} {'Delta':>10s}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Accuracy (%)':<30s} {sum_reg['accuracy']:>10.2f} {sum_aug['accuracy']:>10.2f} {delta_acc:>+10.2f}")
    print(f"  {'Total Wrong':<30s} {sum_reg['wrong']:>10d} {sum_aug['wrong']:>10d} {delta_wrong:>+10d}")
    print(f"  {'Total Hallucination':<30s} {sum_reg['total_hallucination']:>10d} {sum_aug['total_hallucination']:>10d} {delta_hall:>+10d}")
    print()

    # Per-category comparison
    all_cats = sorted(
        set(sum_reg["error_distribution"]) | set(sum_aug["error_distribution"])
    )
    if all_cats:
        print(f"  {'Error Category':<30s} {'Regular':>10s} {'Augmented':>10s} {'Delta':>10s}")
        print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
        for cat in all_cats:
            r = sum_reg["error_distribution"].get(cat, 0)
            a = sum_aug["error_distribution"].get(cat, 0)
            print(f"  {cat:<30s} {r:>10d} {a:>10d} {a-r:>+10d}")
        print()

    # ── Novel-view selection statistics ──
    novel_stats = _compute_novel_stats(novel_tracker) if novel_tracker else {}
    if novel_stats:
        _print_divider("═")
        print("  NOVEL VIEW SELECTION BY VIEW SELECTOR")
        _print_divider("─")
        print(f"  Total questions evaluated     : {novel_stats['total_questions']}")
        print(f"  Questions with ≥1 novel view  : {novel_stats['questions_with_novel_selected']} "
              f"({novel_stats['pct_questions_with_novel']:.1f}%)")
        print(f"  Total novel views selected    : {novel_stats['total_novel_selected']} "
              f"(out of {novel_stats['total_novel_available']} available)")
        print(f"  Avg novel views per question  : {novel_stats['avg_novel_per_question']:.2f}")
        print(f"  Max novel views in one question: {novel_stats['max_novel_per_question']}")
        print()
        print(f"  Distribution of novel views selected per question:")
        print(f"  {'# Novel Views':>15s} {'# Questions':>15s}")
        print(f"  {'─'*15} {'─'*15}")
        for n_novel, n_questions in sorted(novel_stats["novel_count_distribution"].items()):
            print(f"  {n_novel:>15d} {n_questions:>15d}")
        _print_divider("═")
        print()

    # ── Per-question transitions ──
    reg_map = {str(p["question_id"]): p["answer_top10"][0] for p in predictions_regular}
    aug_map = {str(p["question_id"]): p["answer_top10"][0] for p in predictions_augmented}
    gt_map = {str(g["question_id"]): g for g in ground_truth}

    # Also build a tracker lookup for enriching transition data
    tracker_map = {}
    if novel_tracker:
        tracker_map = {str(t["question_id"]): t for t in novel_tracker}

    fixed, broken, both_wrong, both_right = [], [], [], []
    for qid in gt_map:
        r_ans = reg_map.get(qid, "")
        a_ans = aug_map.get(qid, "")
        gt_ans = gt_map[qid]["answers"]
        r_ok = is_answer_correct(r_ans, gt_ans)
        a_ok = is_answer_correct(a_ans, gt_ans)

        info = {
            "question_id": qid,
            "scene_id": gt_map[qid]["scene_id"],
            "question": gt_map[qid]["question"],
            "situation": gt_map[qid].get("situation", ""),
            "gt": gt_ans,
            "regular_ans": r_ans,
            "augmented_ans": a_ans,
        }
        # Enrich with novel-view info
        if qid in tracker_map:
            info["novel_selected"] = tracker_map[qid]["novel_selected"]
            info["novel_selected_names"] = tracker_map[qid]["novel_selected_names"]

        if r_ok and a_ok:
            both_right.append(info)
        elif r_ok and not a_ok:
            broken.append(info)
        elif not r_ok and a_ok:
            fixed.append(info)
        else:
            both_wrong.append(info)

    _print_divider("═")
    print(f"  Question-level transitions:")
    print(f"    Both correct      : {len(both_right)}")
    print(f"    Both wrong        : {len(both_wrong)}")
    print(f"    Fixed by augment  : {len(fixed)}")
    print(f"    Broken by augment : {len(broken)}")
    _print_divider("═")

    # Show a few examples
    if fixed:
        print("\n  ── Examples FIXED by augmentation ──")
        for ex in fixed[:5]:
            n_novel = ex.get("novel_selected", "?")
            print(f"    [{ex['scene_id']}] Q: {ex['question'][:70]}")
            print(f"       GT: {ex['gt']}  |  Reg: '{ex['regular_ans']}'  →  Aug: '{ex['augmented_ans']}'")
            print(f"       Novel views in final 9: {n_novel}")

    if broken:
        print("\n  ── Examples BROKEN by augmentation ──")
        for ex in broken[:5]:
            n_novel = ex.get("novel_selected", "?")
            print(f"    [{ex['scene_id']}] Q: {ex['question'][:70]}")
            print(f"       GT: {ex['gt']}  |  Reg: '{ex['regular_ans']}'  →  Aug: '{ex['augmented_ans']}'")
            print(f"       Novel views in final 9: {n_novel}")

    # ── Save full report ────────────────────────────────────────────────
    report = {
        "summary": {
            "regular": sum_reg,
            "augmented": sum_aug,
            "delta": {
                "accuracy": round(delta_acc, 2),
                "wrong": delta_wrong,
                "hallucination": delta_hall,
            },
        },
        "novel_view_selection": novel_stats,
        "transitions": {
            "both_correct": len(both_right),
            "both_wrong": len(both_wrong),
            "fixed_by_augment": len(fixed),
            "broken_by_augment": len(broken),
        },
        "fixed_questions": fixed,
        "broken_questions": broken,
        "wrong_regular": wrong_reg,
        "wrong_augmented": wrong_aug,
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{tag}_comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Full report saved to: {report_path}")
    print()
