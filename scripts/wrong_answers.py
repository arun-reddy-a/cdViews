"""
Hallucination Analysis Script for cdViews
==========================================
Analyzes wrong answers produced by cdViews on ScanQA and SQA datasets.

Usage:
    python wrong_answers.py --dataset ScanQA --split val
    python wrong_answers.py --dataset SQA --split test
    python wrong_answers.py --dataset ScanQA --split val --visualize --num_samples 10
"""

import json
import os
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import random


def load_data(dataset: str, split: str, data_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Load predicted answers and ground truth data."""
    
    # Predicted answers file (inside dataset folder)
    if dataset == "ScanQA":
        pred_file = os.path.join(data_dir, "ScanQA", f"ScanQA_{split}_answers.json")
        gt_file = os.path.join(data_dir, "ScanQA", f"ScanQA_v1.0_{split}.json")
    else:  # SQA
        pred_file = os.path.join(data_dir, "SQA", f"SQA_{split}_answers.json")
        gt_file = os.path.join(data_dir, "SQA", f"SQA_{split}.json")
    
    print(f"Loading predictions from: {pred_file}")
    print(f"Loading ground truth from: {gt_file}")
    
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Predictions file not found: {pred_file}")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Check if ground truth has answers
    if len(ground_truth) > 0 and 'answers' not in ground_truth[0]:
        raise ValueError(
            f"Ground truth file does not contain 'answers' field.\n"
            f"This is a blind test set meant for benchmark submission.\n"
            f"Available splits with answers:\n"
            f"  - ScanQA: val\n"
            f"  - SQA: test, val\n"
            f"Try: python wrong_answers.py --dataset {dataset} --split val"
        )
    
    return predictions, ground_truth


def is_answer_correct(pred_answer: str, gt_answers: List[str], strict: bool = False) -> bool:
    """Check if predicted answer matches any ground truth answer."""
    pred = pred_answer.lower().strip()
    gt_lower = [a.lower().strip() for a in gt_answers]
    
    if strict:
        # Exact match only
        return pred in gt_lower
    else:
        # Flexible matching: exact, substring, or contained
        for gt_a in gt_lower:
            if pred == gt_a:
                return True
            if pred in gt_a or gt_a in pred:
                return True
            # Check word overlap for longer answers
            pred_words = set(pred.split())
            gt_words = set(gt_a.split())
            if len(pred_words) > 1 and len(gt_words) > 1:
                overlap = len(pred_words & gt_words) / max(len(pred_words), len(gt_words))
                if overlap > 0.5:
                    return True
        return False


def categorize_error(pred: str, gt_answers: List[str], question: str) -> str:
    """Categorize the type of error/hallucination."""
    pred = pred.lower()
    gt_lower = [g.lower() for g in gt_answers]
    question = question.lower()
    
    # Color-related error
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 
              'gray', 'grey', 'orange', 'purple', 'pink', 'beige', 'tan']
    pred_colors = [c for c in colors if c in pred]
    gt_colors = [c for c in colors if any(c in g for g in gt_lower)]
    if pred_colors and gt_colors and set(pred_colors) != set(gt_colors):
        return 'wrong_color'
    if 'color' in question and pred_colors and not gt_colors:
        return 'hallucinated_color'
    
    # Count/number-related error
    numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 
               'eight', 'nine', 'ten', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    pred_nums = [n for n in numbers if n in pred.split()]
    gt_nums = [n for n in numbers if any(n in g.split() for g in gt_lower)]
    if pred_nums and gt_nums and set(pred_nums) != set(gt_nums):
        return 'wrong_count'
    if ('how many' in question or 'number' in question) and pred_nums:
        return 'wrong_count'
    
    # Spatial relation error
    spatial = ['left', 'right', 'front', 'behind', 'above', 'below', 'near', 
               'next to', 'between', 'under', 'over', 'beside', 'corner']
    if any(s in question for s in spatial):
        return 'wrong_spatial'
    
    # Object-related error
    common_objects = ['chair', 'table', 'door', 'window', 'bed', 'desk', 'sofa', 
                      'couch', 'lamp', 'shelf', 'cabinet', 'tv', 'monitor', 'computer',
                      'refrigerator', 'sink', 'toilet', 'shower', 'bathtub', 'mirror']
    pred_objects = [o for o in common_objects if o in pred]
    gt_objects = [o for o in common_objects if any(o in g for g in gt_lower)]
    if pred_objects and gt_objects and set(pred_objects) != set(gt_objects):
        return 'wrong_object'
    if pred_objects and not gt_objects:
        return 'hallucinated_object'
    
    # Yes/No error
    if pred in ['yes', 'no'] or any(g in ['yes', 'no'] for g in gt_lower):
        return 'wrong_yes_no'
    
    # Check for completely unrelated answer (no word overlap)
    pred_words = set(pred.split())
    gt_words = set(' '.join(gt_lower).split())
    meaningful_overlap = pred_words & gt_words - {'the', 'a', 'an', 'is', 'are', 'it', 'in', 'on', 'of'}
    if len(meaningful_overlap) == 0 and len(pred_words) > 0:
        return 'unrelated_hallucination'
    
    return 'other'


def compare_answers(predictions: List[Dict], ground_truth: List[Dict], 
                    dataset: str, strict: bool = False) -> Tuple[List[Dict], List[Dict]]:
    """Compare predictions with ground truth and categorize errors."""
    
    # Create lookup dictionaries
    pred_dict = {str(p['question_id']): p for p in predictions}
    gt_dict = {str(g['question_id']): g for g in ground_truth}
    
    correct = []
    wrong = []
    
    for qid, gt in gt_dict.items():
        if qid not in pred_dict:
            print(f"Warning: Question {qid} not found in predictions")
            continue
        
        pred = pred_dict[qid]
        pred_answer = pred['answer_top10'][0].lower().strip()
        gt_answers = gt['answers']
        
        # Build entry with all relevant info
        entry = {
            'question_id': qid,
            'scene_id': gt['scene_id'],
            'question': gt['question'],
            'predicted': pred_answer,
            'ground_truth': gt_answers,
            'object_names': gt.get('object_names', []),
            'object_ids': gt.get('object_ids', []),
        }
        
        # For SQA, include situation
        if dataset == 'SQA' and 'situation' in gt:
            entry['situation'] = gt['situation']
        
        if is_answer_correct(pred_answer, gt_answers, strict=strict):
            correct.append(entry)
        else:
            entry['error_type'] = categorize_error(pred_answer, gt_answers, gt['question'])
            wrong.append(entry)
    
    return correct, wrong


def analyze_errors(wrong: List[Dict], correct: List[Dict]) -> Dict:
    """Perform detailed analysis on errors."""
    
    analysis = {
        'total_wrong': len(wrong),
        'total_correct': len(correct),
        'accuracy': 100 * len(correct) / (len(correct) + len(wrong)) if (len(correct) + len(wrong)) > 0 else 0,
    }
    
    # 1. Error type distribution
    error_types = defaultdict(list)
    for entry in wrong:
        error_types[entry['error_type']].append(entry)
    
    analysis['error_distribution'] = {k: len(v) for k, v in error_types.items()}
    analysis['error_distribution_pct'] = {
        k: 100 * len(v) / len(wrong) for k, v in error_types.items()
    } if len(wrong) > 0 else {}
    
    # 2. Question type analysis (by first word)
    question_types = defaultdict(lambda: {'correct': 0, 'wrong': 0})
    for entry in correct:
        q_start = entry['question'].split()[0].lower() if entry['question'] else 'unknown'
        question_types[q_start]['correct'] += 1
    for entry in wrong:
        q_start = entry['question'].split()[0].lower() if entry['question'] else 'unknown'
        question_types[q_start]['wrong'] += 1
    
    analysis['question_type_accuracy'] = {}
    for qtype, counts in question_types.items():
        total = counts['correct'] + counts['wrong']
        analysis['question_type_accuracy'][qtype] = {
            'accuracy': 100 * counts['correct'] / total if total > 0 else 0,
            'correct': counts['correct'],
            'wrong': counts['wrong'],
            'total': total
        }
    
    # 3. Object-related errors
    object_errors = defaultdict(int)
    for entry in wrong:
        for obj in entry.get('object_names', []):
            object_errors[obj] += 1
    analysis['objects_with_most_errors'] = dict(sorted(object_errors.items(), key=lambda x: -x[1])[:20])
    
    # 4. Hallucinated words (words in prediction but not in ground truth)
    hallucinated_words = []
    for entry in wrong:
        pred_words = set(entry['predicted'].lower().split())
        gt_words = set(' '.join(entry['ground_truth']).lower().split())
        hallucinated = pred_words - gt_words - {'the', 'a', 'an', 'is', 'are', 'it', 'in', 'on', 'of', 'to'}
        hallucinated_words.extend(list(hallucinated))
    
    word_counts = Counter(hallucinated_words)
    analysis['most_hallucinated_words'] = dict(word_counts.most_common(30))
    
    # 5. Scene-level error analysis
    scene_errors = defaultdict(int)
    scene_totals = defaultdict(int)
    for entry in wrong:
        scene_errors[entry['scene_id']] += 1
        scene_totals[entry['scene_id']] += 1
    for entry in correct:
        scene_totals[entry['scene_id']] += 1
    
    scene_error_rates = {
        scene: 100 * scene_errors[scene] / scene_totals[scene] 
        for scene in scene_totals
    }
    worst_scenes = sorted(scene_error_rates.items(), key=lambda x: -x[1])[:10]
    analysis['worst_scenes'] = {scene: {'error_rate': rate, 'num_errors': scene_errors[scene]} 
                                for scene, rate in worst_scenes}
    
    return analysis, error_types


def print_analysis(analysis: Dict, error_types: Dict):
    """Print analysis results in a formatted way."""
    
    print("\n" + "="*60)
    print("HALLUCINATION ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nOverall Accuracy: {analysis['accuracy']:.2f}%")
    print(f"Total Correct: {analysis['total_correct']}")
    print(f"Total Wrong: {analysis['total_wrong']}")
    
    print("\n" + "-"*40)
    print("ERROR TYPE DISTRIBUTION")
    print("-"*40)
    for error_type, count in sorted(analysis['error_distribution'].items(), key=lambda x: -x[1]):
        pct = analysis['error_distribution_pct'].get(error_type, 0)
        print(f"  {error_type:25s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n" + "-"*40)
    print("ACCURACY BY QUESTION TYPE")
    print("-"*40)
    sorted_qtypes = sorted(analysis['question_type_accuracy'].items(), 
                          key=lambda x: x[1]['total'], reverse=True)
    for qtype, data in sorted_qtypes[:10]:
        print(f"  {qtype:15s}: {data['accuracy']:5.1f}% ({data['correct']}/{data['total']})")
    
    print("\n" + "-"*40)
    print("OBJECTS WITH MOST ERRORS")
    print("-"*40)
    for obj, count in list(analysis['objects_with_most_errors'].items())[:10]:
        print(f"  {obj:20s}: {count}")
    
    print("\n" + "-"*40)
    print("MOST HALLUCINATED WORDS")
    print("-"*40)
    for word, count in list(analysis['most_hallucinated_words'].items())[:15]:
        if len(word) > 2:
            print(f"  {word:20s}: {count}")
    
    print("\n" + "-"*40)
    print("SCENES WITH HIGHEST ERROR RATES")
    print("-"*40)
    for scene, data in analysis['worst_scenes'].items():
        print(f"  {scene}: {data['error_rate']:.1f}% ({data['num_errors']} errors)")


def print_sample_errors(error_types: Dict, num_samples: int = 3):
    """Print sample errors from each category."""
    
    print("\n" + "="*60)
    print("SAMPLE ERRORS BY CATEGORY")
    print("="*60)
    
    for error_type, entries in error_types.items():
        print(f"\n--- {error_type.upper()} ({len(entries)} total) ---")
        samples = random.sample(entries, min(num_samples, len(entries)))
        for i, entry in enumerate(samples, 1):
            print(f"\n  [{i}] Scene: {entry['scene_id']}")
            if 'situation' in entry:
                print(f"      Situation: {entry['situation']}")
            print(f"      Question: {entry['question']}")
            print(f"      Predicted: '{entry['predicted']}'")
            print(f"      Ground Truth: {entry['ground_truth']}")


def visualize_errors(wrong: List[Dict], data_dir: str, ranked_views_dir: str,
                     dataset: str, split: str, num_samples: int = 3):
    """Visualize error cases with their images - samples from EACH error category."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("matplotlib or PIL not available. Skipping visualization.")
        return
    
    image_folder = os.path.join(data_dir, 'scannetv2', 'frames_square')
    
    # Check both possible locations for ranked views
    ranked_views_file = os.path.join(data_dir, dataset, f"{dataset}_{split}_view_ranking.json")
    if not os.path.exists(ranked_views_file):
        ranked_views_file = os.path.join(data_dir, f"{dataset}_{split}_view_ranking.json")
    
    # Load ranked views if available
    ranked_views = {}
    if os.path.exists(ranked_views_file):
        with open(ranked_views_file, 'r') as f:
            ranked_views = json.load(f)
        print(f"Loaded ranked views from {ranked_views_file}")
    
    # Group errors by category
    error_categories = defaultdict(list)
    for entry in wrong:
        error_categories[entry.get('error_type', 'unknown')].append(entry)
    
    print(f"\nGenerating visualizations for {len(error_categories)} error categories...")
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(data_dir), 'analysis_outputs', f"{dataset}_{split}_visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    total_saved = 0
    
    # Sample from EACH error category
    for category, entries in error_categories.items():
        # Create subdirectory for each category
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Sample entries from this category
        samples = random.sample(entries, min(num_samples, len(entries)))
        
        print(f"  {category}: saving {len(samples)} samples...")
        
        for idx, entry in enumerate(samples):
            scene_id = entry['scene_id']
            color_folder = os.path.join(image_folder, scene_id, 'color')
            
            if not os.path.exists(color_folder):
                print(f"    Warning: Color folder not found for {scene_id}")
                continue
            
            # Get images
            if str(entry['question_id']) in ranked_views:
                image_files = ranked_views[str(entry['question_id'])][:9]
            else:
                all_images = os.listdir(color_folder)
                image_files = sorted(all_images)[:9]
            
            # Create figure
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            
            title = (
                f"Scene: {scene_id} | Error Type: {category}\n"
                f"Q: {entry['question'][:80]}{'...' if len(entry['question']) > 80 else ''}\n"
                f"Predicted: '{entry['predicted']}' | GT: {entry['ground_truth']}"
            )
            if 'situation' in entry:
                title = f"Situation: {entry['situation'][:60]}...\n" + title
            fig.suptitle(title, fontsize=10, wrap=True)
            
            for i, ax in enumerate(axes.flat):
                if i < len(image_files):
                    img_path = os.path.join(color_folder, image_files[i])
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        ax.imshow(img)
                        ax.set_title(image_files[i], fontsize=8)
                ax.axis('off')
            
            plt.tight_layout()
            
            # Save figure with category in filename
            fig_path = os.path.join(category_dir, f"{idx+1}_{entry['question_id']}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
            total_saved += 1
    
    print(f"\nSaved {total_saved} visualizations to: {output_dir}")


def save_results(analysis: Dict, wrong: List[Dict], correct: List[Dict], 
                 output_path: str):
    """Save analysis results to JSON file."""
    
    output = {
        'summary': {
            'total': analysis['total_correct'] + analysis['total_wrong'],
            'correct': analysis['total_correct'],
            'wrong': analysis['total_wrong'],
            'accuracy': analysis['accuracy']
        },
        'error_distribution': analysis['error_distribution'],
        'error_distribution_pct': analysis['error_distribution_pct'],
        'question_type_accuracy': analysis['question_type_accuracy'],
        'objects_with_most_errors': analysis['objects_with_most_errors'],
        'most_hallucinated_words': analysis['most_hallucinated_words'],
        'worst_scenes': analysis['worst_scenes'],
        'wrong_answers': wrong,
        'correct_answers': correct
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nFull results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze hallucinations in cdViews predictions')
    parser.add_argument('--dataset', type=str, default='ScanQA', choices=['ScanQA', 'SQA'],
                        help='Dataset to analyze (default: ScanQA)')
    parser.add_argument('--split', type=str, default='val',
                        help='Data split to analyze (default: val)')
    parser.add_argument('--data_dir', type=str, default='../data/qa',
                        help='Path to data directory (default: ../data/qa)')
    parser.add_argument('--strict', action='store_true',
                        help='Use strict matching (exact match only)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of errors')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize (default: 5)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Load data
    print(f"\nAnalyzing {args.dataset} - {args.split} split")
    print("="*50)
    
    predictions, ground_truth = load_data(args.dataset, args.split, args.data_dir)
    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth samples")
    
    # Compare answers
    correct, wrong = compare_answers(predictions, ground_truth, args.dataset, strict=args.strict)
    
    # Analyze errors
    analysis, error_types = analyze_errors(wrong, correct)
    
    # Print results
    print_analysis(analysis, error_types)
    print_sample_errors(error_types, num_samples=3)
    
    # Visualize if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_errors(wrong, args.data_dir, args.data_dir, 
                        args.dataset, args.split, args.num_samples)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(os.path.dirname(args.data_dir), 'analysis_outputs')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{args.dataset}_{args.split}_hallucination_analysis.json")
    
    save_results(analysis, wrong, correct, output_path)


if __name__ == "__main__":
    main()
