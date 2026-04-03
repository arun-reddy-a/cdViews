import re
import json
import argparse
from collections import Counter
from tqdm import tqdm


def clean_answer(data):
    """Clean and normalize answer string (from LEO: embodied-generalist)."""
    data = data.lower()
    data = re.sub('[ ]+$', '', data)
    data = re.sub('^[ ]+', '', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub(r'\.[ ]{2,}', '. ', data)
    data = re.sub(r'[^a-zA-Z0-9,\'\s\-:]+', '', data)
    data = re.sub('ç', 'c', data)
    data = re.sub('\u2019', '\'', data)
    data = re.sub(r'\bletf\b', 'left', data)
    data = re.sub(r'\blet\b', 'left', data)
    data = re.sub(r'\btehre\b', 'there', data)
    data = re.sub(r'\brigth\b', 'right', data)
    data = re.sub(r'\brght\b', 'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\btv\b', 'TV', data)
    data = re.sub(r'\bchai\b', 'chair', data)
    data = re.sub(r'\bwasing\b', 'washing', data)
    data = re.sub(r'\bwaslked\b', 'walked', data)
    data = re.sub(r'\boclock\b', 'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b', 'o\'clock', data)

    digit_map = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
        '18': 'eighteen', '19': 'nineteen', '20': 'twenty',
        '23': 'twenty-three',
    }
    for digit, word in digit_map.items():
        data = re.sub(rf'\b{digit}\b', word, data)

    data = re.sub(r'\b([a-zA-Z]+)([0-9])\b', r'\g<1>', data)
    data = re.sub(r'\ba\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\ban\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\bthe\b ([a-zA-Z]+)', r'\g<1>', data)
    data = re.sub(r'\bbackwards\b', 'backward', data)

    return data


def answer_match(pred, gts):
    """Return (EM, refined_EM). Refined EM allows substring containment."""
    if pred in gts:
        return 1, 1
    for gt in gts:
        if ''.join(pred.split()) in ''.join(gt.split()) or \
           ''.join(gt.split()) in ''.join(pred.split()):
            return 0, 1
    return 0, 0


def extract_answer(entry):
    """Extract a single answer string from a data entry.

    Supports formats:
      - {"answer_top10": [...]}  -> majority vote, strip trailing punctuation
      - {"text": "..."}         -> direct string
      - {"answers": [...]}      -> first answer
    """
    if 'answer_top10' in entry:
        answers = [a.strip().rstrip('.,;!?') for a in entry['answer_top10']]
        return Counter(answers).most_common(1)[0][0]
    if 'text' in entry:
        return entry['text']
    if 'answers' in entry:
        return entry['answers'][0]
    raise KeyError(f"No recognized answer field in entry: {list(entry.keys())}")


def extract_gt_answers(entry):
    """Extract the full set of unique GT answers from a data entry."""
    if 'answer_top10' in entry:
        return list({a.strip().rstrip('.,;!?') for a in entry['answer_top10']})
    if 'text' in entry:
        return [entry['text']]
    if 'answers' in entry:
        return list(set(entry['answers']))
    raise KeyError(f"No recognized answer field in entry: {list(entry.keys())}")


def get_question_id(entry):
    """Get question_id as a string regardless of stored type."""
    qid = entry.get('question_id')
    if qid is None:
        raise KeyError(f"No 'question_id' field in entry: {list(entry.keys())}")
    return str(qid)


def load_json(path):
    """Load a JSON file — supports both a single JSON array/object and JSONL."""
    with open(path, 'r') as f:
        content = f.read().strip()
    if content.startswith('[') or content.startswith('{'):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def calc_em_scores(pred_path, gt_path):
    preds_raw = load_json(pred_path)
    gts_raw = load_json(gt_path)

    if isinstance(preds_raw, dict):
        preds_raw = [{'question_id': k, **v} if isinstance(v, dict)
                      else {'question_id': k, 'text': v}
                      for k, v in preds_raw.items()]
    if isinstance(gts_raw, dict):
        gts_raw = [{'question_id': k, **v} if isinstance(v, dict)
                    else {'question_id': k, 'text': v}
                    for k, v in gts_raw.items()]

    gt_lookup = {get_question_id(g): g for g in gts_raw}

    em_total = 0
    em_refined_total = 0
    matched = 0
    missing = 0

    for pred_entry in tqdm(preds_raw, desc="Evaluating"):
        qid = get_question_id(pred_entry)
        if qid not in gt_lookup:
            missing += 1
            continue

        gt_entry = gt_lookup[qid]
        pred_answer = clean_answer(extract_answer(pred_entry))
        gt_answers = [clean_answer(a) for a in extract_gt_answers(gt_entry)]

        em, em_refined = answer_match(pred_answer, gt_answers)
        em_total += em
        em_refined_total += em_refined
        matched += 1

    if matched == 0:
        print("ERROR: No matching question_ids found between pred and gt.")
        return {}

    scores = {
        'EM': em_total / matched,
        'EM_refined': em_refined_total / matched,
        'matched': matched,
        'total_pred': len(preds_raw),
        'total_gt': len(gts_raw),
        'missing_gt': missing,
    }
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compute Exact Match metrics (SQA3D-style)")
    parser.add_argument('--pred', required=True, help="Path to prediction JSON (or JSONL)")
    parser.add_argument('--gt', required=True, help="Path to ground-truth JSON (or JSONL)")
    args = parser.parse_args()

    scores = calc_em_scores(args.pred, args.gt)

    print("\n===== Exact Match Results =====")
    for k, v in scores.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f} ({v * 100:.2f}%)")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
