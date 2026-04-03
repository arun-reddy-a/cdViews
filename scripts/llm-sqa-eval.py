"""SQA3D-style evaluation: accuracy with GPT-based semantic judge.

Replicates the evaluation logic from the SQA3D codebase exactly:
  - clean_answer normalisation (typo fixes, digit-to-word, article removal)
  - Cascaded string matching: exact → substring → collapsed-substring → token-overlap
  - Optional GPT-based semantic judge for remaining mismatches
  - Compares prediction against the FIRST ground-truth answer only

Usage:
    python llm-sqa-eval.py --pred ../data/qa/SQA_test_answers_3d.json \
                           --gt ../data/qa/SQA/SQA_test.json

    # With GPT semantic judge:
    OPENAI_API_KEY="sk-..." python llm-sqa-eval.py \
        --pred ../data/qa/SQA_test_answers_3d.json \
        --gt ../data/qa/SQA/SQA_test.json \
        --use_gpt_metric --gpt_model gpt-4o-mini
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

from tqdm import tqdm

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


def clean_answer(data):
    """Normalise answer list in-place (SQA3D reference, verbatim)."""
    key = 'answer'
    for index in range(len(data)):
        data[index][key] = data[index][key].lower()
        data[index][key] = re.sub('[ ]+$' ,'', data[index][key])
        data[index][key] = re.sub('^[ ]+' ,'', data[index][key])
        data[index][key] = re.sub(' {2,}', ' ', data[index][key])

        data[index][key] = re.sub(r'\.[ ]{2,}', '. ', data[index][key])
        data[index][key] = re.sub(r'[^a-zA-Z0-9,\'\s\-:]+', '', data[index][key])
        data[index][key] = re.sub('ç' ,'c', data[index][key])
        data[index][key] = re.sub('\u2019' ,'\'', data[index][key])
        data[index][key] = re.sub(r'\bletf\b' ,'left', data[index][key])
        data[index][key] = re.sub(r'\blet\b' ,'left', data[index][key])
        data[index][key] = re.sub(r'\btehre\b' ,'there', data[index][key])
        data[index][key] = re.sub(r'\brigth\b' ,'right', data[index][key])
        data[index][key] = re.sub(r'\brght\b' ,'right', data[index][key])
        data[index][key] = re.sub(r'\bbehine\b', 'behind', data[index][key])
        data[index][key] = re.sub(r'\btv\b' ,'TV', data[index][key])
        data[index][key] = re.sub(r'\bchai\b' ,'chair', data[index][key])
        data[index][key] = re.sub(r'\bwasing\b' ,'washing', data[index][key])
        data[index][key] = re.sub(r'\bwaslked\b' ,'walked', data[index][key])
        data[index][key] = re.sub(r'\boclock\b' ,'o\'clock', data[index][key])
        data[index][key] = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data[index][key])

        data[index][key] = re.sub(r'\b0\b', 'zero', data[index][key])
        data[index][key] = re.sub(r'\bnone\b', 'zero', data[index][key])
        data[index][key] = re.sub(r'\b1\b', 'one', data[index][key])
        data[index][key] = re.sub(r'\b2\b', 'two', data[index][key])
        data[index][key] = re.sub(r'\b3\b', 'three', data[index][key])
        data[index][key] = re.sub(r'\b4\b', 'four', data[index][key])
        data[index][key] = re.sub(r'\b5\b', 'five', data[index][key])
        data[index][key] = re.sub(r'\b6\b', 'six', data[index][key])
        data[index][key] = re.sub(r'\b7\b', 'seven', data[index][key])
        data[index][key] = re.sub(r'\b8\b', 'eight', data[index][key])
        data[index][key] = re.sub(r'\b9\b', 'nine', data[index][key])
        data[index][key] = re.sub(r'\b10\b', 'ten', data[index][key])
        data[index][key] = re.sub(r'\b11\b', 'eleven', data[index][key])
        data[index][key] = re.sub(r'\b12\b', 'twelve', data[index][key])
        data[index][key] = re.sub(r'\b13\b', 'thirteen', data[index][key])
        data[index][key] = re.sub(r'\b14\b', 'fourteen', data[index][key])
        data[index][key] = re.sub(r'\b15\b', 'fifteen', data[index][key])
        data[index][key] = re.sub(r'\b16\b', 'sixteen', data[index][key])
        data[index][key] = re.sub(r'\b17\b', 'seventeen', data[index][key])
        data[index][key] = re.sub(r'\b18\b', 'eighteen', data[index][key])
        data[index][key] = re.sub(r'\b19\b', 'nineteen', data[index][key])
        data[index][key] = re.sub(r'\b20\b', 'twenty', data[index][key])
        data[index][key] = re.sub(r'\b23\b', 'twenty-three', data[index][key])

        data[index][key] = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data[index][key])
        data[index][key] = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data[index][key])
        data[index][key] = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data[index][key])
        data[index][key] = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data[index][key])

        data[index][key] = re.sub(r'\bbackwards\b', 'backward', data[index][key])
    return data


def chat_llm(history, temperature=0, max_tokens=100, model='gpt-3.5-turbo'):
    """Call OpenAI chat API (matches reference SQA3D interface)."""
    if type(history) == str:
        history = [('user', history)]

    chat_history = []
    for i in history:
        if i[0] == 'user':
            chat_history.append({'role': 'user', 'content': i[1]})
        elif i[0] == 'assistant':
            chat_history.append({'role': 'assistant', 'content': i[1]})
        else:
            raise NotImplementedError

    total_trials = 0
    while True:
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=chat_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            time.sleep(1)
            break
        except openai.OpenAIError as e:
            total_trials += 1
            print(e)
            time.sleep(1)
            if total_trials > 10:
                return ""
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
            sys.exit(1)

    return response.choices[0].message.content


def gpt_llm_eval(q, pred, a, model='gpt-4o-mini'):
    """GPT-based semantic judge (exact SQA3D reference prompt)."""
    eval_prompt = f"""
    Given the question "{q}", given the true answer "{a}", does the prediction "{pred}" imply the true answer? Answer with Yes or No.
    """
    output = chat_llm(eval_prompt, model=model)
    return output.lower().strip() == 'yes'


def load_json(path):
    with open(path, 'r') as f:
        content = f.read().strip()
    if content.startswith('[') or content.startswith('{'):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def extract_pred_answer(entry):
    """Extract a single predicted answer string from a prediction entry."""
    if 'answer_top10' in entry:
        answers = [a.strip().rstrip('.,;!?') for a in entry['answer_top10']]
        return Counter(answers).most_common(1)[0][0]
    if 'answer' in entry:
        return entry['answer']
    if 'text' in entry:
        return entry['text']
    raise KeyError(f"No answer field in pred entry: {list(entry.keys())}")


def extract_gt_answer(entry):
    """Extract the FIRST ground-truth answer (matches SQA3D reference)."""
    if 'answers' in entry:
        ans = entry['answers']
        if isinstance(ans, list) and len(ans) > 0:
            if isinstance(ans[0], dict):
                return ans[0]['answer']
            return ans[0]
    if 'answer' in entry:
        return entry['answer']
    if 'text' in entry:
        return entry['text']
    raise KeyError(f"No answer field in gt entry: {list(entry.keys())}")


def extract_question(entry):
    """Build question string (situation + question for SQA)."""
    parts = []
    if 'situation' in entry:
        parts.append(entry['situation'])
    if 'question' in entry:
        parts.append(entry['question'])
    return ' '.join(parts) if parts else ''


def compute_accuracy(prediction, gt_lookup, use_gpt_metric=False, gpt_model='gpt-4o-mini'):
    """SQA3D-style accuracy: exact match with cascaded relaxation + optional GPT judge.

    Matching cascade (same as reference compute_accuracy):
      1. Exact string match
      2. Substring containment (pred in gt OR gt in pred)
      3. Whitespace-collapsed substring
      4. Token overlap (any shared word)
      5. [optional] GPT semantic judge
    """
    prediction = clean_answer(prediction)

    corr = {}
    for ind, i in enumerate(tqdm(prediction, desc="Evaluating")):
        qid = str(i['question_id'])
        if qid not in corr:
            corr[qid] = 0

        if qid not in gt_lookup:
            continue

        gt_entry = gt_lookup[qid]
        answer = gt_entry['answer']

        if use_gpt_metric:
            if i['answer'] == answer:
                corr[qid] += 1
            elif gpt_llm_eval(
                extract_question(gt_entry), i['answer'], answer, model=gpt_model
            ):
                print(i['answer'], '==', answer)
                corr[qid] += 1
            else:
                print(i['answer'], '!=', answer)
        else:
            if i['answer'] == answer:
                corr[qid] += 1
            elif i['answer'] in answer:
                corr[qid] += 1
            elif ''.join(i['answer'].split()) in ''.join(answer.split()):
                corr[qid] += 1
            elif len(set(i['answer'].split()).intersection(answer.split())) > 0:
                corr[qid] += 1
            else:
                continue

    total = len(corr)
    cnt = sum([1 if v >= 1 else 0 for v in corr.values()])

    print('Acc: {}/{} = {:.4f}'.format(cnt, total, cnt / total))
    return {'accuracy': cnt / total, 'correct': cnt, 'total': total}


def main():
    parser = argparse.ArgumentParser(
        description="SQA3D-style evaluation with optional GPT semantic judge"
    )
    parser.add_argument('--pred', required=True, help="Path to prediction JSON")
    parser.add_argument('--gt', required=True, help="Path to ground-truth JSON")
    parser.add_argument('--use_gpt_metric', action='store_true',
                        help="Use GPT-based semantic judge for non-matching predictions")
    parser.add_argument('--gpt_model', type=str, default='gpt-4o-mini',
                        help="OpenAI model for GPT judge (default: gpt-4o-mini)")
    args = parser.parse_args()

    if args.use_gpt_metric:
        if not HAS_OPENAI:
            print("ERROR: openai package not installed. Run: pip install openai")
            sys.exit(1)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: Set OPENAI_API_KEY environment variable.")
            sys.exit(1)
        openai.api_key = api_key

    preds_raw = load_json(args.pred)
    gts_raw = load_json(args.gt)

    if isinstance(preds_raw, dict):
        preds_raw = [{'question_id': k, **(v if isinstance(v, dict) else {'text': v})}
                     for k, v in preds_raw.items()]
    if isinstance(gts_raw, dict):
        gts_raw = [{'question_id': k, **(v if isinstance(v, dict) else {'text': v})}
                   for k, v in gts_raw.items()]

    prediction = []
    for entry in preds_raw:
        prediction.append({
            'question_id': str(entry.get('question_id', '')),
            'question': entry.get('question', ''),
            'situation': entry.get('situation', ''),
            'answer': extract_pred_answer(entry),
        })

    gt_lookup = {}
    for entry in gts_raw:
        qid = str(entry.get('question_id', ''))
        gt_lookup[qid] = {
            'question_id': qid,
            'question': entry.get('question', ''),
            'situation': entry.get('situation', ''),
            'answer': extract_gt_answer(entry),
        }

    gt_as_list = list(gt_lookup.values())
    gt_as_list = clean_answer(gt_as_list)
    gt_lookup = {e['question_id']: e for e in gt_as_list}

    print(f"Predictions: {len(prediction)}")
    print(f"Ground truth: {len(gt_lookup)}")

    compute_accuracy(prediction, gt_lookup, args.use_gpt_metric, args.gpt_model)


if __name__ == '__main__':
    main()
