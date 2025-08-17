# Add the following functions to your existing utils.py file
import json

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


POPE_METRICS = ["accuracy", "precision", "recall", "f1_score", "yes_ratio"]


def pope_doc_to_visual(doc):
    # Assuming the 'doc' dictionary has a key 'image' with image data
    return [doc["image"].convert("RGB")]


def pope_doc_to_text(doc, lmms_eval_specific_kwargs):
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    # Assuming the 'doc' dictionary has a key 'question' with the question text
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def pope_process_results(doc, results):
    pred = results[0].lower().strip()
    gt_ans = doc["answer"].lower().strip()
    assert gt_ans in ["yes", "no"]
    score = 1.0 if pred == gt_ans else 0.0

    data_dict = {
        "question_id": doc["question_id"],
        "question": doc["question"],
        "category": doc["category"],
        "answer": gt_ans,
        "exact_match": score,
        "prediction": pred
    }

    return {f"pope_{metric}": data_dict for metric in POPE_METRICS}

def pope_aggregate_accuracy(results, args):
    task_name = "pope"
    if args and args.tasks:
        for task in args.tasks.split(","):
            if "pope" in task:
                task_name = task
                break

    screen_percent = 0
    if args and args.tasks_screen_thres:
        for k in args.tasks_screen_thres.keys():
            if "pope" in k:
                screen_percent = args.tasks_screen_thres[k]
                break

    total_score = 0
    good_case = []
    bad_case = []
    for result in results:
        total_score += result["exact_match"]

        if screen_percent:
            if result["exact_match"] >= screen_percent:
                good_case.append(result)
            else:
                bad_case.append(result)

    # screen good case
    if screen_percent:
        good_path = generate_submission_file(f"{task_name}-exact_match.json", args, subpath="goodcase")
        with open(good_path, "w") as f:
            json.dump(good_case, f, indent=4)

        bad_path = generate_submission_file(f"{task_name}-exact_match.json", args, subpath="badcase")
        with open(bad_path, "w") as f:
            json.dump(bad_case, f, indent=4)

    avg_score = total_score / len(results)
    return avg_score


def pope_aggregate_precision(results):
    true_positives = 0
    false_positives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["answer"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "no" and pred == "yes":
            false_positives += 1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    return precision


def pope_aggregate_recall(results):
    true_positives = 0
    false_negatives = 0
    for result in results:
        pred = result["prediction"]
        gt = result["answer"]
        if gt == "yes" and pred == "yes":
            true_positives += 1
        elif gt == "yes" and pred == "no":
            false_negatives += 1
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return recall


def pope_aggregate_f1_score(results):
    precision = pope_aggregate_precision(results)
    recall = pope_aggregate_recall(results)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def pope_aggregate_yes_ratio(results):
    yes_count = 0
    no_count = 0
    for result in results:
        gt = result["answer"]
        if gt == "yes":
            yes_count += 1
        elif gt == "no":
            no_count += 1
    yes_ratio = yes_count / (yes_count + no_count) if (yes_count + no_count) > 0 else 0
    return yes_ratio
