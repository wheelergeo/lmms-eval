import string
import os
import json
import numpy as np

from datasets import load_dataset
from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None
GQA_METRICS = ["exact_match"]


def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        GQA_RAW_IMAGE_DATASET = load_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev", token=True)
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]


def gqa_doc_to_text(doc, lmms_eval_specific_kwargs):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def _exact_match(
    predictions,
    references,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    predictions = np.asarray(predictions)
    references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return np.mean(score_list)

def gqa_process_result(doc, result):
    gold = doc["answer"]

    if not isinstance(gold, list):
        gold = [gold]

    exact_match = _exact_match(
        references=gold,
        predictions=result,
        ignore_case=True,
        ignore_punctuation=True,
    )

    data_dict = {"id": doc["id"], "image_id": doc["imageId"], "question": doc["question"], "exact_match": exact_match}

    return {f"gqa_{metric}": data_dict for metric in GQA_METRICS}


def gqa_aggregation_result(results, metric, args):
    task_name = "gqa"
    if args and args.tasks:
        for task in args.tasks.split(","):
            if "gqa" in task:
                task_name = task
                break

    screen_percent = 0
    if args and args.tasks_screen_thres:
        for k in args.tasks_screen_thres.keys():
            if "gqa" in k:
                screen_percent = args.tasks_screen_thres[k]
                break

    match metric:
        case "mean":
            total = 0
            cnt = 0
            good_case = []
            bad_case = []
            for result in results:
                total += result["exact_match"]
                cnt += 1
                if screen_percent:
                    if result["exact_match"] >= screen_percent:
                        good_case.append(result)
                    else:
                        bad_case.append(result)
            if screen_percent:
                good_path = generate_submission_file(f"{task_name}-exact_match.json", args, subpath="goodcase")
                with open(good_path, "w") as f:
                    json.dump(good_case, f, indent=4)

                bad_path = generate_submission_file(f"{task_name}-exact_match.json", args, subpath="badcase")
                with open(bad_path, "w") as f:
                    json.dump(bad_case, f, indent=4)
        case _:
            return 0

    return total / cnt if cnt else 0


def gqa_mean(results, args):
    return gqa_aggregation_result(results, "mean", args)