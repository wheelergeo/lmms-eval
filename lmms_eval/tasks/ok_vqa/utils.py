import datetime
import json
import os
import pathlib
import re
import statistics

import yaml
from loguru import logger as eval_logger

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


OK_VQA_METRICS = ["exact_match"]


def ok_vqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def ok_vqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        if gtAcc:
            accuracy = statistics.mean(gtAcc)
        else:
            accuracy = 0

    data_dict = {
        "id": doc["question_id"], 
        "question_type": doc["question_type"], 
        "answers": doc["answers"],
        "pred": resAns,
        "exact_match": accuracy,
    }

    return {
        **{f"ok_vqa_{metric}": data_dict for metric in OK_VQA_METRICS},
        "submission": {
            "image": f"{doc['question_id']}.jpg",
            "answer": resAns,
        },
    }


def ok_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    return f"{pre_prompt}{question}{post_prompt}"


def ok_vqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m%d-%H%M-%S")
    file = f"ok_vqa-test-submission-{now_date_time}.json"
    path = generate_submission_file(file, args)
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Submission file saved to {path}")


def ok_vqa_aggregation_result(results, metric, args):
    task_name = "ok_vqa"
    if args and args.tasks:
        for task in args.tasks.split(","):
            if "ok_vqa" in task:
                task_name = task
                break

    screen_percent = 0
    if args and args.tasks_screen_thres:
        for k in args.tasks_screen_thres.keys():
            if "ok_vqa" in k:
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


def ok_vqa_mean(results, args):
    return ok_vqa_aggregation_result(results, "mean", args)
