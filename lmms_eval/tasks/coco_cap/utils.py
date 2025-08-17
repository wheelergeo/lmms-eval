import json
import os
import copy
from io import BytesIO

import requests
from loguru import logger as eval_logger
from PIL import Image
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

COCO_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]
COCO_METRICS_SCORE_LIMIT = {"Bleu_4": 1.0, "Bleu_3": 1.0, "Bleu_2": 1.0, "Bleu_1": 1.0, "METEOR": 1.0, "ROUGE_L": 1.0, "CIDEr": 8.0}


def coco_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def coco_doc_to_visual_karpathy(doc):
    image_url = doc["url"]
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    return [image.convert("RGB")]


def coco_doc_to_text(doc):
    return f"Provide a one-sentence caption for the provided image."


def coco_process_result_karpathy(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    question_id = doc["filename"]
    # The question id in our dataset is the image file itself
    image_id = int(question_id.split("_")[-1].split(".")[0])
    id = doc["imgid"]

    data_dict = {"answer": doc["sentences"], "pred": pred, "image_id": image_id, "id": id}

    return {f"coco_{metric}": data_dict for metric in COCO_METRICS}


def coco_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    pred = result[0] if len(result) > 0 else ""
    question_id = doc["question_id"]
    # The question id in our dataset is the image file itself
    image_id = int(question_id.split("_")[-1].split(".")[0])
    
    data_dict = {"id": doc["id"], "question": doc["question"], "answer": doc["answer"], "pred": pred, "image_id": image_id}

    return {f"coco_{metric}": data_dict for metric in COCO_METRICS}


def coco_aggregation_result(results, metric, args):
    scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]  # , (Spice(), "SPICE")]
    scorers_dict = {s[1]: s for s in scorers}

    task_name = "coco_captions"
    if args and args.tasks:
        for task in args.tasks.split(","):
            if "coco" in task:
                task_name = task
                break

    screen_percent = 0
    if args and args.tasks_screen_thres:
        for k in args.tasks_screen_thres.keys():
            if "coco" in k:
                screen_percent = args.tasks_screen_thres[k]
                break
        
    stored_results = []
    screen_results = copy.deepcopy(results)
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    for result in results:
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})
        for a in result["answer"]:
            dataset["annotations"].append({"image_id": int(result["image_id"]), "caption": a, "id": idx})
            idx += 1
        dataset["images"].append({"id": result["image_id"]})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    coco_result = coco.loadRes(stored_results)
    coco_eval = COCOEvalCap(coco, coco_result)

    imgIds = coco_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = coco_eval.coco.imgToAnns[imgId]
        res[imgId] = coco_eval.cocoRes.imgToAnns[imgId]

    eval_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    eval_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]
        scores = scores[n - 1]

    # screen good case
    if screen_percent:
        good_case = []
        bad_case = []

        for i, sample_score in enumerate(scores):
            screen_results[i][f"{metric}-score"] = sample_score
            if sample_score >= screen_percent * COCO_METRICS_SCORE_LIMIT[metric]:
                good_case.append(screen_results[i])
            else:
                bad_case.append(screen_results[i])

        good_path = generate_submission_file(f"{task_name}-{metric}.json", args, subpath="goodcase")
        with open(good_path, "w") as f:
            json.dump(good_case, f, indent=4)

        bad_path = generate_submission_file(f"{task_name}-{metric}.json", args, subpath="badcase")
        with open(bad_path, "w") as f:
            json.dump(bad_case, f, indent=4)

    path = generate_submission_file("coco_captions_val2014_alg_results.json", args)
    if not os.path.exists(path):
        eval_logger.info("Storing prediction that can be submitted to the server ...")
        with open(path, "w") as f:
            json.dump(stored_results, f, indent=4)

    return score


def coco_bleu4(results, args):
    return coco_aggregation_result(results, "Bleu_4", args)


def coco_bleu3(results, args):
    return coco_aggregation_result(results, "Bleu_3", args)


def coco_bleu2(results, args):
    return coco_aggregation_result(results, "Bleu_2", args)


def coco_bleu1(results, args):
    return coco_aggregation_result(results, "Bleu_1", args)


def coco_meteor(results, args):
    return coco_aggregation_result(results, "METEOR", args)


def coco_rougel(results, args):
    return coco_aggregation_result(results, "ROUGE_L", args)


def coco_cider(results, args):
    return coco_aggregation_result(results, "CIDEr", args)


def coco_spice(results, args):
    return coco_aggregation_result(results, "SPICE", args)


def coco_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case coco_passthrough), value: metric value
    """
    question_id = doc["question_id"]
    # The question id in our dataset is the image file itself
    image_id = int(question_id.split("_")[-1].split(".")[0])
    return {"coco_passthrough": {"pred": result, "image_id": image_id}}


def coco_test_aggregation_result(results, args):
    stored_results = []
    for result in results:
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})

    path = generate_submission_file("coco_captions_test2014_alg_results.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Your test result has been stored in to {path}. Make sure you also have the val result stored to submit to the server on https://codalab.lisn.upsaclay.fr/competitions/7404#participate.")
