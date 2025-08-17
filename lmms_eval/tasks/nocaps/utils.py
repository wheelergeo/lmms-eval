import json
import os
import copy

from loguru import logger as eval_logger
from pycocoevalcap.eval import Bleu, Cider, COCOEvalCap, Meteor, Rouge, Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

dir_name = os.path.dirname(os.path.abspath(__file__))

NOCAPS_METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]  # , "SPICE"]
NOCAPS_METRICS_SCORE_LIMIT = {"Bleu_4": 1.0, "Bleu_3": 1.0, "Bleu_2": 1.0, "Bleu_1": 1.0, "METEOR": 1.0, "ROUGE_L": 1.0, "CIDEr": 8.0}


def nocaps_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def nocaps_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # question = "Please carefully observe the image and come up with a caption for the image"
    return lmms_eval_specific_kwargs["prompt"]


def nocaps_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name, value: metric value
    """
    # The question id in our dataset is the image file itself

    data_dict = {
        "image_id": doc["image_id"],
        "image_file_name": doc["image_file_name"],
        "image_coco_url": doc["image_coco_url"],
        "answer": doc["annotations_captions"], 
        "pred": result[0]
    }

    return {f"nocaps_{metric}": data_dict for metric in NOCAPS_METRICS}


def nocaps_aggregation_result(results, metric, args=None):
    scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]  # , (Spice(), "SPICE")]
    scorers_dict = {s[1]: s for s in scorers}

    task_name = "nocaps_val"
    if args and args.tasks:
        for task in args.tasks.split(","):
            if "nocaps" in task:
                task_name = task
                break

    screen_percent = 0
    if args and args.tasks_screen_thres:
        for k in args.tasks_screen_thres.keys():
            if "nocaps" in k:
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

    nocaps_result = coco.loadRes(stored_results)
    nocaps_eval = COCOEvalCap(coco, nocaps_result)

    imgIds = nocaps_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = nocaps_eval.coco.imgToAnns[imgId]
        res[imgId] = nocaps_eval.cocoRes.imgToAnns[imgId]

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
            if sample_score >= screen_percent * NOCAPS_METRICS_SCORE_LIMIT[metric]:
                good_case.append(screen_results[i])
            else:
                bad_case.append(screen_results[i])

        good_path = generate_submission_file(f"{task_name}-{metric}.json", args, subpath="goodcase")
        with open(good_path, "w") as f:
            json.dump(good_case, f, indent=4)

        bad_path = generate_submission_file(f"{task_name}-{metric}.json", args, subpath="badcase")
        with open(bad_path, "w") as f:
            json.dump(bad_case, f, indent=4)

    path = generate_submission_file(f"nocaps_val_{metric}_scores.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)
    eval_logger.info(f"Your result has been saved to {path}.")

    return score


def nocaps_bleu4(results, args=None):
    return nocaps_aggregation_result(results, "Bleu_4", args)


def nocaps_bleu3(results, args=None):
    return nocaps_aggregation_result(results, "Bleu_3", args)


def nocaps_bleu2(results, args=None):
    return nocaps_aggregation_result(results, "Bleu_2", args)


def nocaps_bleu1(results, args=None):
    return nocaps_aggregation_result(results, "Bleu_1", args)


def nocaps_meteor(results, args=None):
    return nocaps_aggregation_result(results, "METEOR", args)


def nocaps_rougel(results, args=None):
    return nocaps_aggregation_result(results, "ROUGE_L", args)


def nocaps_cider(results, args=None):
    return nocaps_aggregation_result(results, "CIDEr", args)


def nocaps_spice(results, args=None):
    return nocaps_aggregation_result(results, "SPICE", args)


def nocaps_test_process_result(doc, result):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case nocaps_passthrough), value: metric value
    """
    return {"nocaps_passthrough": {"pred": result[0], "image_id": doc["image_id"]}}


def nocaps_test_aggregation_result(results, args=None):
    stored_results = []
    for result in results:
        stored_results.append({"image_id": int(result["image_id"]), "caption": result["pred"]})

    path = generate_submission_file("nocaps_captions_nocaps_test_alg_results.json", args)
    eval_logger.info("Storing prediction that can be submitted to the server ...")
    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    eval_logger.info(f"Your test result has been stored in {path}. Make sure you also have the val result stored to submit to the server on https://codalab.lisn.upsaclay.fr/competitions/7404#participate.")
