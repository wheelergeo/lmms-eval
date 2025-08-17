import json
import copy


from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


def seed_doc_to_visual(doc):
    return [image.convert("RGB") for image in doc["image"]]


def seed_doc_to_text(doc):
    question = doc["question"]
    question += "\n" + f"A. {doc['choice_a']}\n"
    question += f"B. {doc['choice_b']}\n"
    question += f"C. {doc['choice_c']}\n"
    question += f"D. {doc['choice_d']}"
    return f"{question}\nAnswer with the option's letter from the given choices directly."


def seed_process_result(doc, result):
    pred = result[0].strip()
    if len(pred) > 1:
        pred = pred[0]
    answer = doc["answer"]
    data_type = doc["data_type"]

    dict_data = {
        "question_id": doc["question_id"],
        "question_type_id": doc["question_type_id"],
        "question": doc["question"],
        "data_type": data_type,
        "data_id": doc["data_id"],
        "choice_a": doc["choice_a"],
        "choice_b": doc["choice_b"],
        "choice_c": doc["choice_c"],
        "choice_d": doc["choice_d"],
        "answer": doc["answer"],
        "pred": pred,
    }

    return {
        f"seed_{data_type}": dict_data, 
        f"seed_all": {"pred": pred, "answer": answer, "question_id": doc["question_id"]}
    }


def seed_aggregation_result(results):
    total_count = 0
    total_correct = 0
    scores = []
    for result in results:
        if result["pred"].lower().strip() == result["answer"].lower().strip():
            total_correct += 1
            scores.append(1.0)
        else:
            scores.append(0.0)
        total_count += 1
    return total_correct / total_count, scores


def seed_aggregation_result_all(results, metric, args):
    task_name = "seedbench"
    if args and args.tasks:
        for task in args.tasks.split(","):
            if "seedbench" in task:
                task_name = task
                break

    screen_percent = 0
    if args and args.tasks_screen_thres:
        for k in args.tasks_screen_thres.keys():
            if "seedbench" in k:
                screen_percent = args.tasks_screen_thres[k]
                break

    score, scores = seed_aggregation_result(results)
    stored_results = []
    screen_results = copy.deepcopy(results)
    good_case = []
    bad_case = []
    for i, result in enumerate(results):
        stored_results.append({"question_id": result["question_id"], "prediction": result["pred"]})
        
        if screen_percent and metric != "all":
            screen_results[i]["score"] = scores[i]
            if scores[i] >= screen_percent:
                good_case.append(screen_results[i])
            else:
                bad_case.append(screen_results[i])

    # screen good case
    if screen_percent and metric != "all":
        good_path = generate_submission_file(f"{task_name}-{metric}.json", args, subpath="goodcase")
        with open(good_path, "w") as f:
            json.dump(good_case, f, indent=4)

        bad_path = generate_submission_file(f"{task_name}-{metric}.json", args, subpath="badcase")
        with open(bad_path, "w") as f:
            json.dump(bad_case, f, indent=4)

    with open("./seed_submission.json", "w") as f:
        json.dump(stored_results, f, indent=4)
    print("Storing files for seed_submission ...")

    return score


def seed_doc_to_text_mc(doc):
    question = doc["question"]
    return f"{question} Answer :"


def seed_doc_to_choice(doc):
    return [doc["choice_a"], doc["choice_b"], doc["choice_c"], doc["choice_d"]]


def seed_doc_to_mc_target(doc):
    answer2choice = {"A": "choice_a", "B": "choice_b", "C": "choice_c", "D": "choice_d"}
    return doc[answer2choice[doc["answer"]]]


def seed_image(results, args):
    return seed_aggregation_result_all(results, "image", args)


def seed_video(results, args):
    return seed_aggregation_result_all(results, "video", args)


def seed_all(results, args):
    return seed_aggregation_result_all(results, "all", args)