import json

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file


SQA_METRICS = ["exact_match"]


def sqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    context, question, choices = doc["hint"], doc["question"], doc["choices"]
    len_choices = len(choices)
    options = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
    if lmms_eval_specific_kwargs["format"] == "default":
        if context:
            context = f"Context: {context}\n"

        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
        return f"{pre_prompt}{context}{question}\n{choices_str}{post_prompt}"
    elif lmms_eval_specific_kwargs["format"] == "qwen_vl":
        prompt = "Context: {}\nQuestion: {}\nOptions: {}\nAnswer:"
        context = context if context else "N/A"
        prompt = prompt.format(context, question, choices_str)
        return prompt
    else:
        raise ValueError(f"Unknown prompt format: {lmms_eval_specific_kwargs}")


def sqa_doc_to_visual(doc):
    if doc["image"] is None:
        return []
    return [doc["image"].convert("RGB")]


def sqa_doc_to_target(doc):
    len_choices = len(doc["choices"])
    options = [chr(ord("A") + i) for i in range(len_choices)]
    return options[doc["answer"]]


def sqa_process_results(doc, results):
    # I know this is weird, but it's how llava parse it.
    target = sqa_doc_to_target(doc).strip().lower()
    data_dict = {
        "question": doc["question"],
        "hint": doc["hint"],
        "task": doc["task"],
        "grade": doc["grade"],
        "subject": doc["subject"],
        "topic": doc["topic"],
        "category": doc["category"],
        "skill": doc["skill"],
        "lecture": doc["lecture"],
        "solution": doc["solution"],
        "choices": doc["choices"],
        "answer": target,
    }
    pred = results[0].strip()
    if pred.lower() == target:
        data_dict["pred"] = pred.lower()
        data_dict["exact_match"] = 1.0
        return {f"sqa_{metric}": data_dict for metric in SQA_METRICS}
    # pattern: ^[A-Z]\. .*
    if len(pred) >= 2 and pred[0].isupper() and pred[1] == ".":
        result = 1.0 if pred[0].lower() == target else 0.0
        data_dict["pred"] = pred[0].lower()
        data_dict["exact_match"] = result
        return {f"sqa_{metric}": data_dict for metric in SQA_METRICS}

    data_dict["pred"] = pred
    data_dict["exact_match"] = 0.0
    return {f"sqa_{metric}": data_dict for metric in SQA_METRICS}


def sqa_aggregation_result(results, metric, args):
    task_name = "scienceqa"
    if args and args.tasks:
        for task in args.tasks.split(","):
            if "scienceqa" in task:
                task_name = task
                break

    screen_percent = 0
    if args and args.tasks_screen_thres:
        for k in args.tasks_screen_thres.keys():
            if "scienceqa" in k:
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


def sqa_mean(results, args):
    return sqa_aggregation_result(results, "mean", args)