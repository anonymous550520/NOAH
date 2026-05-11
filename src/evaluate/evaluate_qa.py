import json
import argparse

def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def save_json(data, json_path):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

def parse_predict(predict):
    # parse yes no
    if predict is not None:
        predict = predict.lower()
        is_yes = "yes" in predict
        is_no = "no" in predict
        if is_yes and is_no:
            return "No"
        elif is_yes:
            return "Yes"
        else:
            return None
    return None

def evaluate_narrative(results):
    total_present_pairs = 0
    total_absent_pairs = 0
    correct_present_pairs = 0
    correct_absent_pairs = 0
    
    for items in results.values():
        # Separate present and absent questions
        present_questions = []
        absent_questions = []
        
        for item in items:
            question = item["question"].lower()
            if "absent in the video" in question:
                absent_questions.append(item)
            elif "present in the video" in question:
                present_questions.append(item)
        
        # Evaluate present questions
        if len(present_questions) > 0:
            total_present_pairs += 1
            all_present_correct = True
            for item in present_questions:
                if parse_predict(item["predict"]) != item["answer"]:
                    all_present_correct = False
                    break
            if all_present_correct:
                correct_present_pairs += 1
        
        # Evaluate absent questions
        if len(absent_questions) > 0:
            total_absent_pairs += 1
            all_absent_correct = True
            for item in absent_questions:
                if parse_predict(item["predict"]) != item["answer"]:
                    all_absent_correct = False
                    break
            if all_absent_correct:
                correct_absent_pairs += 1
        
    result = (correct_absent_pairs + correct_present_pairs) / (total_absent_pairs + total_present_pairs)

    return result

def run_evaluation(args):
    results = load_json(args.results)

    # group by id within narrative type
    narrative_results = {}

    for item in results:
        if item.get("type") != "narrative":
            continue
        item_id = item["id"]
        if item_id not in narrative_results:
            narrative_results[item_id] = []
        narrative_results[item_id].append(item)

    print(len(narrative_results))

    narrative_score = evaluate_narrative(narrative_results)
    print(narrative_score)

    #  Save results
    results = {
        "narrative": narrative_score,
    }

    save_json(results, args.output)

def parse_args():
    arg = argparse.ArgumentParser("Evaluate qa results")
    arg.add_argument("--results", required=True)
    arg.add_argument("--output", required=True)
    return arg.parse_args()

def main():
    args = parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()
    