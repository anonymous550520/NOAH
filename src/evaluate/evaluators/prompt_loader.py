def load_hallucination_prompt(recipe, result):
    template_path = "src/evaluate/prompts/hallucination_prompt.txt"
    with open(template_path, "r") as f:
        template = f.read()

    formatted_gt_events = "\n\n".join(
        f"Ground Truth Event #{i+1}:\n"
        f"Ground Truth Caption: {sentence.strip()}\n\n"
        f"Supporting Visual Description:\n{aug.strip()}"
        for i, (sentence, aug) in enumerate(zip(recipe["sentences"], recipe["aug_sentences"]))
    )

    prompt = (
        template
        .replace("{GROUND_TRUTH_EVENTS}", formatted_gt_events)
        .replace("{INFERENCE_CAPTION}", result["caption"].strip())
    )

    return prompt


def load_omission_prompt(recipe, result):
    
    template_path = "src/evaluate/prompts/omission_prompt.txt"
    with open(template_path, "r") as f:
        template = f.read()

    # Safe calculation of insert_position with error handling

    insert_position = recipe["insert_event_idx"] + 1

    # Regular mode: multiple events with one inserted
    formatted_gt_events = "\n\n".join(
        f"Ground Truth Event #{i+1}:\n"
        f"Ground Truth Caption: {sentence.strip()}\n\n"
        f"Supporting Visual Description:\n{aug.strip()}"
        for i, (sentence, aug) in enumerate(zip(recipe["sentences"], recipe["aug_sentences"]))
    )

    inserted_event = (
        "Ground Truth Caption: " + recipe["sentences"][recipe["insert_event_idx"]].strip() + "\n\n" 
        +"Supporting Visual Description:\n" + recipe["aug_sentences"][recipe["insert_event_idx"]].strip()
    )

    prompt = (
        template
        .replace("{GROUND_TRUTH_EVENTS}", formatted_gt_events)
        .replace("{INSERTED_EVENT}", inserted_event)
        .replace("{INSERT_POSITION}", str(insert_position))
        .replace("{INFERENCE_CAPTION}", result["caption"].strip())
    )

    return prompt