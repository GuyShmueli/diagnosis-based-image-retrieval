"""
Objective:
  Build prompts for classifying positive pairs of medical-image captions.
  Loads previously generated Q&A, caption paths, stems, and indices, filters
  out unwanted captions, then emits a JSONL file for batch OpenAI chat calls.

Usage:
  python generate_positive_prompts.py \
    --answers-json /path/to/generated_answers.json \
    --questions-json /path/to/generated_questions.json \
    --pairs-text-json /path/to/pairs_text.json \
    --common-indices-json /path/to/common_indices.json \
    --stems-3d-json /path/to/stems_3d.json \
    --diagrams-filtered-json /path/to/diagrams_compound_filtered_paths_list.json \
    --output-jsonl /path/to/generate_positives.jsonl \
    --openai-key YOUR_OPENAI_API_KEY
"""

import os
import json
import argparse
from pathlib import Path
from openai import OpenAI

def build_stem_caption_3d(pairs_text, stems_3d):
    """
    Return a 3D list of dicts mapping stem -> caption, parallel to pairs_text.
    """
    return [
        [
            {stem: cap for stem, cap in zip(stems_patient, caps_patient)}
            for stems_patient, caps_patient in zip(stems_pair, caps_pair)
        ]
        for stems_pair, caps_pair in zip(stems_3d, pairs_text)
    ]

def main(args):
    # 1) Load all required JSON files
    with open(args.answers_json, 'r', encoding='utf-8') as f:
        generated_answers = json.load(f)
    with open(args.questions_json, 'r', encoding='utf-8') as f:
        generated_questions = json.load(f)
    with open(args.pairs_text_json, 'r', encoding='utf-8') as f:
        pairs_text = json.load(f)
    with open(args.common_indices_json, 'r', encoding='utf-8') as f:
        common_indices = json.load(f)
    with open(args.stems_3d_json, 'r', encoding='utf-8') as f:
        stems_3d = json.load(f)
    with open(args.diagrams_filtered_json, 'r', encoding='utf-8') as f:
        diagrams_filtered_paths = json.load(f)

    # 2) Build the 3D stem->caption structure
    stem_caption_list = build_stem_caption_3d(pairs_text, stems_3d)

    # 3) Filter out captions not in the diagram/compound-filtered set
    diagrams_stems = {Path(p).stem for p in diagrams_filtered_paths}
    filtered_stem_caption_list = []
    for group in stem_caption_list:
        new_group = []
        for d in group:
            filtered = {k: v for k, v in d.items() if k in diagrams_stems}
            if filtered:
                new_group.append(filtered)
        filtered_stem_caption_list.append(new_group)

    # 4) Identify invalid pair indices
    invalid_indices = {
        i for i, grp in enumerate(filtered_stem_caption_list) if len(grp) < 2
    }

    # 5) Prepare the system/user prompt template
    system_prompt = (
        "You are a medical-AI expert. Your job is to classify *pairs* of "
        "medical-image captions as **positive** if they share the same "
        "diagnosis & modality.\n"
        "NOTE: Classify **only inter-case pairs** (the two caption-IDs come from "
        "different cases).\n"
        "Output **only** valid JSON matching this schema (no markdown):\n"
        "{\n"
        '  "pair_id": "<label_1, label_2>",\n'
        '  "reasoning": "<short explanation why it is positive>",\n'
        '  "modality": "<the shared modality>",\n'
        '  "anatomy": "<the anatomy>",\n'
        '  "diagnosis": "<the shared diagnosis you found>",\n'
        "}\n"
        "DON'T classify (return nothing) if any is true:\n"
        "1. They are negatives.\n"
        "2. Either caption describes a diagram/illustration.\n"
        "3. Either caption describes a compound figure (labels like 'A', 'B', '(b)', etc.)\n"
        "\n"
        "FINAL SANITY CHECK:\n"
        "✓ Inter-case pair?\n"
        "✓ Same diagnosis **and** modality?\n"
        "✓ No diagram/illustration words?\n"
        "✓ No compound-figure cues?\n"
        "If all YES → produce the JSON object.\n"
        "Otherwise → output {}."
    )

    # 6) Build the list of chat-completion requests
    prompts = []
    for idx, common_idx in enumerate(common_indices):
        if common_idx in invalid_indices:
            continue
        captions1 = filtered_stem_caption_list[common_idx][0]
        captions2 = filtered_stem_caption_list[common_idx][1]
        q = generated_questions[idx]
        a = generated_answers[idx]

        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": (
                "### PATIENT-CONTEXT QUERIES & ANSWERS\n"
                f"{q}\n\n{a}\n\n"
                "### IMAGE CAPTIONS (MULTIPLE PAIRS)\n"
                f"{captions1}\n\n{captions2}"
            )}
        ]
        prompts.append(messages)

    # 7) Write out the JSONL for batch upload
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    with open(args.output_jsonl, 'w', encoding='utf-8') as out_f:
        for i, messages in enumerate(prompts):
            req = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "o4-mini",
                    "reasoning_effort": "low",
                    "messages": messages
                }
            }
            out_f.write(json.dumps(req, ensure_ascii=False) + "\n")

    # 8) Upload and create the batch
    client = OpenAI(api_key=args.openai_key)
    batch_file = client.files.create(
        file=open(args.output_jsonl, "rb"),
        purpose="batch"
    )

    client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": args.batch_description}
    )

    print(f"Wrote {len(prompts)} requests to {args.output_jsonl} and started batch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a batch JSONL of prompts for OpenAI chat completions.")
    parser.add_argument("--answers-json", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/generated_answers.json")
    parser.add_argument("--questions-json",
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/generated_questions.json")
    parser.add_argument("--pairs-text-json",
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/pairs_text.json")
    parser.add_argument("--common-indices-json",  
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/common_indices.json")
    parser.add_argument("--stems-3d-json",       
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/stems_3d.json")
    parser.add_argument("--diagrams-filtered-json", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/diagrams_compound_filtered_paths_list.json")
    parser.add_argument("--output-jsonl",         
                        default="/cs/labs/tomhope/dhtandguy21/largeListsGuy/generate_positives.jsonl")
    parser.add_argument("--openai-key",           
                        required=True, help="Your OpenAI API key")
    parser.add_argument("--batch-description",    
                        default="Generating captions positive pairs")
    args = parser.parse_args()
    main(args)