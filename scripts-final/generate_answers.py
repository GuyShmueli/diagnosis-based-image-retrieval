"""
Objective:
  Generate and submit a batch of prompts asking for the answers to the questions from before
  (about the similarities and differences between pairs of patient cases, focused on diagnosis).

Usage:
  python generate_answer_batches.py \
    --path-case-summary-list      /path/to/case_summary_pairs.json \
    --path-citation-reason        /path/to/citation_reasons.json \
    --path-valid-citations        /path/to/citation_valid_indices.json \
    --path-common-indices         /path/to/common_indices.json \
    --path-generated-questions    /path/to/generated_questions.json \
    --path-file-A                 /output/path/batch_part1.jsonl \
    --path-file-B                 /output/path/batch_part2.jsonl \
    --openai-key                  YOUR_OPENAI_API_KEY
"""
from openai import OpenAI
import argparse
import json
import os


def create_batch_requests(common_indices, case_summary_pairs_list, citation_reason_list,
                          citation_valid_indices_dict, generated_questions,
                          path_file_A, path_file_B):
    """ Write a JSONL file of chat-completion requests."""
    # Create the prompt by taking into account both precision and costs
    answering_prompt = """You will be given case details for Case A, Case B, Reason for Citation (one case cites the other), along with a list of questions.
    You are an experienced clinician specializing in medical imaging. Previously, a list of comparative, clinically oriented questions was generated for the two cases.
    Now, your task is to answer those questions in a way that highlights the diagnostic similarities and differences between the two cases.
    DO NOT provide any additional information or context outside of the answers to the questions.
    """
    # Write to file A
    os.makedirs(os.path.dirname(path_file_A), exist_ok=True)
    with open(path_file_A, "w", encoding="utf-8") as f:
        for idx in range(len(common_indices))[:len(common_indices)//2]:
            common_index = common_indices[idx]
            case1 = case_summary_pairs_list[common_index][0]
            case2 = case_summary_pairs_list[common_index][1]
            citation_reason = citation_reason_list[citation_valid_indices_dict[common_index]]
            curr_questions = generated_questions[idx]

            answering_user = f"""Below are the two case descriptions and the set of questions to be answered.
            ---
            Case A:
            {case1}

            Case B:
            {case2}

            Reason for Citation, Possible Similarities and Explanation:
            {citation_reason}

            Questions:
            {curr_questions}
            ---
            Provide your answers now:
            """

            request_obj = {
                "custom_id": f"request-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                        "model": "gpt-4.1-mini",
                        "messages": [
                            {
                                "role": "developer",
                                "content": f"{answering_prompt}"
                            },
                            {
                                "role": "user",
                                "content": f"{answering_user}"
                            }
                        ],
                }
            }
            # Write each object on a new line in the JSONL file
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")

    # Write to file B
    os.makedirs(os.path.dirname(path_file_B), exist_ok=True)
    with open(path_file_B, "w", encoding="utf-8") as f:
        for idx in range(len(common_indices))[len(common_indices)//2:]:
            common_index = common_indices[idx]
            case1 = case_summary_pairs_list[common_index][0]
            case2 = case_summary_pairs_list[common_index][1]
            citation_reason = citation_reason_list[citation_valid_indices_dict[common_index]]
            curr_questions = generated_questions[idx]

            answering_user = f"""Below are the two case descriptions and the set of questions to be answered.
            ---
            Case A:
            {case1}

            Case B:
            {case2}

            Reason for Citation, Possible Similarities and Explanation:
            {citation_reason}

            Questions:
            {curr_questions}
            ---
            Provide your answers now:
            """

            request_obj = {
                "custom_id": f"request-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                        "model": "gpt-4.1-mini",
                        "messages": [
                            {
                                "role": "developer",
                                "content": f"{answering_prompt}"
                            },
                            {
                                "role": "user",
                                "content": f"{answering_user}"
                            }
                        ],
                }
            }
            # Write each object on a new line in the JSONL file
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")


def main(args):
    # 1. Load all the required JSON inputs
    with open(args.path_case_summary_list, 'r') as f:
        case_summary_pairs_list = json.load(f)
    with open(args.path_citation_reason, 'r') as f:
        citation_reason_list = json.load(f)
    with open(args.path_valid_citations, 'r') as f:
        citation_valid_indices = json.load(f)
    with open(args.path_common_indices, 'r') as f:
        common_indices = json.load(f)
    with open(args.path_generated_questions, 'r') as f:
        generated_questions = json.load(f)

    # 2. Build a fast lookup for which citation index maps to which reason
    # citation_valid_indices_dict will not be saved as json because it turns the keys into strings (instead of ints)
    citation_valid_indices_dict = {citation_valid_indices[idx]: idx
                                   for idx in range(len(citation_valid_indices))}

    # 3. Initialize the OpenAI client
    client = OpenAI(api_key=args.openai_key)
    

    # 4. Generate the two JSONL batch-request files
    create_batch_requests(common_indices, case_summary_pairs_list, citation_reason_list,
                          citation_valid_indices_dict, generated_questions,
                          args.path_file_A, args.path_file_B)

    # 5. Upload each JSONL as a "batch" file to OpenAI
    batch_input_file_A = client.files.create(
                            file=open(args.path_file_A, "rb"),
                            purpose="batch"
                            )
    batch_input_file_B = client.files.create(
                            file=open(args.path_file_B, "rb"),
                            purpose="batch"
                            )
                        
    # 6. Kick off the batch completion jobs for part A and part B
    client.batches.create(
    input_file_id=batch_input_file_A.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Answers Generation part A"}
    )
    client.batches.create(
    input_file_id=batch_input_file_B.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Answers Generation part B"}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and submit comparative diagnostic answer batches")
    parser.add_argument("--path-case-summary-list", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/path_case_summary_list.json")
    parser.add_argument("--path-citation-reason", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/o3_mini_response_list.json")
    parser.add_argument("--path-valid-citations", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/citation_valid_indices.json")
    parser.add_argument("--path-file-A", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/batch_requests_answer_gen_part1.jsonl")
    parser.add_argument("--path-file-B", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/batch_requests_answer_gen_part2.jsonl")    
    parser.add_argument("--path-common-indices", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/common_indices.json")    
    parser.add_argument("--path-generated-questions", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/generated_questions.json")        
    parser.add_argument("--openai-key",           
                        required=True, help="Your OpenAI API key")
    args = parser.parse_args()

    main(args)