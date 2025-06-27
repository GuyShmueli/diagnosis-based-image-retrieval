"""
Objective:
  Generate and submit a batch of prompts asking for questions about the similarities and
  differences between pairs of patient cases, focused on diagnosis.

Usage:
  python generate_question_batches.py \
    --path-case-summary-list /path/to/case_summary_pairs.json \
    --path-citation-reason /path/to/citation_reason_list.json \
    --path-uids-pmid-list /path/to/matching_uids_pmid_list.json \
    --path-valid-summaries /path/to/summaries_valid_indices.json \
    --path-valid-citations /path/to/citation_valid_indices.json \
    --path-common-indices  /path/to/common_indices.json \
    --path-file-A /path/to/questions_batch_requests_A.jsonl \
    --path-file-B /path/to/questions_batch_requests_B.jsonl \
    --openai-key YOUR_OPENAI_API_KEY
"""
from openai import OpenAI
import argparse
import json
import os


def find_common_indices(summaries_valid_indices, citation_valid_indices, matching_uids_pmid_list):
    """
    Find the common indices between summaries_valid_indices and citation_valid_indices:
      - summaries_valid_indices - represents the indices corresponding to successfully extracted case summaries (0-31,691) 
      - citation_valid_indices - represents the indices corresponding to successfully cited-citing pairs (0-31,691)
    """
    summaries_valid_indices_set = set(summaries_valid_indices)
    citation_valid_indices_set = set(citation_valid_indices)
    common_indices = []

    for i in range(len(matching_uids_pmid_list)):
        if i in summaries_valid_indices_set and i in citation_valid_indices_set:
            common_indices.append(i)
    
    return common_indices



def create_batch_requests(common_indices, case_summary_pairs_list, citation_reason_list,
                          citation_valid_indices_dict, path_file_A, path_file_B):
    """ Write a JSONL file of chat-completion requests."""
    question_generating_prompt = """
    You are an experienced clinician specializing in medical imaging. You will be given two cases and the reason for citation between the two (one case report cites the other), and asked to generate a set of comparative, clinically relevant questions that highlight both similarities and differences, helping differentiate potential diagnoses. Your questions should:

    1. **Focus on Diagnostic Clues and Clinical Significance:**
    - Avoid trivial or purely technical details (like minor differences in image orientation).
    - Emphasize findings that inform diagnosis (e.g., presence of lesions, pattern of abnormalities, symptoms).

    2. **Use a Comparative Framing:**
    - Explicitly compare Case A and Case B.
    - Aim for questions that probe both common ground and distinguishing features.
    - If possible, try asking mostly yes/no questions.

    3. **Follow a Hierarchical Reasoning Approach:**
    - **Level 1 (Broad Context)**: Are both cases the same modality/organ system?
    - **Level 2 (General Diagnosis Category)**: Are they both infectious, both neoplastic, etc.?
    - **Level 3 (Specific Features & Findings)**: Detailed imaging findings (e.g., cavitation, consolidation, nodules), clinical presentation, organism type, etc.

    4. **Consider Step-by-Step Reasoning:**
    - You may first silently analyze the diagnoses and imaging findings for each case.
    - Then generate questions that a clinician would naturally ask to tease out whether the two cases have the same underlying pathology or different pathologies.

    5. **Provide a subtitle specifying the level (1/2/3) of the current set of questions.**
    """

    question_generating_user = f"""
    Below are two cases and the reason for citation between the two. Read them carefully, then generate a concise list of question prompts (in numbered format) that compare these cases on clinically meaningful aspects.

    ---
    Case A:
    %s

    Case B:
    %s

    Reason for Citation, Possible Similarities and Explanation:
    %s
    ---

    Now, **generate the list of comparative, clinically relevant questions** that a radiologist or physician might ask to determine whether these two cases share a diagnosis or differ in key features. 

    Remember:
    - Maintain clinical depth: mention imaging findings, pathophysiology, or hallmark symptoms.
    - Emphasize similarities/differences that impact diagnosis.
    - Keep the questions focused and direct, as if you are performing a differential diagnosis.
    - Avoid vague or generic questions; ensure each question contributes to understanding diagnostic overlap or divergence.
    - Start with broad-level questions (anatomy, broad diagnosis similarity), then proceed to more specific questions about the nature of the condition, causative agents, imaging signs, etc.
    - Adhere to yes/no questions, if possible.
    - Provide a subtitle specifying the level (1/2/3) of each set of questions.

    Provide your final list of questions now.
    """
    # Writing to file A
    os.makedirs(os.path.dirname(path_file_A), exist_ok=True)
    with open(path_file_A, "w", encoding="utf-8") as f:
        for idx in range(len(common_indices))[:len(common_indices)//2]:
            common_index = common_indices[idx]
            case1 = case_summary_pairs_list[common_index][0]
            case2 = case_summary_pairs_list[common_index][1]
            citation_reason = citation_reason_list[citation_valid_indices_dict[common_index]]
            request_obj = {
                "custom_id": f"request-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "o3-mini",
                    "reasoning_effort": "low",
                    "messages": [
                        {"role": "developer", "content": question_generating_prompt},
                        {"role": "user", "content": question_generating_user % (case1, case2, citation_reason)}
                    ]
                }
            }
            # Write each object on a new line in the JSONL file
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
    # Writing to file B
    os.makedirs(os.path.dirname(path_file_B), exist_ok=True)
    with open(path_file_B, "w", encoding="utf-8") as f:
        for idx in range(len(common_indices))[len(common_indices)//2:]:
            common_index = common_indices[idx]
            case1 = case_summary_pairs_list[common_index][0]
            case2 = case_summary_pairs_list[common_index][1]
            citation_reason = citation_reason_list[citation_valid_indices_dict[common_index]]
            request_obj = {
                "custom_id": f"request-{idx+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "o3-mini",
                    "reasoning_effort": "low",
                    "messages": [
                        {"role": "developer", "content": question_generating_prompt},
                        {"role": "user", "content": question_generating_user % (case1, case2, citation_reason)}
                    ]
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
    with open(args.path_uids_pmid_list, 'r') as f:
        matching_uids_pmid_list = json.load(f)
    with open(args.path_valid_summaries, 'r') as f:
        summaries_valid_indices = json.load(f)
    with open(args.path_valid_citations, 'r') as f:
        citation_valid_indices = json.load(f)

    # 2. Find the common indices between summaries_valid_indices and citation_valid_indices
    common_indices = find_common_indices(summaries_valid_indices, citation_valid_indices, matching_uids_pmid_list)
    with open(args.path_common_indices, 'w') as f:
        json.dump(common_indices, f)

    # 3. Build a fast lookup for which citation index maps to which reason
    # citation_valid_indices_dict will not be saved as json because it turns the keys into strings (instead of ints)
    citation_valid_indices_dict = {citation_valid_indices[idx]: idx for idx in range(len(citation_valid_indices))}



    # 4. Initialize the OpenAI client
    client = OpenAI(api_key=args.openai_key)
    

    # 5. Generate the two JSONL batch-request files
    create_batch_requests(common_indices, case_summary_pairs_list, citation_reason_list,
                          citation_valid_indices_dict, args.path_file_A, args.path_file_B)

    # 6. Upload each JSONL as a "batch" file to OpenAI
    batch_input_file_A = client.files.create(
                            file=open(args.path_file_A, "rb"),
                            purpose="batch")
    batch_input_file_B = client.files.create(
                            file=open(args.path_file_B, "rb"),
                            purpose="batch")
                        
    # 7. Kick off the batch completion jobs for part A and part B
    client.batches.create(
    input_file_id=batch_input_file_A.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Questions Generation part A"}
    )
    client.batches.create(
    input_file_id=batch_input_file_B.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Questions Generation part B"}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and submit comparative diagnostic question batches")
    parser.add_argument("--path-case-summary-list", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/path_case_summary_list.json")
    parser.add_argument("--path-citation-reason", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/o3_mini_response_list.json")
    parser.add_argument("--path-uids-pmid-list", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/matching_uids_pmid_list.json")
    parser.add_argument("--path-valid-summaries", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/summaries_valid_indices.json")
    parser.add_argument("--path-valid-citations", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/citation_valid_indices.json")
    parser.add_argument("--path-file-A", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/questions_batch_requests_A.jsonl")
    parser.add_argument("--path-file-B", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/questions_batch_requests_B.jsonl")    
    parser.add_argument("--path-common-indices", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/common_indices.json")    
    parser.add_argument("--openai-key",           
                        required=True, help="Your OpenAI API key")
    args = parser.parse_args()

    main(args)