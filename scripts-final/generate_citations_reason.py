"""
Objective:
  Generate and submit a batch of citation-analysis prompts for patients-pair cases.

Usage:
  python generate_citation_batch.py \
    --batch-requests /path/to/batch_requests.jsonl \
    --citation-list /path/to/citation_list.json \
    --openai-key YOUR_OPENAI_API_KEY
"""
from openai import OpenAI
import argparse
import json
import os


def create_batch_requests(path_batch_requests, citation_list):
    """ Write a JSONL file of chat-completion requests based on the citation list.

    Each record in `citation_list` should be a dict with keys:
      - citing_title
      - citing_abstract
      - citation_paragraph
      - cited_title
      - cited_abstract

    The output file will contain one JSON object per line, ready for OpenAI batch upload.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path_batch_requests), exist_ok=True)

    # Open the JSONL file for writing
    with open(path_batch_requests, "w", encoding="utf-8") as f:
        for i, record in enumerate(citation_list, start=1):
            # Build the user-facing content from the record field
            user_content = (
                f"Citing Title: {record.get('citing_title', '')}\n"
                f"Citing Abstract: {record.get('citing_abstract', '')}\n"
                f"Citation Paragraph: {record.get('citation_paragraph', '')}\n"
                f"Cited Title: {record.get('cited_title', '')}\n"
                f"Cited Abstract: {record.get('cited_abstract', '')}"
            )
            # Construct the JSON object for one request
            request_obj = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "o3-mini",
                    "reasoning_effort": "medium",
                    "messages": [
                        {"role": "developer", "content": 
                            """You are a medical citation analysis assistant. You will read excerpts from a 'citing paper' referencing another 'cited paper,' each describing a single-patient case report. Your task is to produce four pieces of information:

                            1. Reason for Citation (a short phrase or sentence about why the authors mention the cited paper).
                            2. Possible Similarities (a brief mention of how these patients’ cases align, if at all).
                            3. Explanation (one to two sentences summarizing how the citation is used).
                            4. Confidence (High, Medium, or Low), based on the degree of similarity between the single-patient cases:
                            - High: The text explicitly states that the two patients share significant clinical or therapeutic similarities.
                            - Medium: Some parallels are hinted at but lack explicit detail or direct confirmation.
                            - Low: The citation is vague or does not clearly connect the patients’ key features.

                            ----
                            Follow this exact output format:
                            Reason for Citation: <short phrase or sentence>
                            Possible Similarities: <brief mention>
                            Explanation: <one to two sentences>
                            Confidence: <High/Medium/Low>

                            ----
                            Read the following excerpt(s) and generate your final answer in the required format. Unify your response if multiple citations are present. If no citations are specified, base your response on the articles' titles and abstracts.
                            """
                        },

                        {"role": "user", "content": user_content}
                    ]
                }
            }
            # Write each object on a new line in the JSONL file
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")


def main(path_batch_requests, path_citation_list, openai_key):
    """Load the citation list, generate batch requests, and submit them to OpenAI."""
    # 1. Initialize the OpenAI client
    client = OpenAI(api_key=openai_key)
    
    # 2. Load the citation list from the provided JSON file
    with open(path_citation_list, 'r') as f:
        citation_list = json.load(f)

    # 3. Create the batch request JSONL
    create_batch_requests(path_batch_requests, citation_list)

    # 4. Upload JSONL file for batch processing
    batch_input_file = client.files.create(
                            file=open(path_batch_requests, "rb"),
                            purpose="batch")
                        
    # 5. Start the batch job
    client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Citation Reason Extraction"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a batch JSONL of prompts for OpenAI chat completions"
                                    "designated for finding the reason of citation.")
    parser.add_argument("--batch-requests", 
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/batch_requests.jsonl")
    parser.add_argument("--citation-list",
                        default="/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/citation_extrac_list.json")
    parser.add_argument("--openai-key",           
                        required=True, help="Your OpenAI API key")
    args = parser.parse_args()
    main(args.batch_requests, args.citation_list, args.openai_key)