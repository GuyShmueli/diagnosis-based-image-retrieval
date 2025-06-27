"""
Objective:
  Extract “case…” sections (e.g. Case Presentation, Case Report, Patient Summary, etc.) 
  from a list of PMC .nxml files so you can later compute diagnosis similarity via generated queries.

Usage:
  python extract_case_summaries.py \
    --jsons_path /cs/labs/tomhope/dhtandguy21/largeListsGuy
"""
import os
import json
import argparse
from lxml import etree
from multiprocessing import Pool


def process_file_list(file_list):
    """
    Given a list of file paths, parse each file, run the XPath,
    gather the text from <sec> paragraphs, return a dict { filepath: text }.
    """
    # Create an XPath expression that matches <sec> elements with a <title> containing any of the keywords.
    # Using translate() ensures a case-insensitive search.
    xpath_expr = (
        "//sec["
        "  contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case presentation') "
        "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case report') "
        "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case description') "
        "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case summary') "
        "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'patient presentation') "
        "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'patient report') "
        "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'patient description') "
        "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'patient summary') "
        "or contains(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'case') "
        "or (normalize-space(translate(title, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')) = 'patient') "
        "]"
    )
    single_pair_case_summary = {}
    for file in file_list:
        tree = etree.parse(file)
        sections = tree.xpath(xpath_expr)
        for sec in sections:
            paragraphs = sec.xpath(".//p//text()")
            text = " ".join(paragraphs)
            single_pair_case_summary[file] = text
    return single_pair_case_summary


def multiprocess_case_summaries(matching_uids_xml_paths):
    with Pool() as pool:
        # Map each sub-list to a worker process
        case_summary_pairs_list = pool.map(process_file_list, matching_uids_xml_paths)

    return case_summary_pairs_list


def calc_indices_summaries_list(case_summary_pairs_list):
    """
    Returns two lists:
      - A list of indices corresponding to elements depicting both cases.
      - A list of indices corresponding to elements depicting less than 2 cases.
    """
    valid_indices = [idx for idx in range(len(case_summary_pairs_list)) if len(case_summary_pairs_list[idx]) == 2]
    invalid_indices = [idx for idx in range(len(case_summary_pairs_list)) if len(case_summary_pairs_list[idx]) != 2]
    return valid_indices, invalid_indices


def main(jsons_path):
    with open (os.path.join(jsons_path, 'matching_uids_xml_paths.json'), 'r') as f:
        matching_uids_xml_paths = json.load(f)

    case_summary_pairs_list = multiprocess_case_summaries(matching_uids_xml_paths)
    case_summary_pairs_list = [list(case_summary_pairs_list[idx].values()) for idx in range(len(case_summary_pairs_list))]

    summaries_valid_indices, invalid_indices = calc_indices_summaries_list(case_summary_pairs_list)

    with open (os.path.join(jsons_path, 'case_summary_pairs_list.json'), 'w') as f:
        json.dump(case_summary_pairs_list, f)

    with open(os.path.join(jsons_path, 'summaries_valid_indices.json'), 'w') as f:
        json.dump(summaries_valid_indices, f)

    print(f"Extracted {len(summaries_valid_indices)} valid case summaries.")
    print(f"There are {len(invalid_indices)} missing case summaries.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract case summaries from NXML files."
    )
    parser.add_argument(
        "--jsons_path", "-j",
        default='/cs/labs/tomhope/dhtandguy21/largeListsGuy',
        help="directory containing matching_uids_xml_paths.json (default: %(default)s)"
    )
    args = parser.parse_args()

    main(args.jsons_path)