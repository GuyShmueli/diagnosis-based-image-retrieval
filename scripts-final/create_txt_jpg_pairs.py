"""
Objective:
  Given a JSON of positive UID pairs and a filtered DataFrame of captions,
  produce two 3D JSON lists:
    i. pairs_text_list: [[ [txts for patient_i], [txts for patient_j] ], …]
    ii. pairs_image_list: [[ [jpgs for patient_i], [jpgs for patient_j] ], …]

  These will only include panels where both the .txt and .jpg exist
  and whose caption IDs survived filtering.

Usage:
  # with defaults baked in:
  python pair_txt_jpg.py

  # if your data lives elsewhere:
  python pair_txt_jpg.py \
    --data-path /path/to/data2 \
    --jsons-path /path/to/output_dir
"""

import os
import glob
import json
import argparse
import pandas as pd

def creating_txt_jpg_pairs(matching_uids, file_path, patient_uid_to_captions):
    # Each list consists of pairs containing the aligned files for each patient_uid
    pairs_text_list = []
    pairs_image_list = []
    not_found_counter = 0

    # Iterating over the pairs
    for pair in matching_uids:
        uids = [pair[0][:-2], pair[1][:-2]]
    
        pair_aligned_txt_files = []
        pair_aligned_jpg_files = []
    
        # Iterating over each patient_uid within the pair
        for uid in uids:
            # Construct the full path using the base_path parameter
            path = os.path.join(file_path, f'PMC{uid}/{uid}_1')
    
            # Check if the path exists
            if not os.path.exists(path):
                # print(f"Directory {path} does not exist.")
                not_found_counter += 1
                continue
    
            # Get all txt and jpg files within a single patient_uid
            txt_files = glob.glob(os.path.join(path, '*.txt'))
            jpg_files = glob.glob(os.path.join(path, '*.jpg'))

            # Get valid caption IDs for this patient_uid
            valid_caption_ids = patient_uid_to_captions.get(uid+'-1', set())

            # Create mappings: '5603015_1_1' -> '/path/to/5603015_1_1.txt'
            txt_files_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in txt_files}
            jpg_files_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in jpg_files}

            # Find common keys that are in the valid captions for this uid
            common_keys = set(txt_files_dict.keys()) & set(jpg_files_dict.keys()) & valid_caption_ids

            # Sort the paths to align txt and jpg files
            sorted_keys = sorted(common_keys)

            # Align txt and jpg files (within the same patient_uid)
            aligned_txt_files = [txt_files_dict[key] for key in sorted_keys]
            aligned_jpg_files = [jpg_files_dict[key] for key in sorted_keys]

            # Append the aligned files for this patient_uid to the pair's list
            pair_aligned_txt_files.append(aligned_txt_files)
            pair_aligned_jpg_files.append(aligned_jpg_files)
    
        # After processing both patient_uids in the pair, append the pair to the main list
        if len(pair_aligned_txt_files) == 2:
            pairs_text_list.append(pair_aligned_txt_files)
            pairs_image_list.append(pair_aligned_jpg_files)
    
    return pairs_text_list, pairs_image_list, not_found_counter


DATA_PATH = '/cs/labs/tomhope/dhtandguy21/restore_yuvalbus/data2'
JSONS_PATH = '/cs/labs/tomhope/dhtandguy21/largeListsGuy'

def main(data_path, jsons_path):

    # Upload the matching_uids list
    matching_uids_path = os.path.join(jsons_path, 'matching_uids.json')
    with open(matching_uids_path, 'r') as f:
        matching_uids = json.load(f)

    # Upload the filtered df
    filtered_csv_path = os.path.join(jsons_path, 'filtered_df.csv')
    filtered_df = pd.read_csv(filtered_csv_path)

    # Create the mapping from 'patient_uid' to a set of 'caption_id's
    patient_uid_to_captions = filtered_df.groupby('patient_uid')['caption_id'].apply(set).to_dict()

    # Build the two 3D lists
    pairs_text_list, pairs_image_list, missing = \
        creating_txt_jpg_pairs(matching_uids, data_path, patient_uid_to_captions)

    # Save the pairs to JSON files
    with open(os.path.join(jsons_path, 'pairs_text_list.json'), 'w') as f:
        json.dump(pairs_text_list, f)

    with open(os.path.join(jsons_path, 'pairs_image_list.json'), 'w') as f:
        json.dump(pairs_image_list, f)
    
    print(f"Done. {missing} patient‐folders not found. "
          f"Wrote {len(pairs_text_list)} pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create aligned .txt/.jpg panel pairs for each UID pair.")
    
    parser.add_argument(
        "--data_path",
        "-d",                # -d is a short option for data paths
        default=DATA_PATH,
        help="root directory where PMC<id>/<id>_1 folders live (default: %(default)s")
    
    parser.add_argument(
        "--jsons_path",
        "-j",                # -j is a short option for JSONs paths
        default=JSONS_PATH,
        help="where matching_uids.json & filtered_df.csv reside (and outputs go) (default: %(default)s)")
    
    args = parser.parse_args()

    # Run main()
    main(args.data_path, args.jsons_path)