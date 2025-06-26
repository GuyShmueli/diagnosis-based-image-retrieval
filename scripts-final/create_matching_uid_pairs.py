"""
Objective:
  Read a filtered DataFrame of PMC “patient” captions, then
  produce a JSON file `matching_uids.json` containing
  list-of-lists of positive UID pairs, e.g.
    [
      ["1111111-1", "2222222-1"],
      ["3333333-1", "4444444-1"],
      …
    ]
  Pairs are deduped and symmetric (i.e. [A,B] only once).

Usage:
   Default:
  python make_matching_uids.py

   If you’ve saved your filtered CSV elsewhere:
  python make_matching_uids.py \
      --input-dir /path/to/filtered_csv_dir \
      --output-dir /path/to/output_json_dir
"""

import os
import json
import argparse
import pandas as pd

def creating_uids_pairs(df, rows_num):
    matching_uids = []
    checked_pairs = set()
    # Create a set of patient_uids present in the filtered DataFrame
    uids_in_df = set(df['patient_uid'])
    
    for index, row in df.head(rows_num).iterrows():
        patient_uid = row['patient_uid']
        
        # Ensure that patient_uid is in uids_in_df
        if patient_uid not in uids_in_df:
            continue  # This should not happen, but added for safety

        sim_patients = row['unique_articles_sim_patients']
    
        # Skip NaN values
        if pd.isna(sim_patients):
            continue
    
        # Clean the 'unique_articles_sim_patients' column
        sim_patients = sim_patients.strip("[]").replace("'", "")
        sim_patients_list = sim_patients.split(', ') if sim_patients else []
    
        for similar_uid in sim_patients_list:
            # Skip self-matching
            if similar_uid == patient_uid:
                continue

            # Check if similar_uid is in the filtered DataFrame
            if similar_uid not in uids_in_df:
                continue  # Skip if similar_uid not in the DataFrame

            # Create a sorted tuple to handle symmetry
            pair = tuple(sorted([patient_uid, similar_uid]))
    
            # Check if the pair was already added
            if pair not in checked_pairs:
                matching_uids.append([patient_uid, similar_uid])  # Add the pair
                checked_pairs.add(pair)  # Mark this pair as checked

    return matching_uids


JSONS_PATH = '/cs/labs/tomhope/dhtandguy21/largeListsGuy'

def main(input_dir, output_dir):
    # 1) load the filtered CSV
    csv_path = os.path.join(input_dir, 'filtered_df.csv')
    df = pd.read_csv(csv_path)

    # 2) compute all UID pairs
    pairs = creating_uids_pairs(df, len(df))

    # 3) ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 4) write JSON
    out_path = os.path.join(output_dir, 'matching_uids.json')
    with open(out_path, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"Wrote {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate matching_uids.json (positive UID pairs) "
                    "from filtered_df.csv."
    )
    p.add_argument(
        "--input-dir", "-i",
        default=JSONS_PATH,
        help="directory containing filtered_df.csv (default: %(default)s)"
    )
    p.add_argument(
        "--output-dir", "-o",
        default=JSONS_PATH,
        help="where to write matching_uids.json (default: %(default)s)"
    )
    args = p.parse_args()

    main(args.input_dir, args.output_dir)