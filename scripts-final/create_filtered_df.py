"""
Filter out compound‐figure captions.

This script’s objective is to remove any caption that describes more than one sub-image,
because our downstream encoder struggles on multi-panel figures.

Usage:
    python filter_captions.py

If you need custom paths:
    python filter_captions.py --data-path /path/to/data2 \
                              --jsons-path /path/to/output
"""

import os
import pandas as pd
import re
import json
import argparse
from concurrent.futures import ThreadPoolExecutor


def upload_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    df = pd.read_csv(file_path)
    return df


def read_caption(file_path):
    """Read the full text of a caption file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def read_captions_parallel(paths, max_workers=8):
    """Read multiple caption files in parallel and return their contents."""
    captions = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(read_caption, path): idx for idx, path in enumerate(paths)}
        for future in future_to_index:
            idx = future_to_index[future]
            try:
                captions[idx] = future.result()
            except Exception as e:
                print(f"Error reading caption at index {idx}: {e}")
                captions[idx] = ""
    return captions


def is_containing_specified_patterns(caption, patterns):
    """Determine if a caption matches any of the given regex patterns."""
    for pattern in patterns:
        if re.search(pattern, caption):
            return True
    return False


def filter_out_using_regex(df, patterns):
    """Filter out rows whose captions match given patterns and save the result."""
    # Applying the function to the 'caption_text' column
    df['contains_specified_patterns'] = df['caption_text'].apply(lambda caption 
                                                                    : is_containing_specified_patterns(caption, patterns))
    # Creating a new DataFrame that excludes rows with specified patterns
    df_filtered = df[~df['contains_specified_patterns']].reset_index(drop=True)

    # Dropping the 'contains_specified_patterns' column
    df_filtered = df_filtered.drop('contains_specified_patterns', axis='columns')
    
    # Add 'caption_id' of such form "8167975_1_1, 8167975_1_2" to each caption
    # This will help us later in creating a dictionary mapping each patient_uid to all of its captions
    df_filtered['caption_id'] = df_filtered['caption_path'].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0])

    return df_filtered


patterns = [
    r'\(B\)',    # Matches (B)
    r'\(b\)',    # Matches (b)
    r'B\)',      # Matches B)
    r'b\)',      # Matches b)
    r'B\.',      # Matches B.
    r'b\.',      # Matches b.
    r'b\s[A-Z]',   # Matches 'b' followed by a space and a capital letter
    r'B\s[A-Z]',   # Matches 'B' followed by a space and a capital letter
]

DATA_PATH = '/cs/labs/tomhope/dhtandguy21/restore_yuvalbus/data2'
JSONS_PATH = '/cs/labs/tomhope/dhtandguy21/largeListsGuy'

def main(data_path, jsons_path):
    """Load, filter and save caption data for compound‐figure removal."""
    # Uploading the csv
    df = upload_csv(os.path.join(data_path, 'final_csv2.csv'))

    # Reading captions into 'caption_text' column
    txt_paths = [os.path.join(data_path, elem) for elem in df['caption_path'].tolist()]
    df['caption_text'] = read_captions_parallel(txt_paths)

    # Specifying the path where the filtered df will be saved
    filtered_df = filter_out_using_regex(df, patterns)

    # Saving the filtered DataFrame as a CSV file
    filtered_csv_path = os.path.join(jsons_path,'filtered_df.csv')
    filtered_df.to_csv(filtered_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove any caption that describes more than one sub-image.")
    
    parser.add_argument(
        "--data_path",
        "-d",                # -d is a short option for data paths
        default=DATA_PATH,
        help="root directory where original csv is stored (default: %(default)s)")
    
    parser.add_argument(
        "--jsons_path",
        "-j",                # -j is a short option for JSONs paths
        default=JSONS_PATH,
        help="directory where the filtered csv is stored (default: %(default)s)")
    
    args = parser.parse_args()

    # Run main()
    main(args.data_path, args.jsons_path)
