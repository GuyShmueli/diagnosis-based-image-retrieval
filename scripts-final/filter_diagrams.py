"""
Objective:
  Scan a list of image paths, and keep only those that likely represent
  diagram panels by applying two filters:
    1. A “white‐pixel” ratio: if >50% of pixels are above the brightness
       threshold, we assume it’s a diagram (white background).
    2. An OCR‐based text length: if the image has >100 characters of text,
       we also treat it as a diagram.

  Panels that meet **either** condition are retained; all others are dropped.

Usage:
  # With defaults (reads imgs_paths.json, writes imgs_paths_diagrams_new.json):
  python filter_diagrams.py

  # If your JSONs live elsewhere:
  python filter_diagrams.py \
      --jsons-path /path/to/jsons_dir
"""

import os
import argparse
from PIL import Image
import numpy as np
import json
import pytesseract  # for OCR
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def calc_ratio_and_text_len(path, threshold=250):     
    with Image.open(path) as image:  # Ensure image is closed after use
        pixel_values = np.asarray(image)
 
        # Compute ratio early
        tot_pixels = pixel_values.size
        pixels_exceeding_threshold = (pixel_values >= threshold).sum()
        ratio = round(pixels_exceeding_threshold / tot_pixels, 2)

        # If ratio > 0.5, no need for OCR
        if ratio > 0.5:
            return ratio, 0  # Text len won't matter

        # Perform OCR while image is still open
        text = pytesseract.image_to_string(image, lang='eng')
        
    return ratio, len(text)


JSONS_PATH = '/cs/labs/tomhope/dhtandguy21/largeListsGuy'

def main(jsons_path):

    def _process_path(path):
        try:
            ratio, text_len = calc_ratio_and_text_len(path)
            condition1 = ratio > 0.5
            condition2 = text_len > 100
            return path if condition1 or condition2 else None
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None

    with open(os.path.join(jsons_path, 'imgs_paths.json'), 'r') as g:
        imgs_paths = json.load(g)

    imgs_paths_diagrams = []
    # --- Use ProcessPoolExecutor for CPU-bound tasks ---
    num_workers = max(1, multiprocessing.cpu_count() - 2)  # Leave 2 cores free for other tasks
    with ProcessPoolExecutor(max_workers=num_workers) as executor: 
        futures = [executor.submit(_process_path, path) for path in imgs_paths]

        for future in as_completed(futures):
            result = future.result()
            if result:
                imgs_paths_diagrams.append(result)

    print(f"Total images retained: {len(imgs_paths_diagrams)}")

    # Save the non-diagram paths to a new JSON file
    with open(os.path.join(jsons_path, 'imgs_paths_diagrams_new.json'), 'w') as f:
        json.dump(imgs_paths_diagrams, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter diagrams based on pixel value and OCR text length.")
    parser.add_argument(
        "--jsons_path",
        "-j",  # -j is a short option for JSONs paths
        default=JSONS_PATH,
        help="directory where the images paths JSON is stored (default: %(default)s)"
    )
    args = parser.parse_args()

    main(args.jsons_path)