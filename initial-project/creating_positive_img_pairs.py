import os
import re
import pandas as pd
import numpy as np
import json
import pickle
import torch
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Uploading gpt4o responses
files_paths = "/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/"
with open(files_paths + "gpt4o_response_list.json", 'r') as f:
    gpt4o_response_list = json.load(f)

# gpt4o_response_list is a 1D list of length 131,155, of the following form:
# ['Modality: Clinical Photograph  \nAnatomy: Eyes',
# 'Modality: Ultrasound  \nAnatomy: Eyes', 
#  'Modality: Other/None/Unknown  \nAnatomy: Oral Anatomy, ...]

# In order to improve the automatic labeling, we want to exclude "Modality" or "Anatomy" which are "Other/None/Unknown"
# We also want to ascribe indices (in order to maintain ordering after the exclusion)
gpt4o_dict = {idx: response for idx, response in enumerate(gpt4o_response_list) if not "Other/None/Unknown" in response}

# gpt4o_dict is a dictionary of length 81,617, of the following form:
# {0: 'Modality: Clinical Photograph  \nAnatomy: Eyes',
#  1: 'Modality: Ultrasound  \nAnatomy: Eyes',
#  16: 'Modality: Ultrasound  \nAnatomy: Genitourinary System', ...}

# Uploading the filtered dataframe (without the compound figures)
filtered_df_path = "/cs/labs/tomhope/yuvalbus/pmc/pythonProject/pythonFilesGuy/filtered_df.csv"
filtered_df = pd.read_csv(filtered_df_path)

# Adding a column of "gpt4o_response" to responses that don't include "Other/None/Unknown"
filtered_df.loc[list(gpt4o_dict.keys()), "gpt4o_response"] = list(gpt4o_dict.values())

# Excluding white spaces
def normalize(text):
    return re.sub(r'\s+', ' ', text.strip())    
filtered_df["gpt4o_response"] = filtered_df["gpt4o_response"].apply(normalize)

# For each PAIR of similar PATIENTS (not images), similarity_matrices_text is a dictionary
# of patients pairs as keys and the textual similarity matrix as value, for example:
# 'Pair_1 (8167975-1, 5563556-1)': tensor([[0.7489, 0.7291, 0.6919], ..., [0.7280, 0.7488, 0.7403]], device='cuda:0')
with open(files_paths + "dict_of_text_sim_matrices.pkl", "rb") as b:
    similarity_matrices_text_dict = pickle.load(b)

# Extracting the values and keys as separate lists
similarity_matrices_text_values = list(similarity_matrices_text_dict.values())
similarity_matrices_text_keys = list(similarity_matrices_text_dict.keys())

# gpt_zero_one_mat_list is a list of matrices, each containing zeros and ones
with open(files_paths + "gpt_zero_one_mat_list.pkl", "rb") as b:
    gpt_zero_one_mat_list = pickle.load(b)

# positive_caption_pairs is required to automatically label positive image pairs
def positive_caption_pairs(gpt_responses_mat, sim_mat):
    # Performing element-wise multiplication between gpt_responses_mat
    # and sim_mat, and by that excluding different modalities/anatomies
    reponse_sim_combined_mat = gpt_responses_mat * sim_mat

    # Get the indices of non-0's in a coordinates format:
    # ( [nonzero_row_index_1, nonzero_row_index_2, ...], [nonzero_col_index_1, nonzero_col_index_2, ...] )
    nonzero_indices = reponse_sim_combined_mat.nonzero(as_tuple=True)

    # Extracting the nonzero values in the relevant coordinates in the format:
    # [a,  b,  c, ...]
    nonzero_values = reponse_sim_combined_mat[nonzero_indices]

    # Extracting the coordinates of the similarity scores that exceed 0.90
    condition = nonzero_values > 0.90
    exceed_threshold_indices = tuple([idx[condition] for idx in nonzero_indices])
    
    return exceed_threshold_indices

# Uploading the matching_uids list
lists_path = '/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/'
with open(lists_path + 'matching_uids.json', 'r') as f:
    matching_uids = json.load(f)

# positive_pairs_dict is a dictionary, such that the keys are the patients pairs
# and the values are coordinates of captions that passed gpt's filter and also have text_sim > 0.90
# If no such captions exist for that pair, we'll have:
# (tensor([], device='cuda:0', dtype=torch.int64),tensor([], device='cuda:0', dtype=torch.int64))
# len(positive_pairs_dict) is 31692
positive_pairs_dict = {str(uid_pair): positive_caption_pairs(gpt_zero_one_mat_list[idx], similarity_matrices_text_values[idx]) for idx, uid_pair in enumerate(matching_uids)}

# filtered_positive_pairs_including_indices is a list of both the indices and the values of positive_pairs_dict, excluding empty coordinates of the form:
# (tensor([], device='cuda:0', dtype=torch.int64),tensor([], device='cuda:0', dtype=torch.int64))
# len(filtered_positive_pairs_including_indices) is 1056, and there are 1321 positive caption pairs
filtered_positive_pairs_including_indices = [(idx, pair) for idx, pair in enumerate(positive_pairs_dict.values()) if all(tensor.numel() > 0 for tensor in pair)]
filtered_positive_pairs = [pair for pair in positive_pairs_dict.values() if all(tensor.numel() > 0 for tensor in pair)]

# pos_indices_list is a 1056-sized list, containing the indices of filtered_positive_pairs_including_indices list
# For example: [1, 70, 103, 140, 142, 236, 237, 263, 281, 446, ...]
pos_indices_list = [idx_pos_pair[0] for idx_pos_pair in filtered_positive_pairs_including_indices]

# Uploading pairs_image_list
# Recall that pairs_image_list is a 3D list of the images paths
# len(pairs_image_list) is 31692
with open(lists_path + 'pairs_image_list.json', 'r') as h:
    pairs_image_list = json.load(h)

# positive_pairs_image_list is a filtered version of pairs_image_list, extracting only indices from pos_indices_list
# By that, we filter-out PATIENTS that don't have positive IMAGE pairs - not the images themselves yet
# len(positive_pairs_image_list) is 1056
positive_pairs_image_list = [pairs_image_list[idx] for idx in pos_indices_list]

# img_positive_pairs is a list of tuples (similar to a 2D list), extracting from positive_pairs_image_list
# just the positive IMAGE pairs themselves
img_positive_pairs = [(positive_pairs_image_list[imgs_pair_idx][0][fir_img_idx], positive_pairs_image_list[imgs_pair_idx][1][sec_img_idx]) 
                                                for (imgs_pair_idx, imgs_pair) in enumerate(filtered_positive_pairs) for (fir_img_idx, sec_img_idx) in zip(imgs_pair[0], imgs_pair[1])]

# Adding a positive label (1) to the positive IMAGE pairs
labeled_img_positive_pairs = [(img_positive_pairs[idx], 1) for idx in range(len(img_positive_pairs))]