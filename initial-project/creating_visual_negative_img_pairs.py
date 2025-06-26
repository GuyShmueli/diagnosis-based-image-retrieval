""" 
Here we take a different approach than the one we took while creating the positive pairs.
The reason is that taking low textual similarity thresholds won't make a difference, because even non fine-tuned
BiomedCLIP can handle these.
Taking intermediate thresholds is tempting, but many times the images are actually positive pairs even if their textual similarity is 0.5-0.7.
The solution we suggest is taking NON-SIMILAR PATIENTS, compute their VISUAL similarity and adjust the visual threshold to be fairly high.
This will not cause overfitting/leakage, because in the inference stage we have only similar patients.
"""
import os
import glob
import pickle
from open_clip import create_model_from_pretrained, get_tokenizer
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Uploading the filtered dataframe (without the compound figures)
filtered_df_path = "/cs/labs/tomhope/yuvalbus/pmc/pythonProject/pythonFilesGuy/filtered_df.csv"
filtered_df = pd.read_csv(filtered_df_path)

# Taking only patients that are not similar to any other patient
filtered_df_nan_uids = filtered_df[filtered_df["unique_articles_sim_patients"].isna()]


def creating_non_matching_uid_pairs(nan_uids_col_uid):
    non_matching_uids = set()  # Use a set to avoid duplicates

    for idx, uid in enumerate(nan_uids_col_uid):
        if idx >= len(nan_uids_col_uid) - 220:
            break

        # Generate the next UIDs
        next_uids = [
            nan_uids_col_uid[idx + offset] for offset in range(1, 80)
        ]

        # Ensure all pairs are added in a consistent order and skip identical pairs
        pairs = [
            tuple(sorted((uid, next_uid)))
            for next_uid in next_uids if uid != next_uid
        ]

        non_matching_uids.update(pairs)  # Add pairs to the set

    return list(non_matching_uids)  # Convert back to a list

# Generating all unique non-matching pairs of UIDs 
non_matching_uids = creating_non_matching_uid_pairs(filtered_df_nan_uids["patient_uid"].tolist())

def get_aligned_image_pairs(non_matching_uids, file_path):
    pairs_image_list = []

    for pair in non_matching_uids:
        uids = [pair[0][:-2], pair[1][:-2]]
        pair_aligned_jpg_files = []

        for uid in uids:
            path = os.path.join(file_path, f'PMC{uid}/{uid}_1')
            if not os.path.exists(path):
                continue

            # Sort .jpg files to maintain `_1`, `_2` order
            jpg_files = sorted(glob.glob(os.path.join(path, '*.jpg')))
            pair_aligned_jpg_files.append(jpg_files)

        if len(pair_aligned_jpg_files) == 2:
            pairs_image_list.append(pair_aligned_jpg_files)

    return pairs_image_list

data_path = '/cs/labs/tomhope/yuvalbus/pmc/pythonProject/data2'
img_non_matching_pairs = get_aligned_image_pairs(non_matching_uids, data_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

biomedclip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
biomedclip_model.to(device)
tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# Note that using list comprehension for big data is highly inefficient, hence we'll utilize parallelized computation
# First, we'll create a custom dataset for images
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Initialize sets to store unique image paths
all_image_paths = set()

for pair in img_non_matching_pairs:
    first_list = pair[0]
    second_list = pair[1]
    all_image_paths.update(first_list)
    all_image_paths.update(second_list)

# Convert set to list
all_image_paths = list(all_image_paths)

# Compute embeddings for all images
image_to_embedding = {}  # Dictionary to map image paths to embeddings

# Define the dataset and dataloader
dataset = ImageDataset(all_image_paths, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

embeddings = []

biomedclip_model.eval()

with torch.no_grad():
    for idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        emb = biomedclip_model.encode_image(batch)
        emb = F.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu())

# Concatenate all embeddings
embeddings = torch.cat(embeddings, dim=0)

# Map image paths to embeddings
for img_path, emb in zip(all_image_paths, embeddings):
    image_to_embedding[img_path] = emb

# By now, we have computed each image visual embedding
# We want to move on to calculate visual similarities
# List to hold visual similarity matrices
non_similar_uids_visual_sim_matrices = []

for pair in img_non_matching_pairs:
    first_list = pair[0]
    second_list = pair[1]

    # Retrieve embeddings for images in first_list
    embeddings1 = torch.stack([image_to_embedding[img_path] for img_path in first_list])

    # Retrieve embeddings for images in second_list
    embeddings2 = torch.stack([image_to_embedding[img_path] for img_path in second_list])

    # Compute similarity matrix
    similarities = torch.matmul(embeddings1, embeddings2.t())

    # Append the similarity matrix to the list
    non_similar_uids_visual_sim_matrices.append(similarities)

# This function creates a coordinates pair, exceeding the threshold
def create_visual_non_similar_uids_negative_pairs(vis_sim_mat):
    nonzero_indices = vis_sim_mat.nonzero(as_tuple=True)
    nonzero_values = vis_sim_mat[nonzero_indices]
    # Extracting the coordinates of the similarity scores that exceed 0.70, but don't exceed 0.80
    condition = (nonzero_values > 0.7) & (nonzero_values < 0.8)
    exceed_threshold_indices = tuple([idx[condition] for idx in nonzero_indices])  
    return exceed_threshold_indices

# visual_negative_pairs_dict is a dictionary, such that for each non-similar PATIENT pair (key)
# the value is the coordinates pair of visual similarities exceeding the predefined threshold
# For example: {"('6031211-1', '6232997-1')": (tensor([], dtype=torch.int64), tensor([], dtype=torch.int64)), ...}
visual_negative_pairs_dict = {str(uid_pair): create_visual_non_similar_uids_negative_pairs(non_similar_uids_visual_sim_matrices[idx]) for idx, uid_pair in enumerate(non_matching_uids)}

# Filtering-out empty coordinates 
visual_filtered_negative_pairs_including_indices = [(idx, pair) for idx, pair in enumerate(visual_negative_pairs_dict.values()) if all(tensor.numel() > 0 for tensor in pair)]
visual_filtered_negative_pairs = [pair for pair in visual_negative_pairs_dict.values() if all(tensor.numel() > 0 for tensor in pair)]

visual_neg_indices_list = [idx_neg_pair[0] for idx_neg_pair in visual_filtered_negative_pairs_including_indices]
visual_negative_pairs_image_list = [img_non_matching_pairs[idx] for idx in visual_neg_indices_list]
visual_img_negative_pairs = [(visual_negative_pairs_image_list[imgs_pair_idx][0][fir_img_idx], visual_negative_pairs_image_list[imgs_pair_idx][1][sec_img_idx]) for (imgs_pair_idx, imgs_pair) in enumerate(visual_filtered_negative_pairs) for (fir_img_idx, sec_img_idx) in zip(imgs_pair[0], imgs_pair[1])]
visual_labeled_img_negative_pairs = [(visual_img_negative_pairs[idx], 0) for idx in range(len(visual_img_negative_pairs))]

files_paths = "/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/"
with open(files_paths + "visual_labeled_img_negative_pairs.pkl", "wb") as f:
    pickle.dump(visual_labeled_img_negative_pairs, f)