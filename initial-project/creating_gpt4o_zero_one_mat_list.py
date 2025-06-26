"""
This file creates a list of matrices, such that:
1. Each matrix corresponds to a pair of similar PATIENTS
2. each matrix's element is 1 if there is a match between the responses of the corresponding pair's captions,
   and 0 otherwise
"""
import torch
import json
import pickle
import pandas as pd
import numpy as np

def sim_gpt_responses_within_pair(filtered_df, uid_pair):
    first_uid, second_uid = uid_pair
    first_uid_responses = filtered_df[filtered_df["patient_uid"] == first_uid]["gpt4o_response"].to_list()
    second_uid_responses = filtered_df[filtered_df["patient_uid"] == second_uid]["gpt4o_response"].to_list()

    mat = np.zeros((len(first_uid_responses), len(second_uid_responses)))
    for i, first_res in enumerate(first_uid_responses):
        for j, second_res in enumerate(second_uid_responses):
            if first_res == second_res and first_res != "nan":
                mat[i][j] = 1

    return torch.tensor(mat, device=0)

# Uploading the matching_uids list
lists_path = '/cs/labs/tomhope/yuvalbus/pmc/pythonProject/largeListsGuy/'
with open(lists_path + 'matching_uids.json', 'r') as f:
    matching_uids = json.load(f)

# Uploading the filtered df
filtered_csv_path = '/cs/labs/tomhope/yuvalbus/pmc/pythonProject/pythonFilesGuy/filtered_df.csv'
filtered_df = pd.read_csv(filtered_csv_path)

gpt_responses_mat_list = [sim_gpt_responses_within_pair(filtered_df, uid_pair) for uid_pair in matching_uids]

with open(lists_path + "gpt4o_zero_one_mat_list.pkl", "wb") as g:
    pickle.dump(gpt_responses_mat_list, g)
