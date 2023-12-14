import pickle
import numpy as np
import glob

def shuffle_dict(dict):
    lens = [len(dict[k]) for k in dict]
    assert max(lens) == min(lens)
    p = np.random.permutation(lens[0])
    return {k:dict[k][p] for k in dict}


file_list = glob.glob("/home/parham/projects/fhcr/team_e/Task_2/task23/dataset/downsampled/*.pkl")

output_path = 'dataset/combined/dataset_downsampled.pkl'

dataset_combined = {}

sum_lens = 0

for file_path in file_list:
    with (open(file_path, "rb")) as openfile:
        dataset = pickle.load(openfile)
    
    sum_lens+=len(dataset['rgbs'])
    
    for k in dataset:
        if k in dataset_combined:
            dataset_combined[k] = np.append(dataset_combined[k], dataset[k], axis=0)
        else:
            dataset_combined[k] = dataset[k]


dataset_combined = shuffle_dict(dataset_combined)


length = dataset_combined["rgbs"].shape[0]
train_idx = int(length*0.75)-1
val_idx = int(length*0.9)-1
ref = {"train":[0, train_idx], "val": [train_idx,val_idx], "test": [val_idx, length]}
dataset_divided = {j:{k: dataset_combined[k][ref[j][0]:ref[j][1], ...] for k in dataset_combined} for j in ref}


with (open(output_path, "wb")) as openfile3:
    pickle.dump(dataset_divided, openfile3)
