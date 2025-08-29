import os
from mmsdk.mmdatasdk import mmdataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Point to the same MOSEI dataset you already downloaded
DATA_ROOT = "data/raw/mosei"

CSD_FILES = {
    "words": os.path.join(DATA_ROOT, "CMU_MOSEI_TimestampedWords.csd"),
    "phones": os.path.join(DATA_ROOT, "CMU_MOSEI_TimestampedPhones.csd"),
    "glove_vectors": os.path.join(DATA_ROOT, "CMU_MOSEI_TimestampedWordVectors.csd"),
    "All Labels": os.path.join(DATA_ROOT, "CMU_MOSEI_Labels.csd")
}


class MOSEIDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_mosei_dataloader(batch_size=16, shuffle=True):
    dataset_obj = mmdataset(CSD_FILES)
    glove_seq = dataset_obj["glove_vectors"]
    labels_seq = dataset_obj["All Labels"]

    common_utt_ids = set(glove_seq.data.keys()).intersection(set(labels_seq.data.keys()))
    all_X = []
    all_y = []

    for utt_id in common_utt_ids:
        X = glove_seq.data[utt_id]["features"][:].astype('float32')
        y = labels_seq.data[utt_id]["features"][:].astype('float32')

        # Truncate or pad y to match X length
        if y.shape[0] != X.shape[0]:
            min_len = min(X.shape[0], y.shape[0])
            X = X[:min_len]
            y = y[:min_len]

        all_X.append(torch.tensor(X))
        all_y.append(torch.tensor(y))
        

    X_padded = pad_sequence(all_X, batch_first=True)
    y_padded = pad_sequence(all_y, batch_first=True)

    
    
    dataset = MOSEIDataset(X_padded, y_padded)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

