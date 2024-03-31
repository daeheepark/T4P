from typing import Dict
from glob import glob
import os.path as osp

import torch
from torch.utils.data import Dataset

class TmpWaymo_01(Dataset):
    def __init__(self, preprocessed_dir) -> None:
        super().__init__()

        print("Assessing Waymo 0.1 preprocessed files...")
        self._processed_elements = glob(osp.join(preprocessed_dir, '*.pt'))
        print(f"Done ({len(self._processed_elements)} files) ..")

        self._data_len: int = len(self._processed_elements)

    def __len__(self) -> int:
        return self._data_len
    
    def __getitem__(self, idx: int) -> Dict:
        data = torch.load(self._processed_elements[idx])
        return data