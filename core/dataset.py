from collections import namedtuple
import enum
from typing import OrderedDict
import numpy as np
from pydantic import NoneIsAllowedError
from torch.utils.data import Dataset
import pandas as pd
import torch
from core.time_series import standard_scaler, stride_series, min_max_scaler
class DataTypes(enum.IntEnum):
    """Defines numerical types of each culumn."""
    CONTINUOUS = 0
    CATEGORICAL = 1
    DATE = 2
    STR = 3

class InputTypes(enum.IntEnum):
    """Defines input types of each column."""
    TARGET = 0
    OBSERVED = 1
    KNOWN = 2
    STATIC = 3
    ID = 4 # used as an only identifier
    TIME = 5 # used as as a time index

FeatureSpec = namedtuple('FeatureSpec', ['name', 'feature_type', 'feature_embed_type'])
DTYPE_MAP = {
    DataTypes.CONTINUOUS : np.float32,
    DataTypes.CATEGORICAL : np.int64,
    DataTypes.DATE : 'datetime64[ns]',
    DataTypes.STR : str
}

FEAT_ORDER = [
    (InputTypes.STATIC, DataTypes.CATEGORICAL),
    (InputTypes.STATIC, DataTypes.CONTINUOUS),
    (InputTypes.KNOWN, DataTypes.CATEGORICAL),
    (InputTypes.KNOWN, DataTypes.CONTINUOUS),
    (InputTypes.OBSERVED, DataTypes.CATEGORICAL),
    (InputTypes.OBSERVED, DataTypes.CONTINUOUS),
    (InputTypes.TARGET, DataTypes.CONTINUOUS),
    (InputTypes.ID, DataTypes.CATEGORICAL)
]

FEAT_NAMES = ['s_cat' , 's_cont' , 'k_cat' , 'k_cont' , 'o_cat' , 'o_cont' , 'target']
# DEFAULT_ID_COL = 'id'

class GHLDataset(Dataset):
    def __init__(self, path, config, train=True, dataset=None):
        super().__init__()
        self.features = config.features
        raw_data = pd.read_csv(path, index_col=False)
        self.data = raw_data.copy()
        self.example_length = config.example_length
        self.stride = config.dataset_stride

        col_dtypes = {v.name:DTYPE_MAP[v.feature_embed_type] for v in self.features}
        # print(col_dtypes)
        self.data = self.data[set(x.name for x in self.features)]
        self.data = self.data.astype(col_dtypes)
        # print(self.data[0:3])

        mu_sigma_values = []
        self.data_scaled, self.mu_sigma_values = standard_scaler(self.data, mu_sigma_values)

        # min_max_values = []
        # self.data_scaled, self.mu_sigma_values = min_max_scaler(self.data, min_max_values)

        if train is True:
            self.series = stride_series(self.data_scaled, self.example_length, self.stride)
            self.labels = None
            
        else:
            self.series, self.labels = stride_series(self.data_scaled, self.example_length, self.stride, train=train, raw_data=raw_data, dataset=dataset)
                
        # print(len(raw_data))
        # print(len(self.labels))
    def __len__(self):
        return len(self.series)

    def __getitem__(self, index):
        series = self.series[index]
        
        tensors = tuple([] for _ in range(7))
        for v in self.features:
            # print(v.name)
            if v.feature_type == InputTypes.STATIC and v.feature_embed_type == DataTypes.CATEGORICAL:
                tensors[0].append(torch.from_numpy(series[v.name].to_numpy()))
            elif v.feature_type == InputTypes.STATIC and v.feature_embed_type == DataTypes.CONTINUOUS:
                tensors[1].append(torch.from_numpy(series[v.name].to_numpy()))
            elif v.feature_type == InputTypes.KNOWN and v.feature_embed_type == DataTypes.CATEGORICAL:
                tensors[2].append(torch.from_numpy(series[v.name].to_numpy()))
            elif v.feature_type == InputTypes.KNOWN and v.feature_embed_type == DataTypes.CONTINUOUS:
                tensors[3].append(torch.from_numpy(series[v.name].to_numpy()))
            elif v.feature_type == InputTypes.OBSERVED and v.feature_embed_type == DataTypes.CATEGORICAL:
                tensors[4].append(torch.from_numpy(series[v.name].to_numpy()))
            elif v.feature_type == InputTypes.OBSERVED and v.feature_embed_type == DataTypes.CONTINUOUS:
                tensors[5].append(torch.from_numpy(series[v.name].to_numpy()))
            elif v.feature_type == InputTypes.TARGET:
                tensors[6].append(torch.from_numpy(series[v.name].to_numpy()))
                # print(tensors[6].shape)
        tensors = [torch.stack(x, dim=-1) if x else torch.empty(0) for x in tensors]
        # print("Tensor 6 : {}".format(tensors[6].shape))
        if self.labels is None:
            return OrderedDict(zip(FEAT_NAMES, tensors))
        else:
            labels = self.labels[index]
            return OrderedDict(zip(FEAT_NAMES, tensors)), labels
            