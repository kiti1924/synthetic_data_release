"""
A generative model training algorithm based on
"AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data"
by R. McKenna, B. Mullins, D. Sheldon, G. Miklau, published in Proceedings of the VLDB Endowment (PVLDB), 2022
Adapted from: https://github.com/vrtoddy/tab_bench
original repository: https://github.com/ryan112358/private-pgm
"""

from itertools import combinations

import numpy as np
import pandas as pd
from pandas import DataFrame
import torch

from utils.constants import CATEGORICAL, ORDINAL
from utils.logging import LOGGER
from utils.device_utils import get_device, validate_and_get_device

from generative_models.generative_model import GenerativeModel
from method.AIM.aim import AIM as AIMMechanism
from method.AIM.mbi.Dataset import Dataset
from method.AIM.mbi.Domain import Domain
from method.AIM.mbi.torch_factor import Factor


class AIM(GenerativeModel):
    """A wrapper for the AIM synthetic data mechanism."""

    def __init__(self, metadata=None, epsilon=1.0, delta=1e-9, degree=2, 
                 max_model_size=80, max_iters=1000, max_cells=10000, bounded=False, rounds=None, 
                 multiprocess=False, device=None,
    ):
        self.metadata = metadata
        self.epsilon = epsilon
        self.delta = delta
        self.degree = degree
        self.max_model_size = max_model_size
        self.max_iters = max_iters
        self.max_cells = max_cells
        self.bounded = bounded
        self.rounds = rounds
        self.multiprocess = bool(multiprocess)
        
        # Set device with fallback to default
        self.device, self.is_gpu = validate_and_get_device(device)

        self.datatype = DataFrame
        self.mechanism = None
        self.dataset = None
        self.trained = False
        self.__name__ = f'AIMEps{self.epsilon}'

        self._reverse_maps = {}
        
        # Set the device for torch_factor
        Factor.set_device(self.device)
        
        LOGGER.debug(f"AIM initialized with device: {self.device} (GPU: {self.is_gpu})")

    def fit(self, data):
        assert isinstance(data, self.datatype), (
            f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        )

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')

        encoded_data = self._encode_data(data)
        domain = self._build_domain(encoded_data)

        self.dataset = Dataset(encoded_data, domain)
        workload = self._build_workload(domain)

        self.mechanism = AIMMechanism(
            epsilon=self.epsilon,
            delta=self.delta,
            bounded=self.bounded,
            rounds=self.rounds,
            max_model_size=self.max_model_size,
            max_iters=self.max_iters,
        )
        self.mechanism.run(self.dataset, workload)

        self.trained = True

    def generate_samples(self, nsamples):
        assert self.trained, 'Model must first be fitted to some data.'
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        synthetic_dataset = self.mechanism.syn_data(num_synth_rows=nsamples)
        synthetic_data = synthetic_dataset.df.copy()
        synthetic_data = self._decode_data(synthetic_data)

        return synthetic_data

    def _encode_data(self, data):
        encoded = DataFrame(index=data.index)
        self._reverse_maps = {}

        for column in data.columns:
            series = data[column]
            if self.metadata is not None:
                column_meta = next((c for c in self.metadata['columns'] if c['name'] == column), None)
            else:
                column_meta = None

            if column_meta is not None and column_meta['type'] in [CATEGORICAL, ORDINAL]:
                categories = list(column_meta['i2s'])
                mapping = {value: idx for idx, value in enumerate(categories)}
                encoded[column] = series.map(mapping).astype(int)
                self._reverse_maps[column] = {idx: value for value, idx in mapping.items()}
            else:
                values, codes = np.unique(series.astype(object), return_inverse=True)
                encoded[column] = codes.astype(int)
                self._reverse_maps[column] = {idx: value for idx, value in enumerate(values)}

        return encoded

    def _decode_data(self, data):
        decoded = DataFrame(index=data.index)
        for column in data.columns:
            if column in self._reverse_maps:
                reverse_map = self._reverse_maps[column]
                decoded[column] = pd.to_numeric(data[column], errors='coerce').round().astype(int).map(reverse_map)
            else:
                decoded[column] = data[column]
        return decoded

    def _build_domain(self, data):
        return Domain(data.columns, [int(data[column].nunique()) for column in data.columns])

    def _build_workload(self, domain):
        workload = list(combinations(domain, self.degree))
        workload = [cl for cl in workload if domain.size(cl) <= self.max_cells]
        return [(cl, 1.0) for cl in workload]
