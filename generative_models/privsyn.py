"""
A generative model training algorithm based on
"PrivSyn: Differentially Private Data Synthesis"
by Z. Zhang, T. Wang, N. Li, J. Honorio, M. Backes, S. He, J. Chen, Y. Zhang, published in 30th USENIX Security Symposium, 2021
Adapted from: https://github.com/vrtoddy/tab_bench
original repository: https://github.com/ruizhang-p/PrivSyn
"""

import argparse

import numpy as np
from pandas import DataFrame

from generative_models.generative_model import GenerativeModel
from method.privsyn.run_privsyn import privsyn_main
from utils.constants import CATEGORICAL, ORDINAL
from utils.device_utils import validate_and_get_device
from utils.logging import LOGGER

from method.AIM.cdp2adp import cdp_rho


class _PrivSynPreprocessor:
    def __init__(self):
        self.synthetic_df = None

    def reverse_data(self, df, path=None):
        self.synthetic_df = df.copy()
        return self.synthetic_df


class PrivSyn(GenerativeModel):
    """A wrapper for the PrivSyn synthetic data mechanism."""

    def __init__(self, metadata=None, epsilon=1.0, delta=1e-9, device=None):
        self.metadata = metadata
        self.epsilon = epsilon
        self.delta = delta
        self.device_str, self.is_gpu = validate_and_get_device(device)

        self.datatype = DataFrame
        self.generator = None
        self.trained = False
        self.__name__ = 'PrivSyn'

        self._reverse_maps = {}
        self._column_names = []

    def fit(self, data):
        assert isinstance(data, self.datatype), (
            f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        )

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')

        data = data.reset_index(drop=True)
        self._column_names = list(data.columns)
        encoded_data = self._encode_data(data)
        domain = self._build_domain(encoded_data)
        rho = cdp_rho(self.epsilon, self.delta)

        args = argparse.Namespace(
            dataset='adult',
            epsilon=self.epsilon,
            delta=self.delta,
            device=self.device_str,
        )
        self.generator = privsyn_main(args, encoded_data, domain, rho)['privsyn_generator']
        self.trained = True

    def generate_samples(self, nsamples):
        assert self.trained, 'Model must first be fitted to some data.'
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        preprocessor = _PrivSynPreprocessor()
        self.generator.syn(nsamples, preprocessor, parent_dir=None)

        synthetic_data = preprocessor.synthetic_df
        if synthetic_data is None:
            synthetic_data = getattr(self.generator, 'synthesized_df', None)
        if synthetic_data is None:
            raise RuntimeError('PrivSyn did not produce synthetic data.')

        synthetic_data = synthetic_data.copy()
        synthetic_data.columns = self._column_names
        return self._decode_data(synthetic_data)

    def _encode_data(self, data):
        encoded = DataFrame(index=data.index)
        self._reverse_maps = {}

        for column in data.columns:
            series = data[column]
            column_meta = next((c for c in self.metadata['columns'] if c['name'] == column), None) if self.metadata else None

            if column_meta and column_meta['type'] in [CATEGORICAL, ORDINAL]:
                mapping = {value: idx for idx, value in enumerate(column_meta['i2s'])}
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
                decoded[column] = data[column].round().astype(int).map(self._reverse_maps[column])
            else:
                decoded[column] = data[column]
        return decoded

    def _build_domain(self, data):
        domain = {}
        for column in data.columns:
            column_meta = next((c for c in self.metadata['columns'] if c['name'] == column), None) if self.metadata else None
            if column_meta and column_meta['type'] in [CATEGORICAL, ORDINAL]:
                domain[column] = len(column_meta['i2s'])
            else:
                domain[column] = int(data[column].nunique())
        return domain
