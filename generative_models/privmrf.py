"""
A generative model training algorithm based on
"Data Synthesis via Differentially Private Markov Random Fields"
by K. Cai, X. Lei, J. Wei, X. Xiao, published in Proceedings of the VLDB Endowment (PVLDB), 2021
Adapted from: https://github.com/vrtoddy/tab_bench
original repository: https://github.com/caicre/PrivMRF
"""

import numpy as np
from pandas import DataFrame

from generative_models.generative_model import GenerativeModel
from method.AIM.cdp2adp import cdp_rho
from method.PrivMRF.PrivMRF.domain import Domain
from method.PrivMRF.PrivMRF.main import run as run_privmrf
from utils.constants import CATEGORICAL, ORDINAL
from utils.device_utils import validate_and_get_device
from utils.logging import LOGGER


class _PrivMRFPreprocessor:
    def __init__(self):
        self.synthetic_df = None

    def reverse_data(self, df, path=None):
        self.synthetic_df = df.copy()
        return self.synthetic_df


class PrivMRF(GenerativeModel):
    """A wrapper for the PrivMRF synthetic data mechanism."""

    def __init__(
        self,
        metadata=None,
        epsilon=1.0,
        delta=1e-9,
        theta=6,
        max_measure_attr_num=6,
        estimation_iter_num=3000,
        multiprocess=False,
        device=None,
    ):
        self.metadata = metadata
        self.epsilon = epsilon
        self.delta = delta
        self.theta = theta
        self.max_measure_attr_num = max_measure_attr_num
        self.estimation_iter_num = estimation_iter_num
        self.multiprocess = bool(multiprocess)
        self.device, self.is_gpu = validate_and_get_device(device)

        self.datatype = DataFrame
        self.mechanism = None
        self.trained = False
        self.__name__ = 'PrivMRF'

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

        config = {
            'print': False,
            'theta': self.theta,
            'max_measure_attr_num': self.max_measure_attr_num,
            'estimation_iter_num': self.estimation_iter_num,
            'gpu': bool(self.is_gpu),
        }
        self.mechanism = run_privmrf(
            encoded_data.to_numpy(dtype=int),
            Domain(self._build_json_domain(domain), list(range(encoded_data.shape[1]))),
            attr_hierarchy=None,
            exp_name='wrapper',
            rho=rho,
            p_config=config,
        )

        self.trained = True

    def generate_samples(self, nsamples):
        assert self.trained, 'Model must first be fitted to some data.'
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        preprocessor = _PrivMRFPreprocessor()
        self.mechanism.syn(nsamples, preprocessor, path=None)

        synthetic_data = preprocessor.synthetic_df
        if synthetic_data is None:
            synthetic_data = getattr(self.mechanism, 'df', None)
        if synthetic_data is None:
            raise RuntimeError('PrivMRF did not produce synthetic data.')

        synthetic_data = synthetic_data.copy()
        synthetic_data.columns = self._column_names
        decoded_data = self._decode_data(synthetic_data)

        return decoded_data

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
        return {idx: int(data.iloc[:, idx].nunique()) for idx in range(data.shape[1])}

    def _build_json_domain(self, domain):
        return {idx: {'type': 'discrete', 'domain': size} for idx, size in domain.items()}
