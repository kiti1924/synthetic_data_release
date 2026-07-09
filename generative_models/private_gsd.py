"""
A generative model training algorithm based on
"Generating Private Synthetic Data with Genetic Algorithms"
by T. Liu, J. Tang, G. Vietri, Z. S. Wu, published in International Conference on Machine Learning (ICML), 2023
Adapted from: https://github.com/vrtoddy/tab_bench
original repository: https://github.com/giusevtr/private_gsd
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from generative_models.generative_model import GenerativeModel
from utils.constants import CATEGORICAL, ORDINAL
from utils.logging import LOGGER

class PrivateGSD(GenerativeModel):
    """A wrapper for the Private-GSD synthetic data mechanism."""

    def __init__(self, metadata=None, epsilon=1.0, delta=1e-9, epochs=100, batch_size=500):
        self.metadata = metadata
        self.epsilon = epsilon
        self.delta = delta
        self.epochs = epochs
        self.batch_size = batch_size

        self.datatype = DataFrame
        self.mechanism = None
        self.trained = False
        self.__name__ = 'PrivateGSD'

        self._reverse_maps = {}

    def fit(self, data):
        assert isinstance(data, self.datatype), (
            f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        )

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
        
        data = data.reset_index(drop=True)
        encoded_data = self._encode_data(data)

        from jax.random import PRNGKey
        from method.private_gsd.models import GSD as PrivateGSDMechanism
        from method.private_gsd.utils.utils_data import Dataset, Domain
        from method.private_gsd.stats import Marginals, ChainedStatistics
        from method.private_gsd.utils.cdp2adp import cdp_rho

        domain_dict = {col: len(self._reverse_maps[col]) for col in encoded_data.columns}
        domain = Domain.fromdict(domain_dict)
        dataset = Dataset(encoded_data, domain)

        marginal_module2 = Marginals.get_all_kway_combinations(dataset.domain, k=2, bins=[2, 4, 8, 16, 32])
        stat_module = ChainedStatistics([marginal_module2])
        stat_module.fit(dataset)

        class DummyArgs:
            pass

        self.mechanism = PrivateGSDMechanism(
            domain=dataset.domain,
            print_progress=False,
            stop_early=True,
            population_size_muta=50,
            population_size_cross=50,
            num_encoder=None,
            num_idx=[],
            args=DummyArgs()
        )
        
        rho = cdp_rho(self.epsilon, self.delta)
        key = PRNGKey(0)
        self.mechanism.zcdp_syn_init(key, stat_module, rho)

        self.trained = True

    def generate_samples(self, nsamples):
        assert self.trained, 'Model must first be fitted to some data.'
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        synthetic_data = None
        if hasattr(self.mechanism, 'generate'):
            synthetic_data = self.mechanism.generate(nsamples)
        elif hasattr(self.mechanism, 'syn'):
            class DummyPreprocesser:
                def __init__(self):
                    self.synthetic_df = None

                def reverse_data(self, df, path):
                    self.synthetic_df = df.copy()

            preprocesser = DummyPreprocesser()
            result = self.mechanism.syn(nsamples, preprocesser)
            synthetic_data = preprocesser.synthetic_df if preprocesser.synthetic_df is not None else result
        elif hasattr(self.mechanism, 'sample'):
            synthetic_data = self.mechanism.sample(nsamples)
        else:
            raise NotImplementedError("The underlying mechanism does not have a known generate/sample method.")

        if not isinstance(synthetic_data, DataFrame):
            column_names = [column['name'] for column in self.metadata['columns']] if self.metadata else None
            synthetic_data = DataFrame(synthetic_data, columns=column_names)

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
                decoded[column] = pd.to_numeric(data[column], errors='coerce').round().astype(int).map(self._reverse_maps[column])
            else:
                decoded[column] = data[column]
        return decoded
