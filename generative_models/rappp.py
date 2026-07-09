"""
A generative model training algorithm based on
"Private Synthetic Data for Multitask Learning and Marginal Queries" (RAP++)
by G. Vietri, C. Archambeau, S. Aydore, W. Brown, M. Kearns, A. Roth, A. Siva, S. Tang, Z. S. Wu, published in Advances in Neural Information Processing Systems (NeurIPS), 2022
Adapted from: https://github.com/vrtoddy/tab_bench
original repository: https://github.com/amazon-science/dp-synthetic-data-rap
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from generative_models.generative_model import GenerativeModel
from utils.constants import CATEGORICAL, ORDINAL
from utils.logging import LOGGER

from method.AIM.cdp2adp import cdp_rho

class DummyEncoder:
    def inverse_transform(self, df):
        return df

class DummyArgs:
    def __init__(self):
        self.dataset = 'adult'

class RAPpp(GenerativeModel):
    """A wrapper for the RAP++ synthetic data mechanism."""

    def __init__(self, metadata=None, epsilon=1.0, delta=1e-9):
        self.metadata = metadata
        self.epsilon = epsilon
        self.delta = delta

        self.datatype = DataFrame
        self.generator = None
        self.trained = False
        self.__name__ = f'RAPppEps{self.epsilon}'

        self._reverse_maps = {}

    def fit(self, data):
        assert isinstance(data, self.datatype), (
            f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        )

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
        
        data = data.reset_index(drop=True)
        encoded_data = self._encode_data(data)

        from method.RAP.run import run_experiment
        from method.RAP.modules.marginal_queries import MarginalQueryClass
        try:
            from method.RAP.mechanisms.rap_pp import RAPpp as RAPMechanism
            from method.RAP.mechanisms.rap_pp import RAPppConfiguration
        except ImportError:
            try:
                from method.RAP.mechanisms.rap_pp import RAP_PP as RAPMechanism
                from method.RAP.mechanisms.rap_pp import RAPppConfiguration
            except ImportError:
                # Optional fallback or raise error
                raise

        domain_dict = {col: len(self._reverse_maps[col]) for col in encoded_data.columns}

        if RAPMechanism is None:
            raise ImportError(
                "Could not import RAP mechanism from method.RAP.mechanisms.rap_pp. "
                "Please verify the actual class name (e.g. RAPpp, RAP_PP, or RAPPlusPlus)."
            )

        default_args = RAPppConfiguration(
            iterations=[30],
            sigmoid_doubles=[10],
            optimizer_learning_rate=[0.003],
            top_q=1,
            get_dp_select_epochs=lambda domain: len(domain.get_cat_cols()),
            get_privacy_budget_weight=lambda domain: len(domain.get_cat_cols()),
            debug=False,
        )

        mechanism = RAPMechanism(
            args=[default_args],
            stats_module=[MarginalQueryClass(K=2, conditional=False)],
            name="RAPpp",
        )
        rho = cdp_rho(self.epsilon, self.delta)

        self.generator = run_experiment(
            mechanism=mechanism,
            df=encoded_data,
            domain=domain_dict,
            rho=rho,
            algorithm_seed=0,
            args=DummyArgs(),
            num_encoder=DummyEncoder(),
            num_idx=[]
        )

        self.trained = True

    def generate_samples(self, nsamples):
        assert self.trained, 'Model must first be fitted to some data.'
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        synthetic_data = None
        class DummyPreprocesser:
            def __init__(self):
                self.synthetic_df = None

            def reverse_data(self, df, path):
                self.synthetic_df = df.copy()

        preprocesser = DummyPreprocesser()
        result = self.generator.syn(nsamples, preprocesser=preprocesser)
        synthetic_data = preprocesser.synthetic_df if preprocesser.synthetic_df is not None else result

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
