"""
A generative model training algorithm based on
"DP-MERF: Differentially Private Mean Embeddings with Random Features for Practical Privacy-Preserving Data Generation"
by F. Harder, M. Bauer, M. Park, published in Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS), 2021
Adapted from: https://github.com/vrtoddy/tab_bench
original repository: https://github.com/ParkLabML/DP-MERF
"""

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from generative_models.generative_model import GenerativeModel
from preprocess_common.preprocess import discretizer, rare_merger
from utils.constants import CATEGORICAL, ORDINAL, FLOAT, INTEGER
from utils.logging import LOGGER
from utils.device_utils import validate_and_get_device
from method.AIM.cdp2adp import cdp_rho


class _DP_MERFPreprocessor:
    def __init__(self, num_preprocess, rare_threshold, rho):
        self.num_preprocess = num_preprocess
        self.rare_threshold = rare_threshold
        self.rho = rho

        self.num_encoder = None
        self.cat_encoder = None
        self.num_col = 0
        self.cat_col = 0

    def fit_transform(self, X_num, X_cat):
        if X_num is not None:
            if self.num_preprocess != 'none':
                self.num_encoder = discretizer(self.num_preprocess, self.rho, ord=True)
                X_num = self.num_encoder.fit_transform(X_num)
            else:
                self.num_encoder = MinMaxScaler(feature_range=(0, 1))
                X_num = self.num_encoder.fit_transform(X_num)
            self.num_col = X_num.shape[1]
        if X_cat is not None:
            self.cat_encoder = rare_merger(self.rho, output_type='one_hot', rare_threshold=self.rare_threshold)
            X_cat = self.cat_encoder.fit_transform(X_cat)
            self.cat_col = X_cat.shape[1]
        return X_num, X_cat

    def reverse_data(self, df, path=None):
        if isinstance(df, pd.DataFrame):
            df = df.to_numpy()

        x_num = None
        x_cat = None
        if self.num_col > 0:
            x_num = df[:, 0 : self.num_col]
            if x_num.ndim == 1:
                x_num = x_num.reshape(-1, 1)
            if self.num_encoder is not None:
                x_num = self.num_encoder.inverse_transform(x_num).astype(float)
            if path is not None:
                np.save(os.path.join(path, 'X_num_train.npy'), x_num)

        if self.cat_col > 0:
            x_cat = df[:, self.num_col : self.num_col + self.cat_col]
            if x_cat.ndim == 1:
                x_cat = x_cat.reshape(-1, 1)
            if self.cat_encoder is not None:
                x_cat = self.cat_encoder.inverse_transform(x_cat).astype(str)
            if path is not None:
                np.save(os.path.join(path, 'X_cat_train.npy'), x_cat)

        y = df[:, -1].reshape(-1).astype(int)
        if path is not None:
            np.save(os.path.join(path, 'y_train.npy'), y)
        return x_num, x_cat, y


class DP_MERF(GenerativeModel):
    """A wrapper for the DP-MERF synthetic data mechanism."""

    def __init__(
        self,
        metadata=None,
        epsilon=1.0,
        delta=1e-5,
        num_preprocess='privtree',
        rare_threshold=0.005,
        num_features=1000,
        mini_batch_size=0.05,
        how_many_epochs=100,
        device=None,
        seed=0,
        label_column=None,
    ):
        # Set device with device_utils
        self.device, self.is_gpu = validate_and_get_device(device)
        
        self.metadata = metadata
        self.epsilon = epsilon
        self.delta = delta
        self.num_preprocess = num_preprocess
        self.rare_threshold = rare_threshold
        self.num_features = num_features
        self.mini_batch_size = mini_batch_size
        self.how_many_epochs = how_many_epochs
        self.seed = seed
        self.label_column = label_column

        self.datatype = DataFrame
        self.preprocessor = None
        self.generator = None
        self.parent_dir = None
        self._tmp_dir = None
        self._feature_columns = []
        self._numeric_columns = []
        self._categorical_columns = []
        self._original_columns = []
        self._original_label_column = None
        self.trained = False
        self.__name__ = f'DP_MERFEps{self.epsilon}Delta{self.delta}'

    def fit(self, data):
        assert isinstance(data, self.datatype), (
            f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        )

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')

        X_num, X_cat, y = self._build_feature_arrays(data)
        rho = cdp_rho(self.epsilon, self.delta) if self.epsilon is not None else 0.0
        self.preprocessor = _DP_MERFPreprocessor(self.num_preprocess, self.rare_threshold, rho)
        X_num, X_cat = self.preprocessor.fit_transform(X_num, X_cat)

        self._tmp_dir = tempfile.TemporaryDirectory()
        self.parent_dir = self._tmp_dir.name

        domain = {
            'X_num': [len(np.unique(X_num[:, i])) for i in range(X_num.shape[1])] if X_num is not None else [],
            'X_cat': [len(categories) for categories in self.preprocessor.cat_encoder.ordinal_encoder.categories_] if X_cat is not None else [],
            'y': [len(np.unique(y))],
        }

        from method.DP_MERF.single_generator_priv_all import merf_main

        args = SimpleNamespace(
            device=self.device,
            n_features_arg=self.num_features,
            mini_batch_size_arg=self.mini_batch_size,
            how_many_epochs_arg=self.how_many_epochs,
            test=False,
            dataset='merf',
        )

        self.generator = merf_main(
            args,
            {'X_num': X_num, 'X_cat': X_cat, 'y': y},
            domain,
            rho,
            parent_dir=self.parent_dir,
            seed_number=self.seed,
            is_priv_arg=self.epsilon is not None,
        )['merf_generator']

        self.trained = True

    def generate_samples(self, nsamples):
        assert self.trained, 'Model must first be fitted to some data.'

        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        self.generator.sample(nsamples, self.preprocessor, self.parent_dir, self.device)

        x_num, x_cat, y = self._load_generated_arrays(self.parent_dir)
        synthetic = self._arrays_to_dataframe(x_num, x_cat, y)
        return synthetic

    def _build_feature_arrays(self, data):
        self._original_columns = list(data.columns)
        numeric_columns = []
        categorical_columns = []
        for column in data.columns:
            if self.label_column is not None and column == self.label_column:
                continue
            column_meta = None
            if self.metadata is not None:
                column_meta = next((c for c in self.metadata['columns'] if c['name'] == column), None)

            if column_meta is not None:
                if column_meta['type'] in [FLOAT, INTEGER]:
                    numeric_columns.append(column)
                else:
                    categorical_columns.append(column)
            else:
                if data[column].dtype.kind in 'if':
                    numeric_columns.append(column)
                else:
                    categorical_columns.append(column)

        self._numeric_columns = numeric_columns
        self._categorical_columns = categorical_columns
        self._feature_columns = numeric_columns + categorical_columns

        if self.label_column is not None and self.label_column in data.columns:
            labels = data[self.label_column].astype(int).to_numpy()
            features = data.drop(columns=[self.label_column])
            self._original_label_column = self.label_column
        else:
            labels = np.zeros(len(data), dtype=int)
            features = data
            self._original_label_column = None

        X_num = None
        X_cat = None
        if len(numeric_columns) > 0:
            X_num = features[numeric_columns].to_numpy(dtype=float)
        if len(categorical_columns) > 0:
            X_cat = features[categorical_columns].astype(str).to_numpy(dtype=object)

        return X_num, X_cat, labels

    def _load_generated_arrays(self, path):
        x_num = None
        x_cat = None
        y = None
        if os.path.exists(os.path.join(path, 'X_num_train.npy')):
            x_num = np.load(os.path.join(path, 'X_num_train.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(path, 'X_cat_train.npy')):
            x_cat = np.load(os.path.join(path, 'X_cat_train.npy'), allow_pickle=True)
        if os.path.exists(os.path.join(path, 'y_train.npy')):
            y = np.load(os.path.join(path, 'y_train.npy'), allow_pickle=True)
        return x_num, x_cat, y

    def _arrays_to_dataframe(self, x_num, x_cat, y):
        synthetic = DataFrame(index=range(len(y) if y is not None else len(x_num) if x_num is not None else len(x_cat)))

        if x_num is not None:
            for idx, column in enumerate(self._numeric_columns):
                synthetic[column] = x_num[:, idx]
        if x_cat is not None:
            for idx, column in enumerate(self._categorical_columns):
                synthetic[column] = x_cat[:, idx]

        if self._original_label_column is not None and y is not None:
            synthetic[self._original_label_column] = y

        if self._original_columns:
            synthetic = synthetic[self._original_columns]

        return synthetic
