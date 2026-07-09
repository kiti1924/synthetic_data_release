"""
A generative model training algorithm based on
"TabDDPM: Modelling Tabular Data with Diffusion Models"
by A. Kotelnikov, D. Baranchuk, I. Rubachev, A. Babenko, published in International Conference on Machine Learning (ICML), 2023
Adapted from: https://github.com/vrtoddy/tab_bench
original repository: https://github.com/yandex-research/tab-ddpm
"""

import tempfile
import tomllib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame

from generative_models.generative_model import GenerativeModel
from utils.constants import CATEGORICAL, FLOAT, INTEGER, ORDINAL
from utils.logging import LOGGER
from utils.device_utils import validate_and_get_device
from method.AIM.cdp2adp import cdp_rho


make_dataset_from_df = None
finetune = None
ddpm_sampler = None


@dataclass(frozen=True)
class _Transformations:
    seed: int = 0
    normalization: str | None = None
    num_nan_policy: str | None = None
    cat_nan_policy: str | None = None
    cat_min_frequency: float | None = None
    cat_encoding: str | None = None
    y_policy: str | None = 'default'


def _load_config(path):
    with open(path, 'rb') as config_file:
        return tomllib.load(config_file)


class _TabDDPMPreprocessor:
    def __init__(self, numeric_columns, categorical_columns, all_columns):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.all_columns = all_columns
        self.synthetic_df = None

    def reverse_data(self, data, path=None):
        array = np.asarray(data)
        num_count = len(self.numeric_columns)
        cat_count = len(self.categorical_columns)

        numeric_part = array[:, :num_count] if num_count else None
        categorical_part = array[:, num_count:num_count + cat_count] if cat_count else None

        frame = DataFrame(index=range(len(array)))
        if numeric_part is not None:
            for idx, column in enumerate(self.numeric_columns):
                frame[column] = numeric_part[:, idx]
        if categorical_part is not None:
            for idx, column in enumerate(self.categorical_columns):
                frame[column] = categorical_part[:, idx]

        self.synthetic_df = frame.reindex(columns=self.all_columns)
        return self.synthetic_df


class TabDDPM(GenerativeModel):
    """A wrapper for the tabDDPM synthetic data mechanism."""

    def __init__(
        self,
        metadata=None,
        steps=50,
        lr=1e-4,
        batch_size=1024,
        num_timesteps=100,
        epsilon=1.0,
        delta=1e-9,
        device=None,
    ):
        # Set device with device_utils
        self.device, self.is_gpu = validate_and_get_device(device)
        
        self.metadata = metadata
        self.steps = steps
        self.lr = lr
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.epsilon = epsilon
        self.delta = delta

        self.datatype = DataFrame
        self.diffusion = None
        self.sampler = None
        self.dataset = None
        self.trained = False
        
        if self.epsilon is not None:
            self.__name__ = f'TabDDPMEps{self.epsilon}Delta{self.delta}'
        else:
            self.__name__ = 'TabDDPM'

        self._tmp_dir = None
        self._reverse_maps = {}
        self._numeric_columns = []
        self._categorical_columns = []
        self._all_columns = []

    def fit(self, data):
        assert isinstance(data, self.datatype), (
            f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        )

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')

        data = data.reset_index(drop=True)
        self._all_columns = list(data.columns)
        numeric_data, categorical_data, dummy_target = self._split_and_encode_data(data)

        global make_dataset_from_df, finetune, ddpm_sampler
        if make_dataset_from_df is None or finetune is None or ddpm_sampler is None:
            from method.TabDDPM.data.dataset import make_dataset_from_df as _make_dataset_from_df
            from method.TabDDPM.scripts.pretrain_and_finetune import finetune as _finetune
            from method.TabDDPM.scripts.sample import ddpm_sampler as _ddpm_sampler
        else:
            _make_dataset_from_df = make_dataset_from_df
            _finetune = finetune
            _ddpm_sampler = ddpm_sampler

        config = _load_config('method/TabDDPM/exp/colorado/simple_config.toml')
        config['device'] = self.device
        config['train']['main']['steps'] = self.steps
        config['train']['main']['lr'] = self.lr
        config['train']['main']['batch_size'] = self.batch_size
        config['diffusion_params']['num_timesteps'] = self.num_timesteps

        # '__none__' is the TOML-safe sentinel for Python None used in the config files
        t_params = {k: (None if v == '__none__' else v) for k, v in config['train']['T'].items()}

        dataset_dict = {
            'X_num': numeric_data,
            'X_cat': categorical_data,
            'y': dummy_target,
        }
        self.dataset = _make_dataset_from_df(
            dataset_dict,
            T=_Transformations(**t_params),
            y_num_classes=2,
            is_y_cond=True,
            task_type='binclass',
        )

        num_numerical_features = len(self._numeric_columns)

        self._tmp_dir = tempfile.TemporaryDirectory()
        rho = cdp_rho(self.epsilon, self.delta) if self.epsilon is not None else None
        self.diffusion = _finetune(
            **config['train']['main'],
            **config['diffusion_params'],
            parent_dir=self._tmp_dir.name,
            dataset=self.dataset,
            model_type=config['model_type'],
            model_params=config['model_params'],
            model_path=None,
            T_dict=t_params,
            num_numerical_features=num_numerical_features,
            device=self.device,
            dp_epsilon=self.epsilon,
            dp_delta=self.delta,
            rho_used=rho,
            report_every=False,
        )
        self.sampler = _ddpm_sampler(
            diffusion=self.diffusion,
            num_numerical_features=num_numerical_features,
            T_dict=t_params,
            dataset=self.dataset,
            model_params=config['model_params'],
        )

        self.trained = True

    def generate_samples(self, nsamples):
        assert self.trained, 'Model must first be fitted to some data.'
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        preprocessor = _TabDDPMPreprocessor(
            numeric_columns=self._numeric_columns,
            categorical_columns=self._categorical_columns,
            all_columns=self._all_columns,
        )
        self.sampler.sample(
            num_sample=nsamples,
            preprocesser=preprocessor,
            device=self.device,
            parent_dir=self._tmp_dir.name if self._tmp_dir is not None else None,
            batch_size=min(self.batch_size, nsamples) if nsamples > 0 else self.batch_size,
        )

        if preprocessor.synthetic_df is None:
            raise RuntimeError('tabDDPM did not produce synthetic data.')

        synthetic_data = self._decode_data(preprocessor.synthetic_df.copy())
        return synthetic_data

    def _split_and_encode_data(self, data):
        numeric = []
        categorical = []
        categorical_df = DataFrame(index=data.index)
        numeric_df = DataFrame(index=data.index)
        self._reverse_maps = {}

        for column in data.columns:
            column_meta = next((c for c in self.metadata['columns'] if c['name'] == column), None) if self.metadata else None
            if column_meta is not None and column_meta['type'] in [FLOAT, INTEGER]:
                numeric.append(column)
                numeric_df[column] = data[column].astype(float)
            elif column_meta is not None and column_meta['type'] in [CATEGORICAL, ORDINAL]:
                categorical.append(column)
                mapping = {value: idx for idx, value in enumerate(column_meta['i2s'])}
                categorical_df[column] = data[column].map(mapping).astype(int)
                self._reverse_maps[column] = {idx: value for value, idx in mapping.items()}
            else:
                if data[column].dtype.kind in 'if':
                    numeric.append(column)
                    numeric_df[column] = data[column].astype(float)
                else:
                    categorical.append(column)
                    values, codes = np.unique(data[column].astype(object), return_inverse=True)
                    categorical_df[column] = codes.astype(int)
                    self._reverse_maps[column] = {idx: value for idx, value in enumerate(values)}

        self._numeric_columns = numeric
        self._categorical_columns = categorical

        numeric_data = numeric_df.to_numpy(dtype=float) if numeric else None
        categorical_data = categorical_df.to_numpy(dtype=int) if categorical else None
        dummy_target = np.zeros(len(data), dtype=int)

        return numeric_data, categorical_data, dummy_target

    def _decode_data(self, data):
        for column in self._categorical_columns:
            reverse_map = self._reverse_maps[column]
            data[column] = pd.to_numeric(data[column], errors='coerce').round().astype(int).map(reverse_map)
        for column in self._numeric_columns:
            column_meta = next((c for c in self.metadata['columns'] if c['name'] == column), None) if self.metadata else None
            if column_meta is not None and column_meta['type'] == INTEGER:
                data[column] = pd.to_numeric(data[column], errors='coerce').round().astype(int)
        return data
