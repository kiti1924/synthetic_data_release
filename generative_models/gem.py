"""
A generative model training algorithm based on
"Iterative Methods for Private Synthetic Data: Unifying Framework and New Methods"
by T. Liu, G. Vietri, Z. S. Wu, published in Advances in Neural Information Processing Systems (NeurIPS), 2021
Adapted from: https://github.com/vrtoddy/tab_bench
original repository: https://github.com/terranceliu/iterative-dp
"""

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame

from generative_models.generative_model import GenerativeModel
from method.AIM.cdp2adp import cdp_rho
from utils.constants import CATEGORICAL, ORDINAL
from utils.logging import LOGGER
from utils.device_utils import get_device, validate_and_get_device


class GEM(GenerativeModel):
    """A wrapper for the GEM synthetic data mechanism."""

    def __init__(self, metadata=None, epsilon=1.0, delta=1e-9, degree=2,
                 max_iters=500,
                 batch_size=500,
                 lr=1e-3, T=None, alpha=0.5,
                 workload=100000, workload_seed=0,
                 device=None,
                 resample=False,
                 verbose=False):
        self.metadata = metadata
        self.epsilon = epsilon
        self.delta = delta
        self.degree = degree
        self.max_iters = max_iters

        # GEM specific
        self.batch_size = batch_size
        self.lr = lr
        self.T = T
        self.alpha = alpha
        self.workload = workload
        self.workload_seed = workload_seed
        
        # Set device with device_utils
        self.device, self.is_gpu = validate_and_get_device(device)
        
        self.resample = resample
        self.verbose = verbose

        self.datatype = DataFrame
        self.mechanism = None
        self.dataset = None
        self.trained = False
        self.__name__ = f'GEMEps{self.epsilon}'

        self._reverse_maps = {}
        
        LOGGER.debug(f"GEM initialized with device: {self.device} (GPU: {self.is_gpu})")

    def fit(self, data):
        assert isinstance(data, self.datatype), (
            f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        )

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {data.shape}...')
        
        data = data.reset_index(drop=True)

        encoded_data = self._encode_data(data)
        domain = self._build_domain(encoded_data)

        from method.GEM.gem import GEM as GEMMechanism
        from method.GEM.mbi.dataset import Dataset
        from method.GEM.Util.qm import QueryManager
        from method.GEM.Util.util_gem import randomKwayData
        from method.GEM.Util.util_general import get_eps0_simple
        from method.GEM.mbi.torch_factor import Factor

        # Set device for torch_factor
        try:
            torch.tensor([1.0], device=self.device)
            Factor.set_device(self.device)
        except RuntimeError as e:
            LOGGER.warning(f"Device '{self.device}' not available: {e}. Falling back to CPU.")
            self.device = 'cpu'
            Factor.set_device('cpu')

        self.dataset = Dataset(encoded_data, domain)

        workloads = randomKwayData(self.dataset, self.workload, self.degree, seed=self.workload_seed)

        query_manager = QueryManager(self.dataset.domain, workloads)

        real_answers = query_manager.get_answer(self.dataset, concat=False)

        if self.T is None:
            self.T = 5 * len(data.columns)

        rho = cdp_rho(self.epsilon, self.delta)
        eps0 = get_eps0_simple(rho, self.T, alpha=self.alpha)

        dim = sum(domain.shape)

        args = SimpleNamespace(test=False)

        self.mechanism = GEMMechanism(
            embedding_dim=dim,
            device=self.device,
            gen_dim=[dim * 2, dim * 2],
            batch_size=self.batch_size,
            save_dir=None,
            preprocesser=None,
            args=args
        )

        self.mechanism.setup_data(self.dataset.df, list(self.dataset.df.columns), self.dataset.domain,
                                  overrides=['transformer'])

        self.mechanism.fit(
            T=self.T,
            eps0=eps0,
            sensitivity=1 / len(data),
            lr=self.lr,
            eta_min=None,
            qm=query_manager,
            real_answers=np.concatenate(real_answers),
            max_iters=self.max_iters,
            alpha=self.alpha,
            save_num=max(1, int(round(self.T * 0.5))),
            verbose=self.verbose,
            resample=self.resample
        )

        self.trained = True

    def generate_samples(self, nsamples):
        assert self.trained, 'Model must first be fitted to some data.'
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        import torch

        n_batch = int(np.ceil(nsamples / self.batch_size))

        samples = []
        for _ in range(n_batch):
            fake_data = self.mechanism.generate_fake_data(self.mechanism.mean, self.mechanism.std, resample=True)
            x = self.mechanism.get_onehot(fake_data).cpu()
            samples.append(x)
        x = torch.cat(samples, dim=0)
        synthetic_data = self.mechanism.transformer.inverse_transform(x, None)
        synthetic_data = synthetic_data.head(nsamples)

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

    def _build_domain(self, data):
        from method.GEM.mbi.domain import Domain

        return Domain(data.columns, [int(data[column].nunique()) for column in data.columns])
