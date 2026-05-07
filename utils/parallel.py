"""Parallel execution utilities for model evaluation"""
import importlib
import os
import warnings
from multiprocessing import cpu_count
import joblib
from numpy.random import seed
from tqdm import tqdm

from generative_models.generative_model import GenerativeModel


MODEL_REGISTRY = {
    "IndependentHistogram": "generative_models.data_synthesiser.IndependentHistogram",
    "BayesianNet": "generative_models.data_synthesiser.BayesianNet",
    "PrivBayes": "generative_models.data_synthesiser.PrivBayes",
    "CTGAN": "generative_models.ctgan.CTGAN",
    "PATEGAN": "generative_models.pate_gan.PATEGAN",
    "SanitiserNHS": "sanitisation_techniques.sanitiser_nhs.SanitiserNHS",
    "SanitiserNHSLegacy": "sanitisation_techniques.sanitiser_nhs.SanitiserNHSLegacy",
    "SanitiserMondrian": "sanitisation_techniques.sanitiser_mondrian.SanitiserMondrian",
    "SanitiserNHSMondrian": "sanitisation_techniques.sanitiser_mondrian.SanitiserNHSMondrian",
    "AIM": "generative_models.aim.AIM",
    "DP_MERF": "generative_models.dp_merf.DP_MERF",
    "GEM": "generative_models.gem.GEM",
    "PrivateGSD": "generative_models.private_gsd.PrivateGSD",
    "RAPpp": "generative_models.rappp.RAPpp",
    "PrivMRF": "generative_models.privmrf.PrivMRF",
    "TabDDPM": "generative_models.tabddpm.TabDDPM",
    "PrivSyn": "generative_models.privsyn.PrivSyn",
}

UTILITY_TASK_REGISTRY = {
    "RandForestClass": "predictive_models.predictive_model.RandForestClassTask",
    "LogRegClass": "predictive_models.predictive_model.LogRegClassTask",
    "LinReg": "predictive_models.predictive_model.LinRegTask",
}


SANITISER_MODELS = {'SanitiserNHS', 'SanitiserMondrian', 'SanitiserNHSMondrian'}
GPU_MODELS = {'CTGAN', 'PATEGAN', 'AIM', 'GEM', 'TabDDPM', 'DP_MERF', 'PrivateGSD', 'PrivMRF', 'PrivSyn'}


def _resolve_class(import_path):
    module_name, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_model(config, metadata):
    """Create a model instance from a (class_name, *params) config tuple."""
    name, *params = config
    if name not in MODEL_REGISTRY:
        raise ValueError(f'Unknown model: {name}')
    return _resolve_class(MODEL_REGISTRY[name])(metadata, *params)


def model_name_from_config(config, metadata=None):
    """Return a model name for a config, falling back when the model cannot be imported."""
    original_device = os.environ.get('SYNTHETIC_DATA_DEVICE')
    try:
        os.environ['SYNTHETIC_DATA_DEVICE'] = 'cpu'
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
        model = create_model(config, metadata)
        return model.__name__
    except (ModuleNotFoundError, ImportError, RuntimeError, ValueError):
        name, *params = config
        if params:
            param_str = ','.join(str(p) for p in params)
            return f'{name}({param_str})'
        return name
    finally:
        if original_device is None:
            os.environ.pop('SYNTHETIC_DATA_DEVICE', None)
        else:
            os.environ['SYNTHETIC_DATA_DEVICE'] = original_device


def is_generative_model_config(config):
    """Determine model/sanitiser type from config without full instantiation."""
    name, *_ = config
    return name not in SANITISER_MODELS


def model_requires_gpu(config):
    """Check if a model config normally runs on a GPU backend (PyTorch/TF)."""
    name, *_ = config if isinstance(config, (tuple, list)) else (config,)
    return name in GPU_MODELS


def get_optimal_workers_for_config(config, user_workers=None):
    """Get optimal number of workers for a specific config.
    
    If the model uses PyTorch/TF (GPU_MODELS) and a GPU is requested, force 1 worker to avoid 
    CUDA context fragmentation on GPU.
    Otherwise, use optimal parallel count.
    
    :param config: Model/sanitiser config tuple
    :param user_workers: User-specified max workers (None = auto)
    :return: Optimal worker count for this config
    """
    if user_workers == 1:
        return 1
    
    if model_requires_gpu(config) and _gpu_device_requested():
        return 1  # Deep learning model running on GPU: serialize to avoid CUDA context fragmentation
    
    # Standard CPU models (e.g. Scikit-learn, pgx, etc) or Deep Learning models on CPU

    if user_workers is None:
        return min(cpu_count(), 4)  # Reasonable default for CPU tasks
    return user_workers


def create_utility_task(config, metadata):
    """Create a utility task instance from a (task_name, *params) config tuple."""
    name, *params = config
    if name not in UTILITY_TASK_REGISTRY:
        raise ValueError(f'Unknown utility task: {name}')
    return _resolve_class(UTILITY_TASK_REGISTRY[name])(metadata, *params)


def is_generative_model(model):
    """Check if a model is a GenerativeModel (vs a Sanitiser)."""
    return isinstance(model, GenerativeModel)


def _worker_init():
    """Initialize worker process with a unique random seed."""
    seed(os.getpid())


class _StarmapHelper:
    """Picklable wrapper that unpacks a tuple arg for execution."""
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, args):
        return self.fn(*args)

def _gpu_device_requested():
    device = os.environ.get('SYNTHETIC_DATA_DEVICE', '')
    return bool(device) and ('cuda' in device.lower() or device.lower().startswith('gpu'))


def run_parallel_models(worker_fn, tasks, max_workers=None, desc="Models"):
    """Run tasks in parallel using multiprocessing.Pool with tqdm progress bar.

    :param worker_fn: callable: Worker function to execute
    :param tasks: list[tuple]: List of argument tuples for worker_fn
    :param max_workers: int or None: Number of worker processes (None = auto based on task count and device)
    :param desc: str: Description for the progress bar
    :return: list: Results from each worker
    
    Note: For fine-grained GPU/CPU control per model, see get_optimal_workers_for_config().
    For GPU-model-only pools, pass max_workers=1 explicitly.
    """
    if not tasks:
        return []
    if max_workers == 1:
        return [worker_fn(*task) for task in tqdm(tasks, desc=desc)]
    if max_workers is None:
        has_gpu_tasks = any(model_requires_gpu(task[0]) for task in tasks if task and isinstance(task[0], (tuple, list, str)))
        if _gpu_device_requested() and has_gpu_tasks:
            max_workers = 1
        else:
            max_workers = min(cpu_count(), len(tasks))

    # Use joblib.Parallel instead of multiprocessing.Pool
    # Loky backend automatically uses memmapping for arrays > 1MB, solving the IPC bottleneck
    with tqdm(total=len(tasks), desc=desc) as pbar:
        results = joblib.Parallel(n_jobs=max_workers, backend='loky')(
            joblib.delayed(worker_fn)(*task) for task in tasks
        )
        pbar.update(len(results))
            
    return results
