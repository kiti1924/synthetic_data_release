"""Parallel execution utilities for model evaluation"""
import importlib
import os
import warnings
from multiprocessing import cpu_count
import joblib
from numpy.random import seed
from tqdm import tqdm
from utils.logging import LOGGER

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
    """Return a model name for a config without instantiating it."""
    name, *params = config if isinstance(config, (tuple, list)) else (config,)
    if params:
        param_str = ','.join(str(p) for p in params)
        return f'{name}({param_str})'
    return name


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





class _StarmapHelper:
    """Picklable wrapper that unpacks a tuple arg for execution."""
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, args):
        return self.fn(*args)

def _gpu_device_requested():
    device = os.environ.get('SYNTHETIC_DATA_DEVICE', '')
    return bool(device) and ('cuda' in device.lower() or device.lower().startswith('gpu'))


import pickle

def _cached_worker_fn(worker_fn, task, cache_path):
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                res = pickle.load(f)
            LOGGER.info(f"Loaded cached result from {os.path.basename(cache_path)}")
            return res
        except Exception as e:
            LOGGER.warning(f"Failed to load cache {cache_path}: {e}. Recomputing.")
    
    res = worker_fn(*task)
    
    if cache_path:
        try:
            # Atomic save to prevent corruption, using PID to avoid process collisions
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp_path, 'wb') as f:
                pickle.dump(res, f)
            os.replace(tmp_path, cache_path)
        except Exception as e:
            LOGGER.warning(f"Failed to save cache {cache_path}: {e}")
            
    return res

def run_parallel_models(worker_fn, tasks, max_workers=None, desc="Models", cache_keys=None, cache_dir=None):
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

    use_mpi = os.environ.get('USE_MPI', '0') == '1'

    if max_workers == 1 and not use_mpi:
        return [
            _cached_worker_fn(worker_fn, task, os.path.join(cache_dir, cache_keys[i]) if cache_keys and cache_dir else None)
            for i, task in tqdm(enumerate(tasks), total=len(tasks), desc=desc)
        ]
    if max_workers is None and not use_mpi:
        has_gpu_tasks = any(model_requires_gpu(task[0]) for task in tasks if task and isinstance(task[0], (tuple, list, str)))
        if _gpu_device_requested() and has_gpu_tasks:
            max_workers = 1
        else:
            max_workers = min(cpu_count(), len(tasks))

    try:
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_size() > 1:
            use_mpi = True
    except (ImportError, RuntimeError):
        use_mpi = False

    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # SPMD Model: Partition tasks among ranks
        my_tasks = [task for i, task in enumerate(tasks) if i % size == rank]
        my_cache_paths = [
            os.path.join(cache_dir, cache_keys[i]) if cache_keys and cache_dir else None 
            for i in range(len(tasks)) if i % size == rank
        ]
        
        my_results = []
        disable_tqdm = (rank != 0)
        for task, c_path in tqdm(zip(my_tasks, my_cache_paths), total=len(my_tasks), desc=f"{desc}", disable=disable_tqdm):
            my_results.append(_cached_worker_fn(worker_fn, task, c_path))
            
        LOGGER.info(f"Node (Rank {rank}) finished processing {len(my_tasks)} tasks.")
            
        # Scalable Gather: Each rank writes results to a temp file, Rank 0 reads and combines them
        rank_file = os.path.join(cache_dir, f".tmp_results_rank_{rank}_{os.getpid()}.pkl")
        with open(rank_file, 'wb') as f:
            pickle.dump(my_results, f)
            
        comm.barrier()
        
        if rank == 0:
            # Reconstruct original results list order
            results = [None] * len(tasks)
            # Gather all file paths from ranks via MPI string gather to support arbitrary PIDs
            all_rank_files = comm.gather(rank_file, root=0)
            
            for r, rf in enumerate(all_rank_files):
                if rf and os.path.exists(rf):
                    with open(rf, 'rb') as f:
                        r_results = pickle.load(f)
                    
                    for i, res in enumerate(r_results):
                        orig_idx = r + i * size
                        if orig_idx < len(tasks):
                            results[orig_idx] = res
                    
                    try:
                        os.remove(rf)
                    except OSError:
                        pass
            return results
        else:
            comm.gather(rank_file, root=0)
            return []

    # Use joblib.Parallel instead of multiprocessing.Pool
    # Loky backend automatically uses memmapping for arrays > 1MB, solving the IPC bottleneck
    with tqdm(total=len(tasks), desc=desc) as pbar:
        results = joblib.Parallel(n_jobs=max_workers, backend='loky')(
            joblib.delayed(_cached_worker_fn)(worker_fn, task, os.path.join(cache_dir, cache_keys[i]) if cache_keys and cache_dir else None) 
            for i, task in enumerate(tasks)
        )
        pbar.update(len(results))
            
    return results
