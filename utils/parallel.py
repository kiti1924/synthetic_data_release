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
            # Ensure cache directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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
        import numpy as np
        import random
        # 逐次実行時（メインプロセス）にモデル内で乱数が消費・リセットされても、
        # メインプロセスの乱数状態に影響を与えないように状態を退避・復元する。
        # これによりキャッシュヒット時と通常実行時の後続の再現性を完全に一致させる。
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            return [
                _cached_worker_fn(worker_fn, task, os.path.join(cache_dir, cache_keys[i]) if cache_keys and cache_dir else None)
                for i, task in tqdm(enumerate(tasks), total=len(tasks), desc=desc)
            ]
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
    if max_workers is None and not use_mpi:
        has_gpu_tasks = any(model_requires_gpu(task[0]) for task in tasks if task and isinstance(task[0], (tuple, list, str)))
        if _gpu_device_requested() and has_gpu_tasks:
            max_workers = 1
        else:
            max_workers = min(cpu_count(), len(tasks))

    try:
        from mpi4py import MPI
        # Check if actually running in an MPI environment with more than 1 rank
        if MPI.COMM_WORLD.Get_size() > 1:
            use_mpi = True
        else:
            # size=1 but MPI is loaded, we can still use the MPI code path or fallback
            pass
    except (ImportError, RuntimeError):
        use_mpi = False

    # --- EXECUTION PATH VISUALIZATION ---
    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        if rank == 0:
            LOGGER.info(f"🚀 [{desc}] MODE: MPI (Distributed across {size} nodes)")
            LOGGER.info(f"   Strategy: Hierarchical Dynamic Queue (MPI Inter-node + Joblib Intra-node)")
    else:
        LOGGER.info(f"💻 [{desc}] MODE: Local Parallel (Joblib/LOKY)")
        LOGGER.info(f"   Strategy: Multi-processing on {max_workers} cores")

    if use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Hierarchical Dynamic Queue (MPI + Joblib)
        # Each rank processes tasks in parallel using joblib (intra-node)
        # while MPI distributes work chunks across ranks (inter-node).
        local_n_jobs = max_workers if max_workers is not None else 1

        if rank == 0:
            results = [None] * len(tasks)
            next_task_idx = 0
            active_workers = 0
            
            # Use tqdm on Rank 0 to monitor progress
            with tqdm(total=len(tasks), desc=desc) as pbar:
                # Initial dispatch: send a chunk of tasks to each worker
                for r in range(1, size):
                    if next_task_idx < len(tasks):
                        # Chunk size matches local worker count for efficiency
                        chunk_size = min(local_n_jobs, len(tasks) - next_task_idx)
                        chunk_indices = list(range(next_task_idx, next_task_idx + chunk_size))
                        comm.send(chunk_indices, dest=r, tag=10)
                        next_task_idx += chunk_size
                        active_workers += 1
                    else:
                        comm.send(None, dest=r, tag=10)
                
                # If size=1, Rank 0 does all work locally using joblib
                if size == 1:
                    local_results = joblib.Parallel(n_jobs=local_n_jobs, backend='loky')(
                        joblib.delayed(_cached_worker_fn)(
                            worker_fn, tasks[i], 
                            os.path.join(cache_dir, cache_keys[i]) if cache_keys and cache_dir else None
                        ) for i in range(len(tasks))
                    )
                    pbar.update(len(tasks))
                    return local_results

                # Collect results and dispatch remaining chunks
                while active_workers > 0:
                    status = MPI.Status()
                    result_bundle = comm.recv(source=MPI.ANY_SOURCE, tag=20, status=status)
                    worker_rank = status.Get_source()
                    
                    # result_bundle is a list of (idx, res) tuples
                    for idx, res in result_bundle:
                        results[idx] = res
                        pbar.update(1)
                    
                    if next_task_idx < len(tasks):
                        chunk_size = min(local_n_jobs, len(tasks) - next_task_idx)
                        chunk_indices = list(range(next_task_idx, next_task_idx + chunk_size))
                        comm.send(chunk_indices, dest=worker_rank, tag=10)
                        next_task_idx += chunk_size
                    else:
                        comm.send(None, dest=worker_rank, tag=10)
                        active_workers -= 1
            
            LOGGER.info(f"Master (Rank 0) finished collecting {len(tasks)} results.")
            return results
        else:
            # Worker loop: process chunks using local multi-processing
            while True:
                task_indices = comm.recv(source=0, tag=10)
                if task_indices is None:
                    break
                
                # Execute chunk in parallel locally
                # Loky backend is used for efficient memory sharing of DataFrames
                chunk_results = joblib.Parallel(n_jobs=local_n_jobs, backend='loky')(
                    joblib.delayed(lambda idx: (idx, _cached_worker_fn(
                        worker_fn, tasks[idx], 
                        os.path.join(cache_dir, cache_keys[idx]) if cache_keys and cache_dir else None
                    )))(i) for i in task_indices
                )
                
                # Send back the whole chunk of results
                comm.send(chunk_results, dest=0, tag=20)
            
            LOGGER.info(f"Worker (Rank {rank}) finished all assigned chunks.")
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


def get_syn_data_cache_path(cache_dir, model_config, dname, iter_idx, nSynT, sizeSynT):
    """Generate a unique path for cached synthetic data."""
    import hashlib
    if cache_dir is None:
        return None
    
    config_str = str(model_config).encode('utf-8')
    config_hash = hashlib.md5(config_str).hexdigest()[:8]
    model_name = str(model_config[0]).replace('/', '_').replace(' ', '_')
    
    # Key includes model, dataset, iteration, and generation parameters
    filename = f"syn_{model_name}_{config_hash}_{dname}_iter{iter_idx}_n{nSynT}_s{sizeSynT}.pkl"
    return os.path.join(cache_dir, "syn_data", filename)


def load_syn_data(cache_path):
    """Load synthetic data list from cache if it exists."""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                res = pickle.load(f)
            from utils.parallel import LOGGER
            LOGGER.info(f"Loaded cached synthetic data from {os.path.basename(cache_path)}")
            return res
        except Exception as e:
            from utils.parallel import LOGGER
            LOGGER.warning(f"Failed to load syn data cache {cache_path}: {e}")
    return None


def save_syn_data(cache_path, syn_data_list):
    """Save synthetic data list to cache atomically."""
    if not cache_path:
        return
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        import os
        import pickle
        tmp_path = f"{cache_path}.tmp.{os.getpid()}"
        with open(tmp_path, 'wb') as f:
            pickle.dump(syn_data_list, f)
        os.replace(tmp_path, cache_path)
    except Exception as e:
        from utils.parallel import LOGGER
        LOGGER.warning(f"Failed to save syn data cache {cache_path}: {e}")
