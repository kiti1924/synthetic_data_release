# Updates and Enhancements

This document highlights the major updates and improvements made to the framework compared to the original repository.

## 1. MPI Support for Distributed Execution

To handle large-scale datasets and numerous computationally expensive generative models, the framework now supports **distributed parallel execution using MPI (Message Passing Interface)** via `mpi4py`.

### Key Features
- **Multi-Node & Multi-Core Scaling**: Models and evaluation tasks can be distributed across a cluster of nodes, bypassing the single-machine limits of `joblib`.
- **Dynamic Worker Allocation**: The framework automatically partitions tasks across available MPI ranks (SPMD model) while preserving specific serial execution requirements for GPU-bound algorithms to avoid CUDA out-of-memory errors.
- **Graceful Fallback**: If the MPI library (`libmpi.so`) is unavailable or fails to load, the framework automatically catches the `RuntimeError` and falls back to standard `joblib`-based multiprocessing.

### Usage
Enable MPI by setting the `--use-mpi` flag in the CLI:
```bash
mpirun -np 4 uv run python all_cli.py -D data/adult --use-mpi
```

## 2. Robust Execution Caching

Evaluating complex generative models can take days. To prevent data loss from interruptions and speed up iterative testing, a **robust caching system** has been introduced.

### Key Features
- **Automatic Caching and Resumption**: Outputs from generative models and sanitisation techniques are automatically cached. If execution is interrupted (e.g., via `Ctrl+C` or a node failure), restarting the script will instantly load the completed tasks from the cache and resume where it left off.
- **Atomic Writes**: Cache files are written via atomic rename operations (e.g., writing to `.tmp.PID` and then `os.replace`), ensuring that a corrupted or partially written cache is never loaded.
- **Smart Cache Invalidation (MD5 Hashing)**: Cache keys are generated dynamically using an MD5 hash of the exact model hyperparameters and the dataset name. 
  - If you modify `runconfig.json` (e.g., changing epsilon from `1.0` to `10.0`), the hash changes, and the framework automatically re-evaluates the model without erroneously loading the old cache.
- **Strict Reproducibility Check**: When executing models sequentially in the main process (e.g. for GPU synchronization), the global random state is strictly preserved during cache loading. This ensures that random data sampling in subsequent iterations remains 100% reproducible whether the run is continuous or restarted from cache.
