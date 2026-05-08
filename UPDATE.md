# Updates and Enhancements

This document highlights the major updates and improvements made to the framework compared to the original repository.

## 0. Environment Setup

For execution in supercomputer or cluster environments (e.g., Cray XD2000), use the provided `setup.sh` script to build the environment. This script ensures that `mpi4py` is correctly linked with the system's MPI libraries.

```bash
# 1. Run the setup script on the login node
bash setup.sh

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Submit your batch script (e.g., job.sh) to the job scheduler
qsub job.sh
```

## 1. MPI Support for Distributed Execution

To handle large-scale datasets and numerous computationally expensive generative models, the framework now supports **distributed parallel execution using MPI (Message Passing Interface)** via `mpi4py`.

### Key Features
- **Distributed Evaluation Engine**: A robust framework (`EvaluationEngine`) that coordinates task distribution.
- **Hierarchical Parallelism**: Combines MPI for inter-node communication and `joblib` for intra-node core utilization.
- **Task Flattening (Performance Optimized)**: Evaluation tasks are flattened across all iterations, allowing all nodes to work on all model/iteration pairs simultaneously, maximizing throughput on supercomputers.
- **Dual-Mode Execution**:
    - **MPI Mode**: Automatically activated when running with `mpirun`.
    - **Local Mode**: Falls back to standard multi-processing on local machines, ensuring full backward compatibility.

## Execution Patterns
- **Multi-Node & Multi-Core Scaling**: Models and evaluation tasks can be distributed across a cluster of nodes, bypassing the single-machine limits of `joblib`.
- **Hierarchical Parallelism (MPI + Joblib)**: Rank 0 dynamically distributes task chunks across MPI nodes. Within each node, tasks are further parallelized locally using `joblib` (with the `loky` backend) to efficiently share memory for large DataFrames, minimizing inter-process communication overhead.
- **Unified CLI Rank Management**: A centralized `EvaluationEngine` handles all boilerplate execution. Rank 0 exclusively handles directory creation, logging, result aggregation, and data broadcasting (`bcast`), while worker ranks (`rank > 0`) bypass redundant I/O operations, ensuring clean logs and avoiding race conditions.
- **Master-Worker Task Distribution**: The framework utilizes a Master-Worker architecture rather than a symmetric SPMD model. Rank 0 acts as the master, dynamically dispatching task chunks to worker nodes (ranks > 0) to balance the load, while preserving specific serial execution requirements for GPU-bound algorithms to avoid CUDA out-of-memory errors.
- **Graceful Fallback**: If the MPI library (`libmpi.so`) is unavailable or fails to load, the framework automatically catches the `RuntimeError` and falls back to standard `joblib`-based multiprocessing.

### Usage

Enable distributed execution by combining an MPI launcher (`mpirun`) for node-level distribution with the `--workers` (`-W`) flag for local core-level parallelization.

**Example: Supercomputer / Cluster Execution**
In a cluster environment (e.g., using PBS/Torque), you allocate nodes and set 1 MPI process per node (`mpiprocs=1`), then utilize all local cores using `--workers`:

```bash
# Example Job Script Resource Allocation (e.g., 10 nodes, 192 cores per node, 1 MPI rank per node)
# #PBS -l select=10:ncpus=192:mem=740gb:mpiprocs=1

# Execute the framework
mpirun ./.venv/bin/python all_cli.py \
    -D data/dataset_name \
    -O outputs/dataset_name \
    -RCU tests/utility/runconfig.json \
    -RCL tests/linkage/runconfig.json \
    -RCI tests/inference/runconfig.json \
    -W 192 \
    --device cpu \
    --use-mpi
```

## 2. Robust Execution Caching

Evaluating complex generative models can take days. To prevent data loss from interruptions and speed up iterative testing, a **robust caching system** has been introduced.

### Key Features
- **Automatic Caching and Resumption**: Outputs from generative models and sanitisation techniques are automatically cached. If execution is interrupted (e.g., via `Ctrl+C` or a node failure), restarting the script will instantly load the completed tasks from the cache and resume where it left off.
- **Atomic Writes**: Cache files are written via atomic rename operations (e.g., writing to `.tmp.PID` and then `os.replace`), ensuring that a corrupted or partially written cache is never loaded.
- **Smart Cache Invalidation (MD5 Hashing)**: Cache keys are generated dynamically using an MD5 hash of the exact model hyperparameters and the dataset name. 
  - If you modify `runconfig.json` (e.g., changing epsilon from `1.0` to `10.0`), the hash changes, and the framework automatically re-evaluates the model without erroneously loading the old cache.
- **Strict Reproducibility Check**: When executing models sequentially in the main process (e.g. for GPU synchronization), the global random state is strictly preserved during cache loading. This ensures that random data sampling in subsequent iterations remains 100% reproducible whether the run is continuous or restarted from cache.
