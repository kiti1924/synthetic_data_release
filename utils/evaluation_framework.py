import os
import json
from os import mkdir, path
from argparse import ArgumentParser
from numpy import where, mean
from numpy.random import seed

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *
from utils.parallel import (
    is_generative_model_config,
    model_requires_gpu,
    get_optimal_workers_for_config,
    run_parallel_models,
    _gpu_device_requested,
)

SEED = 42

class EvaluationEngine:
    """Consolidated Evaluation Engine to manage boilerplate environment setup, execution, 
    and output tasks across all synthetic data CLIs."""
    def __init__(self, description="Evaluation CLI"):
        self.parser = ArgumentParser(description=description)
        datasource = self.parser.add_mutually_exclusive_group()
        datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
        datasource.add_argument('--datapath', '-D', type=str, help='Relative path to cwd of a local data file')
        self.parser.add_argument('--runconfig', '-RC', default='runconfig_mia.json', type=str, help='Path relative to cwd of runconfig file')
        self.parser.add_argument('--outdir', '-O', default='outputs/test', type=str, help='Path relative to cwd for storing output files')
        self.parser.add_argument('--workers', '-W', type=int, default=None, help='Number of parallel workers (default: CPU count)')
        self.parser.add_argument('--device', type=str, default=None, help='Device to use for models (e.g., "cpu", "cuda:0"). Defaults to GPU if available, otherwise CPU.')
        
        self.args = None
        self.runconfig = None
        self.rawPop = None
        self.metadata = None
        self.dname = None
        
        self.all_model_configs = []
        self.gm_configs = []
        self.san_configs = []

    def setup(self, cwd):
        self.args = self.parser.parse_args()
        
        # Set device environment variable for models
        if self.args.device:
            os.environ['SYNTHETIC_DATA_DEVICE'] = self.args.device
            LOGGER.info(f"Device set to: {self.args.device}")
            if 'cuda:' in self.args.device:
                try:
                    idx = self.args.device.split(':')[1]
                    os.environ['CUDA_VISIBLE_DEVICES'] = idx
                except IndexError:
                    pass
            elif self.args.device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = ''

        # Load runconfig
        with open(path.join(cwd, self.args.runconfig)) as f:
            self.runconfig = json.load(f)
        print('Runconfig:')
        print(self.runconfig)

        # Load data
        if self.args.s3name is not None:
            self.rawPop, self.metadata = load_s3_data_as_df(self.args.s3name)
            self.dname = self.args.s3name
        else:
            self.rawPop, self.metadata = load_local_data_as_df(path.join(cwd, self.args.datapath))
            self.dname = self.args.datapath.split('/')[-1]

        print(f'Loaded data {self.dname}:')
        print(self.rawPop.info())

        # Make sure outdir exists
        os.makedirs(self.args.outdir, exist_ok=True)

        seed(SEED)

        # Build serializable model configs for parallel execution
        if 'generativeModels' in self.runconfig.keys():
            for gm, paramsList in self.runconfig['generativeModels'].items():
                for params in paramsList:
                    self.all_model_configs.append((gm, *params))

        if 'sanitisationTechniques' in self.runconfig.keys():
            for name, paramsList in self.runconfig['sanitisationTechniques'].items():
                for params in paramsList:
                    self.all_model_configs.append((name, *params))

        self.gm_configs = [cfg for cfg in self.all_model_configs if is_generative_model_config(cfg)]
        self.san_configs = [cfg for cfg in self.all_model_configs if not is_generative_model_config(cfg)]

    def run_parallel_evaluation(self, eval_gm_worker, san_tasks, gm_tasks, eval_san_worker=None, iter_idx=None, desc_prefix="eval"):
        # Helper to execute worker distribution efficiently across GPU and CPU boundaries.
        all_results = []
        
        # Suffix handling
        suffix = f" {iter_idx+1}/{self.runconfig['nIter']}" if iter_idx is not None else ""
        eval_san_worker = eval_san_worker or eval_gm_worker
        
        # 1. Evaluate sanitisers (CPU only, optimally parallel)
        if san_tasks:
            san_workers = get_optimal_workers_for_config(san_tasks[0][0], self.args.workers)
            all_results.extend(run_parallel_models(
                eval_san_worker, san_tasks, max_workers=san_workers,
                desc=f"San {desc_prefix}{suffix}"))
                
        if gm_tasks:
            if _gpu_device_requested():
                # Separate GPU-requiring models from CPU-only models for optimal parallelization
                gpu_gm_tasks = [t for t in gm_tasks if model_requires_gpu(t[0])]
                cpu_gm_tasks = [t for t in gm_tasks if not model_requires_gpu(t[0])]
    
                # GPU models must serialize unless managed internally or over sub-devices
                if gpu_gm_tasks:
                    all_results.extend(run_parallel_models(
                        eval_gm_worker, gpu_gm_tasks, max_workers=1,
                        desc=f"GPU GM {desc_prefix}{suffix}"))
                
                # CPU-only models can parallelize
                if cpu_gm_tasks:
                    cpu_workers = get_optimal_workers_for_config(cpu_gm_tasks[0][0], self.args.workers)
                    all_results.extend(run_parallel_models(
                        eval_gm_worker, cpu_gm_tasks, max_workers=cpu_workers,
                        desc=f"CPU GM {desc_prefix}{suffix}"))
            else:
                # If CPU is used, all models can be parallelized based on worker count
                cpu_workers = get_optimal_workers_for_config(gm_tasks[0][0], self.args.workers)
                all_results.extend(run_parallel_models(
                    eval_gm_worker, gm_tasks, max_workers=cpu_workers,
                    desc=f"GM Models {desc_prefix}{suffix}"))
                    
        return all_results

    def dump_results(self, result_dict, prefix="Results"):
        outfile = f"{prefix}_{self.dname}"
        LOGGER.info(f"Write results to {path.join(self.args.outdir, outfile)}")
        with open(path.join(self.args.outdir, f'{outfile}.json'), 'w') as f:
            json.dump(result_dict, f, indent=2, default=json_numpy_serialzer)


def get_accuracy(guesses, labels, targetPresence):
    idxIn = where(targetPresence == LABEL_IN)[0]
    idxOut = where(targetPresence == LABEL_OUT)[0]

    pIn = sum([g == l for g, l in zip(guesses[idxIn], labels[idxIn])]) / len(idxIn)
    pOut = sum([g == l for g, l in zip(guesses[idxOut], labels[idxOut])]) / len(idxOut)
    return pIn, pOut


def get_tp_fp_rates(guesses, labels):
    labels_arr = labels.values
    guesses_arr = guesses.values

    targetIn = where(labels_arr == LABEL_IN)[0]
    targetOut = where(labels_arr == LABEL_OUT)[0]

    return (sum(guesses_arr[targetIn] == LABEL_IN) / len(targetIn),
            sum(guesses_arr[targetOut] == LABEL_IN) / len(targetOut))


def get_probs_correct(pdf, targetPresence):
    idxIn = where(targetPresence == LABEL_IN)[0]
    idxOut = where(targetPresence == LABEL_OUT)[0]

    pdf[pdf > 1.] = 1.
    return mean(pdf[idxIn]), mean(pdf[idxOut])


def get_mia_advantage(tp_rate, fp_rate):
    return tp_rate - fp_rate


def get_ai_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut


def get_util_advantage(pCorrectIn, pCorrectOut):
    return pCorrectIn - pCorrectOut


def get_prob_removed(before, after):
    idxIn = where(before == LABEL_IN)[0]
    return 1.0 - sum(after[idxIn] / len(idxIn))
