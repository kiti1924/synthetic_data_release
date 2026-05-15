"""
Command-line interface for running privacy evaluation under an attribute inference adversary
"""

import json
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from os import path
from numpy.random import choice, seed
from argparse import ArgumentParser
import pandas as pd
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *
from utils.parallel import (create_model, is_generative_model, is_generative_model_config,
                           get_syn_data_cache_path, load_syn_data, save_syn_data)
from utils.evaluation_framework import EvaluationEngine

from attack_models.reconstruction import LinRegAttack, RandForestAttack


def _deep_tuple(obj):
    """Recursively convert lists to tuples so nested configs are hashable."""
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_tuple(x) for x in obj)
    return obj

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)

SEED = 42


def inference_eval_gm_worker(iter_idx, model_config, rawTout, targets, targetIDs,
                             sensitive_attrs, metadata, runconfig, dname, cache_dir):
    """Evaluate one generative model for inference attack across all targets.
    :return: tuple: (iter_idx, model_name, {(tid, sa): result_dict})
    """
    try:
        nSynT = runconfig['nSynT']
        sizeSynT = runconfig['sizeSynT']

        # Check cache for synthetic data WITHOUT target
        syn_cache_path = get_syn_data_cache_path(cache_dir, model_config, dname, iter_idx, nSynT, sizeSynT)
        synT_list = load_syn_data(syn_cache_path)

        if synT_list is None:
            model = create_model(model_config, metadata)
            model.set_seed(SEED)
            model.fit(rawTout)
            synT_list = [model.generate_samples(sizeSynT) for _ in range(nSynT)]
            save_syn_data(syn_cache_path, synT_list)

        model_name = model_config[0]

        attacks = {}
        for sa, atype in sensitive_attrs.items():
            if atype == 'LinReg':
                attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=metadata)
            elif atype == 'Classification':
                attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=metadata)

        results = {}

        for sa, Attack in attacks.items():
            for tid in targetIDs:
                target = targets.loc[[tid]]
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                results[(tid, sa)] = {
                    'AttackerGuess': [],
                    'ProbCorrect': [],
                    'TargetPresence': []
                }

                for syn in synT_list:
                    guess = Attack.attack(targetAux, attemptLinkage=False, data=syn)
                    pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=False, data=syn)

                    results[(tid, sa)]['AttackerGuess'].append(guess)
                    results[(tid, sa)]['ProbCorrect'].append(pCorrect)
                    results[(tid, sa)]['TargetPresence'].append(LABEL_OUT)

        # 2. IN evaluation
        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = pd.concat([rawTout, target])

            # Re-train/generate for "with target" data (not cached)
            model = create_model(model_config, metadata)
            model.set_seed(SEED)
            model.fit(rawTin)
            synTwithTarget = [model.generate_samples(sizeSynT) for _ in range(nSynT)]
            
            for sa, Attack in attacks.items():
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                for syn in synTwithTarget:
                    guess = Attack.attack(targetAux, attemptLinkage=False, data=syn)
                    pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=False, data=syn)

                    results[(tid, sa)]['AttackerGuess'].append(guess)
                    results[(tid, sa)]['ProbCorrect'].append(pCorrect)
                    results[(tid, sa)]['TargetPresence'].append(LABEL_IN)

        return (iter_idx, model_name, results)
    except Exception as e:
        LOGGER.error(f"Inference evaluation failed for model {model_config[0]}: {e}")
        return (iter_idx, model_config[0], {})


def inference_eval_san_worker(iter_idx, model_config, rawTout, targets, targetIDs,
                               sensitive_attrs, metadata, runconfig, dname, cache_dir):
    """Evaluate one sanitiser for inference attack across all targets.
    :return: tuple: (iter_idx, model_name, {(tid, sa): result_dict})
    """
    try:
        model = create_model(model_config, metadata)
        model.set_seed(SEED)
        attack_metadata = model.metadata

        attacks = {}
        for sa, atype in sensitive_attrs.items():
            if atype == 'LinReg':
                attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=attack_metadata)
            elif atype == 'Classification':
                attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=attack_metadata)

        results = {}

        sanOut = model.sanitise(rawTout)

        for sa, Attack in attacks.items():
            Attack.set_seed(SEED)
            Attack.train(sanOut)
            for tid in targetIDs:
                target = targets.loc[[tid]]
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                guess = Attack.attack(targetAux, attemptLinkage=True, data=sanOut)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=sanOut)

                results[(tid, sa)] = {
                    'AttackerGuess': [guess],
                    'ProbCorrect': [pCorrect],
                    'TargetPresence': [LABEL_OUT]
                }

        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = pd.concat([rawTout, target])
            sanIn = model.sanitise(rawTin)

            for sa, Attack in attacks.items():
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                Attack.train(sanIn)

                guess = Attack.attack(targetAux, attemptLinkage=True, data=sanIn)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=sanIn)

                results[(tid, sa)]['AttackerGuess'].append(guess)
                results[(tid, sa)]['ProbCorrect'].append(pCorrect)
                results[(tid, sa)]['TargetPresence'].append(LABEL_IN)

        return (iter_idx, model.name, results)
    except Exception as e:
        LOGGER.error(f"Inference evaluation failed for sanitiser {model_config[0]}: {e}")
        return (iter_idx, model_config[0], {})


def inference_eval_raw_worker(iter_idx, rawTout, targets, targetIDs,
                             sensitive_attrs, metadata, runconfig):
    """Evaluate inference attack on raw data for one iteration.
    :return: tuple: (iter_idx, results dict: {(tid, sa): result_dict})
    """
    try:
        attacks = {}
        for sa, atype in sensitive_attrs.items():
            if atype == 'LinReg':
                attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=metadata)
            elif atype == 'Classification':
                attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=metadata)

        results = {}
        # 1. OUT evaluation
        for sa, Attack in attacks.items():
            Attack.train(rawTout)
            for tid in targetIDs:
                target = targets.loc[[tid]]
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                guess = Attack.attack(targetAux, attemptLinkage=True, data=rawTout)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=rawTout)

                results[(tid, sa)] = {
                    'AttackerGuess': [guess],
                    'ProbCorrect': [pCorrect],
                    'TargetPresence': [LABEL_OUT]
                }

        # 2. IN evaluation
        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = pd.concat([rawTout, target])

            for sa, Attack in attacks.items():
                Attack.train(rawTin)
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                guess = Attack.attack(targetAux, attemptLinkage=True, data=rawTin)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=rawTin)

                results[(tid, sa)]['AttackerGuess'].append(guess)
                results[(tid, sa)]['ProbCorrect'].append(pCorrect)
                results[(tid, sa)]['TargetPresence'].append(LABEL_IN)

        return (iter_idx, results)
    except Exception as e:
        LOGGER.error(f"Raw inference evaluation failed for iteration {iter_idx}: {e}")
        return (iter_idx, {})


def main():
    engine = EvaluationEngine(description="Command-line interface for running privacy evaluation for attribute inference")
    engine.setup(cwd)

    runconfig = engine.runconfig
    rawPop = engine.rawPop
    metadata = engine.metadata
    args = engine.args

    ########################
    #### GAME INPUTS #######
    ########################
    # Pick targets
    targetIDs = choice(list(rawPop.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])

    targets = rawPop.loc[targetIDs, :]

    # Drop targets from population
    rawPopDropTargets = rawPop.drop(targetIDs)

    ##################################
    ######### EVALUATION #############
    ##################################

    resultsTargetPrivacy = {
        tid: {sa: {'Raw': {}} for sa in runconfig['sensitiveAttributes']}
        for tid in targetIDs
    }

    all_rawTout = []
    for nr in range(runconfig['nIter']):
        rIdx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        all_rawTout.append(rawPopDropTargets.loc[rIdx])

    ###############
    ## PARALLEL RAW EVALUATION (Distributed via MPI)
    ###############
    raw_tasks = []
    raw_iter_idxs = []
    for nr in range(runconfig['nIter']):
        raw_tasks.append((nr, all_rawTout[nr], targets, targetIDs,
                         runconfig['sensitiveAttributes'], metadata, runconfig))
        raw_iter_idxs.append(nr)

    raw_results = engine.run_parallel_evaluation(
        eval_gm_worker=inference_eval_raw_worker,
        san_tasks=[],
        gm_tasks=raw_tasks,
        gm_iter_idxs=raw_iter_idxs,
        desc_prefix="raw_eval"
    )

    if not engine.is_worker:
        for nr, results in raw_results:
            for (tid, sa), result_dict in results.items():
                resultsTargetPrivacy[tid][sa]['Raw'][nr] = result_dict

    ###############
    ## PARALLEL MODEL EVALUATION (Flattened across iterations)
    ###############
    gm_tasks = []
    gm_iter_idxs = []
    san_tasks = []
    san_iter_idxs = []

    for nr in range(runconfig['nIter']):
        rawTout = all_rawTout[nr]
        for cfg in engine.gm_configs:
            gm_tasks.append((nr, cfg, rawTout, targets, targetIDs,
                             runconfig['sensitiveAttributes'], metadata, runconfig,
                             engine.dname, engine.cache_dir))
            gm_iter_idxs.append(nr)
        for cfg in engine.san_configs:
            san_tasks.append((nr, cfg, rawTout, targets, targetIDs,
                              runconfig['sensitiveAttributes'], metadata, runconfig))
            san_iter_idxs.append(nr)

    all_results = engine.run_parallel_evaluation(
        eval_gm_worker=inference_eval_gm_worker,
        eval_san_worker=inference_eval_san_worker,
        san_tasks=san_tasks,
        gm_tasks=gm_tasks,
        san_iter_idxs=san_iter_idxs,
        gm_iter_idxs=gm_iter_idxs,
        desc_prefix="eval"
    )

    for nr, model_name, results in all_results:
        for (tid, sa), result_dict in results.items():
            if model_name not in resultsTargetPrivacy[tid][sa]:
                resultsTargetPrivacy[tid][sa][model_name] = {}
            resultsTargetPrivacy[tid][sa][model_name][nr] = result_dict

    engine.dump_results(resultsTargetPrivacy, prefix="ResultsMLEAI")

if __name__ == "__main__":
    main()