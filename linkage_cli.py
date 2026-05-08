"""
Command-line interface for running privacy evaluation with respect to the risk of linkability
"""

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json

from os import path
from numpy.random import choice, seed
from argparse import ArgumentParser
from pandas import DataFrame


def _deep_tuple(obj):
    """Recursively convert lists to tuples so nested configs are hashable."""
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_tuple(x) for x in obj)
    return obj
import pandas as pd
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *
from utils.parallel import create_model, is_generative_model, is_generative_model_config
from utils.evaluation_framework import EvaluationEngine

from feature_sets.independent_histograms import HistogramFeatureSet
from feature_sets.model_agnostic import NaiveFeatureSet, EnsembleFeatureSet
from feature_sets.bayes import CorrelationsFeatureSet

from attack_models.mia_classifier import (MIAttackClassifierRandomForest,
                                          generate_mia_shadow_data,
                                          generate_mia_anon_data)

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)


SEED = 42


def linkage_attack_worker(model_config, tid, target, rawA, metadata, runconfig):
    """Train MIA attacks for one (target, model) pair."""
    try:
        model = create_model(model_config, metadata)
        model.set_seed(SEED)
        model.multiprocess = False  # Pool ワーカー内では子プロセス生成不可
        attack_metadata = metadata if is_generative_model(model) else model.metadata
        trained_attacks = {}

        if is_generative_model(model):
            synA, labelsA = generate_mia_shadow_data(
                model, target, rawA,
                runconfig['sizeRawT'], runconfig['sizeSynT'],
                runconfig['nShadows'], runconfig['nSynA'], SEED)

            for Feature in [NaiveFeatureSet(model.datatype),
                            HistogramFeatureSet(model.datatype, metadata),
                            CorrelationsFeatureSet(model.datatype, metadata)]:
                Attack = MIAttackClassifierRandomForest(metadata, Feature)
                Attack.set_seed(SEED)
                Attack.train(synA, labelsA)
                trained_attacks[Feature.__name__] = Attack
        else:
            sanA, labelsA = generate_mia_anon_data(
                model, target, rawA,
                runconfig['sizeRawT'],
                runconfig['nShadows'] * runconfig['nSynA'], SEED)

            for Feature in [NaiveFeatureSet(DataFrame),
                            HistogramFeatureSet(DataFrame, attack_metadata,
                                               nbins=model.histogram_size),
                            CorrelationsFeatureSet(DataFrame, attack_metadata),
                            EnsembleFeatureSet(DataFrame, attack_metadata,
                                              nbins=model.histogram_size)]:
                Attack = MIAttackClassifierRandomForest(metadata=attack_metadata, FeatureSet=Feature)
                Attack.set_seed(SEED)
                Attack.train(sanA, labelsA)
                trained_attacks[Feature.__name__] = Attack

        return (tid, model.__name__, trained_attacks, _deep_tuple(model_config))
    except Exception as e:
        LOGGER.error(f"Linkage attack training failed for model {model_config[0]} and target {tid}: {e}")
        return (tid, model_config[0], {}, _deep_tuple(model_config))


def linkage_eval_worker(model_config, rawTout, targets, targetIDs,
                        attacks_for_model, metadata, runconfig):
    """Evaluate one model across all targets for one game iteration."""
    try:
        model = create_model(model_config, metadata)
        model.set_seed(SEED)
        model.multiprocess = False  # Pool ワーカー内では子プロセス生成不可
        nSynT = runconfig['nSynT']
        sizeSynT = runconfig['sizeSynT']
        per_target_results = {}

        if is_generative_model(model):
            model.fit(rawTout)
            synTwithoutTarget = [model.generate_samples(sizeSynT) for _ in range(nSynT)]
            synLabelsOut = [LABEL_OUT for _ in range(nSynT)]

            for tid in targetIDs:
                target = targets.loc[[tid]]
                rawTin = pd.concat([rawTout, target])
                model.fit(rawTin)
                synTwithTarget = [model.generate_samples(sizeSynT) for _ in range(nSynT)]
                synLabelsIn = [LABEL_IN for _ in range(nSynT)]

                synT = synTwithoutTarget + synTwithTarget
                synTlabels = synLabelsOut + synLabelsIn

                per_target_results[tid] = {}
                for feature, Attack in attacks_for_model[tid].items():
                    attackerGuesses = Attack.attack(synT)
                    per_target_results[tid][feature] = {
                        'Secret': synTlabels,
                        'AttackerGuess': attackerGuesses
                    }
        else:
            sanOut = model.sanitise(rawTout)
            for tid in targetIDs:
                target = targets.loc[[tid]]
                rawTin = pd.concat([rawTout, target])
                sanIn = model.sanitise(rawTin)

                sanT = [sanOut, sanIn]
                sanTLabels = [LABEL_OUT, LABEL_IN]

                per_target_results[tid] = {}
                for feature, Attack in attacks_for_model[tid].items():
                    attackerGuesses = Attack.attack(sanT, attemptLinkage=True, target=target)
                    per_target_results[tid][feature] = {
                        'Secret': sanTLabels,
                        'AttackerGuess': attackerGuesses
                    }

        return (model.__name__, per_target_results)
    except Exception as e:
        LOGGER.error(f"Linkage evaluation failed for model {model_config[0]}: {e}")
        return (model_config[0], {})


def main():
    engine = EvaluationEngine(description="Command-line interface for running privacy evaluation w.r.t linkage risk")
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

    # Init adversary's prior knowledge
    rawAidx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawA'], replace=False).tolist()
    rawA = rawPop.loc[rawAidx, :]

    # Base targets and pools established

    ###################################
    #### ATTACK TRAINING #############
    ##################################
    if not engine.is_worker:
        print('\n---- Attack training ----')

    attack_gm_tasks = [
        (cfg, tid, targets.loc[[tid]], rawA, metadata, runconfig)
        for tid in targetIDs for cfg in engine.gm_configs
    ]
    attack_san_tasks = [
        (cfg, tid, targets.loc[[tid]], rawA, metadata, runconfig)
        for tid in targetIDs for cfg in engine.san_configs
    ]
    
    attack_results = engine.run_parallel_evaluation(
        eval_gm_worker=linkage_attack_worker,
        san_tasks=attack_san_tasks,
        gm_tasks=attack_gm_tasks,
        desc_prefix="attack training"
    )

    attack_results = engine.bcast_data(attack_results)

    attacks = {}
    cfg_to_model_name = {}
    for tid, model_name, trained, cfg_key in attack_results:
        attacks.setdefault(tid, {})[model_name] = trained
        cfg_to_model_name[cfg_key] = model_name

    ##################################
    ######### EVALUATION #############
    ##################################
    resultsTargetPrivacy = {tid: {} for tid in targetIDs}

    for nr in range(runconfig['nIter']):
        rIdx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawPopDropTargets.loc[rIdx]

        eval_gm_tasks = [
            (cfg, rawTout, targets, targetIDs,
             {tid: attacks[tid][cfg_to_model_name[_deep_tuple(cfg)]] for tid in targetIDs},
             metadata, runconfig)
            for cfg in engine.gm_configs
        ]
        eval_san_tasks = [
            (cfg, rawTout, targets, targetIDs,
             {tid: attacks[tid][cfg_to_model_name[_deep_tuple(cfg)]] for tid in targetIDs},
             metadata, runconfig)
            for cfg in engine.san_configs
        ]
        
        eval_results = engine.run_parallel_evaluation(
            eval_gm_worker=linkage_eval_worker,
            san_tasks=eval_san_tasks,
            gm_tasks=eval_gm_tasks,
            iter_idx=nr,
            desc_prefix="eval iter"
        )

        for model_name, per_target in eval_results:
            for tid, feature_results in per_target.items():
                if model_name not in resultsTargetPrivacy[tid]:
                    resultsTargetPrivacy[tid][model_name] = {}
                resultsTargetPrivacy[tid][model_name][nr] = feature_results

    engine.dump_results(resultsTargetPrivacy, prefix="ResultsMIA")


if __name__ == "__main__":
    main()