import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import gc
import json
import logging
from argparse import ArgumentParser

# Import main functions from the individual CLIs
from utility_cli import main as utility_main
from linkage_cli import main as linkage_main
from inference_cli import main as inference_main

LOGGER = logging.getLogger(__name__)

def main():
    argparser = ArgumentParser(description="Run all synthetic data evaluation games (Utility, Linkage, Inference)")
    datasource = argparser.add_mutually_exclusive_group(required=True)
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str, help='Relative path to cwd of a local data file')
    
    argparser.add_argument('--rc-utility', '-RCU', default='tests/utility/runconfig.json', type=str, help='Path to runconfig for Utility eval')
    argparser.add_argument('--rc-linkage', '-RCL', default='tests/linkage/runconfig.json', type=str, help='Path to runconfig for Linkage eval')
    argparser.add_argument('--rc-inference', '-RCI', default='tests/inference/runconfig.json', type=str, help='Path to runconfig for Inference eval')
    
    argparser.add_argument('--outdir', '-O', default='outputs/all', type=str, help='Path for storing all output files')
    argparser.add_argument('--workers', '-W', type=int, default=None,
                           help='Number of parallel workers (default: CPU count)')
    argparser.add_argument('--device', type=str, default=None,
                           help='Device to use for models (e.g., "cpu", "cuda:0").')
    argparser.add_argument('--use-mpi', action='store_true',
                           help='Use MPI for distributing tasks across multiple nodes.')
                           
    args = argparser.parse_args()
    
    # We will simulate sys.argv for the underlying argparsers in EvaluationEngine
    import sys
    
    # Set MPI environment if requested
    rank = 0
    if args.use_mpi:
        os.environ['USE_MPI'] = '1'
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
        except (ImportError, RuntimeError):
            pass

    if rank > 0:
        # Suppress non-critical logs on workers at this level
        logging.getLogger().setLevel(logging.WARNING)

    def build_argv_for_task(runconfig_path):
        new_argv = [sys.argv[0]]
        if args.s3name:
            new_argv.extend(['--s3name', args.s3name])
        elif args.datapath:
            new_argv.extend(['--datapath', args.datapath])
            
        new_argv.extend(['--runconfig', runconfig_path])
        new_argv.extend(['--outdir', args.outdir])
        
        if args.workers is not None:
            new_argv.extend(['--workers', str(args.workers)])
        if args.device is not None:
            new_argv.extend(['--device', args.device])
        if args.use_mpi:
            new_argv.extend(['--use-mpi'])
            
        return new_argv

    # Ensure output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    original_argv = sys.argv.copy()

    #---------------------------------------------
    # 1. UTILITY EVALUATION
    #---------------------------------------------
    logging.info("====================================")
    logging.info("🚀 STARTING UTILITY EVALUATION")
    logging.info("====================================")
    sys.argv = build_argv_for_task(args.rc_utility)
    try:
        utility_main()
    except Exception as e:
        logging.error(f"Utility evaluation failed: {e}")
        
    gc.collect()
    _clear_cuda()

    #---------------------------------------------
    # 2. LINKAGE EVALUATION
    #---------------------------------------------
    logging.info("====================================")
    logging.info("🚀 STARTING LINKAGE EVALUATION")
    logging.info("====================================")
    sys.argv = build_argv_for_task(args.rc_linkage)
    try:
        linkage_main()
    except Exception as e:
        logging.error(f"Linkage evaluation failed: {e}")

    gc.collect()
    _clear_cuda()

    #---------------------------------------------
    # 3. INFERENCE EVALUATION
    #---------------------------------------------
    logging.info("====================================")
    logging.info("🚀 STARTING INFERENCE EVALUATION")
    logging.info("====================================")
    sys.argv = build_argv_for_task(args.rc_inference)
    try:
        inference_main()
    except Exception as e:
        logging.error(f"Inference evaluation failed: {e}")

    gc.collect()
    _clear_cuda()

    sys.argv = original_argv
    logging.info("====================================")
    logging.info("✅ ALL EVALUATIONS COMPLETED")
    logging.info("====================================")

    if rank == 0:
        from utils.report_validation import check_report_key_consistency
        check_report_key_consistency(args.outdir)


def _clear_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
