import argparse 
import torch 
import torch.distributed as dist 
from pathlib import Path 
import os 
from fsdp_ssr_llama_8b import FsdpSsrLlama8B 

## set master address and port for FSDP 
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # enable debug logging

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine‑tune the Llama‑with‑SSR model on RL replay‑buffer data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    parser.add_argument("--data-path", type=Path, required=True,
                        help="Path to a torch‑serialized ReplayBuffer (e.g. .pt)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory in which to save the fine‑tuned model")
    parser.add_argument("--model-checkpoint", type=Path, default=None,
                        help="Optional path to an existing model checkpoint to resume from")

    # Memorization parameters 
    parser.add_argument("--rl-coef", type=float, default=.5, help="RL loss weighting, relative to LLM weight of 1") 
    parser.add_argument("--n-override", type=int, default=-1, help="If > 0, reweight new information to this sample size") ## TODO MAKE USE OF THIS 
    parser.add_argument("--subsample-size", type=int, default=-1, 
                        help="If > 0, randomly subsample the replay buffer, with slight bias toward non-zero rewards") 
    parser.add_argument("--pad-percentile", type=float, default=-1., 
                        help="in (0,1), if > 0, find this percentile non-zero FIM diagonal estimates and assign all lower values to it.") 

    return parser.parse_args()

def setup(rank, world_size):
    """ Initialize Process Group """
    ## dist.init_process_group("nccl", rank=rank, world_size=world_size) ## CPU+GPU ops needed 
    dist.init_process_group("gloo", rank=rank, world_size=world_size) 
    torch.cuda.set_device(rank)  ## ensure each process uses the correct GPU 
    num_threads = int(os.getenv("OMP_NUM_THREADS", 1)) 
    torch.set_num_threads(num_threads) 
    torch.set_num_interop_threads(2) 
    print(f'Rank {rank} using {torch.get_num_threads()} CPU threads')
    pass 

def cleanup():
    """ Destroy Process Group """
    dist.destroy_process_group() 
    pass 

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main(args) -> None: 
    ## init FSDP 
    setup(args.rank, args.world_size) 
    ## distributed model init 
    model = FsdpSsrLlama8B(load_path=args.model_checkpoint, rl_coef=args.rl_coef, seq_len=2048, use_fsdp=False) 
    ## load data on each rank 
    model.replay_buffer.load(args.data_path) 
    if args.subsample_size > 0:
        model.replay_buffer = model.random_dataset_subset(args.subsample_size) 
        pass 
    if args.rank == 0:
        print(f'Memorizing {len(model.replay_buffer)} observations...') 
        pass 
    ## optimize 
    model.memorize() 
    ## pad info diagonal by percentile 
    # model.ssr_residual_diagonal ## ONLY ON RANK 0 
    if args.rank == 0 and args.pad_percentile > 0.: 
        print('Padding information diagonal...')
        model.ssr_residual_diagonal = winsorize_low_tail(model.ssr_residual_diagonal, p=args.pad_percentile) 
        pass 
    ## save model 
    model.save_quantized(str(args.output_dir)+'_quantized') 
    model.save(str(args.output_dir)+'_full') 
    ## clean-up FSDP 
    cleanup() 
    pass 

def winsorize_low_tail(lam: torch.Tensor, p: float, eps: float = 1e-8):
    mask = lam > eps
    if mask.any():
        q = torch.quantile(lam[mask], p / 100.0)     # scalar (0‑D tensor)
        tmp = lam[mask]                               # avoid double indexing on RHS
        lam[mask] = torch.clamp(tmp, min=q)          # in‑place on lam
    return lam

if __name__ == "__main__": 
    ## torchrun sets these 
    rank = int(os.environ["RANK"]) 
    world_size = int(os.environ["WORLD_SIZE"]) 
    ## run 
    args = _parse_args() 
    args.rank = rank 
    args.world_size = world_size 
    main(args) 
    pass 
