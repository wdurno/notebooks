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
    parser.add_argument("--n-override", type=int, default=-1, help="If >0, reweight new information to this sample size") ## TODO MAKE USE OF THIS 

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
    ## optimize 
    model.memorize() 
    ## save model 
    model.save_quantized(str(args.output_dir)+'_quantized') 
    model.save(str(args.output_dir)+'_full') 
    ## clean-up FSDP 
    cleanup() 
    pass 

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
