import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear
import os

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def fsdp_example(rank, world_size):
    setup(rank, world_size)
    print(f"Rank {rank}: Model initialized on process {os.getpid()}", flush=True)
    model = FSDP(Linear(10, 10).cuda(rank))
    print(f"Rank {rank} initialized FSDP model.")
    dist.destroy_process_group()

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    fsdp_example(rank, world_size)
