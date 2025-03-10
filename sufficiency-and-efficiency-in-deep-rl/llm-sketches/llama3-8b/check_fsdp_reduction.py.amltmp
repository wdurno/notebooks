import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import wrap
from torch.nn.parallel import DistributedDataParallel as DDP

# Set master address and port (if not already set)
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # Enable debug logging

def setup(rank, world_size):
    """ Initialize Process Group """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Ensure each process uses the correct GPU

def cleanup():
    """ Destroy Process Group """
    dist.destroy_process_group()

class SimpleModel(torch.nn.Module):
    """ A small model for testing communication """
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.fc(x)

def fsdp_reduction(rank, world_size):
    """ FSDP Test with Reduction """
    setup(rank, world_size)

    # Create a model and move it to GPU
    model = SimpleModel().cuda(rank)

    # Wrap model in FSDP
    model = FSDP(model, use_orig_params=True)

    # Create a tensor to reduce
    tensor = torch.tensor([rank + 1.0], device=f"cuda:{rank}")

    # Sync before reduction
    dist.barrier()
    print(f"Rank {rank}: Before all_reduce, tensor = {tensor.item()}")

    # Perform all_reduce (SUM operation)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Sync after reduction
    dist.barrier()
    print(f"Rank {rank}: After all_reduce, tensor = {tensor.item()}")

    cleanup()

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    fsdp_reduction(rank, world_size)
