import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp.wrap import wrap
from torch.nn.parallel import DistributedDataParallel as DDP

from fsdp_ssr_llama_8b import FsdpSsrLlama8B 

## set master address and port (if not already set)
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # enable debug logging

def setup(rank, world_size):
    """ Initialize Process Group """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  ## ensure each process uses the correct GPU

def cleanup():
    """ Destroy Process Group """
    dist.destroy_process_group()

def fsdp_reduction(rank, world_size):
    """ FSDP Test with Reduction """
    setup(rank, world_size)

    ## create a model and move it to GPU
    model = FsdpSsrLlama8B() 

    ## sync before gathering and writing quantized model 
    dist.barrier()

    ## write quantized 
    model.save_quantized('model_v0_quantized')

    ## write full model 
    model.save('model_v0_full') 

    cleanup()
    pass 

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    fsdp_reduction(rank, world_size)
