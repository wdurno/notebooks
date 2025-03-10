import torch
import torch.distributed as dist
import os
import psutil
import time

# Set these before initializing torch.distributed
os.environ["MASTER_ADDR"] = "0.0.0.0"
os.environ["MASTER_PORT"] = "29500"
os.environ["OMP_NUM_THREADS"] = "2"
# Find best Gloo network interface  
interfaces = psutil.net_if_addrs()
valid_interfaces = [iface for iface in interfaces.keys() if iface not in ["lo", "docker0"]]
if len(valid_interfaces) > 0: 
    interface = valid_interfaces[0] 
else: 
    interface = list(interfaces.keys())[0] 
os.environ["GLOO_SOCKET_IFNAME"] = str(interface)

def all_reduce_example(rank, world_size):
    """Simple all_reduce example across multiple CPU processes."""
    # Avoid rank 0 reuse 
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Allow each process to be ready 
    if rank != 0:
        time.sleep(2) 

    # Initialize the process group
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # sync before any reductions 
    print(f"Rank {rank} initialized and waiting at barrier.", flush=True) 
    dist.barrier()
    print(f"✅ Rank {rank} successfully passed the barrier.", flush=True)
    
    # Each process starts with a tensor of its rank
    tensor = torch.tensor([rank], dtype=torch.float32)
    
    print(f"Before all_reduce - Rank {rank}: {tensor.item()}")
    
    # Perform all_reduce (sum operation)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"After all_reduce - Rank {rank}: {tensor.item()}")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver", force=True) 
    world_size = 2  # Number of processes
    torch.multiprocessing.spawn(all_reduce_example, args=(world_size,), nprocs=world_size, join=True)
