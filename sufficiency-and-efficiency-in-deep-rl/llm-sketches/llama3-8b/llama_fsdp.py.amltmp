## This code initially drafted by ChatGPT o1 

import os
import torch
import torch.nn as nn
import torch.distributed as dist

# FSDP imports:
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ------------------------------------------------
# 1) Custom regularizer example
# ------------------------------------------------
def custom_regularizer(model, factor=0.01):
    """
    Example L2 penalty. Replace with your advanced approach.
    """
    reg_loss = 0.0
    for param in model.parameters():
        # Make sure we skip any non-float parameter
        if param.requires_grad and param.dtype in (torch.float16, torch.float32, torch.bfloat16):
            reg_loss += torch.sum(param**2)
    return factor * reg_loss


# ------------------------------------------------
# 2) FSDP Fine-tuning function
# ------------------------------------------------
def train_llama_3_8b(
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    output_dir: str = "outputs",
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 1e-5,
    regularizer_factor: float = 0.01,
    max_seq_length: int = 512,
    save_every: int = 500,
):
    """
    Fine-tune the Llama 3 8B model on a small dataset, with FSDP + custom regularizer.
    """

    # 1. Initialize distributed backend
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    # 2. Create tokenizer
    #    According to the Llama 3 model card, you typically use LlamaTokenizer.
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Some older Llama versions do not define a pad token; safely set it
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Load model config + model
    config = AutoConfig.from_pretrained(model_name)

    # If the model card recommends a specific max sequence length, you can do:
    # config.max_position_embeddings = 4096   # or whatever is recommended

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,  # or "auto"
        device_map=None,           # We'll let FSDP handle distribution
        low_cpu_mem_usage=True,    # Helps reduce CPU memory footprint
    )

    # 4. Wrap with FSDP for memory-efficient training
    auto_wrap_policy_config = transformer_auto_wrap_policy(
        # Llama uses internally stacked decoder layers. 
        # Some relevant HF classes might be "LlamaDecoderLayer".
        transformer_layer_cls={
            # For official Llama from Hugging Face:
            # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer
        }
    )

    fsdp_config = dict(
        mixed_precision=True,             # or set MixedPrecision object for finer control
        auto_wrap_policy=auto_wrap_policy_config,
        # For large models, you can also tweak:
        # sharding_strategy=..., 
        # use_orig_params=..., 
        # etc.
    )
    model = FSDP(model, **fsdp_config).cuda()

    # 5. Prepare a small dataset (e.g., wikitext for demonstration)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_seq_length, 
            padding="max_length"
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Distributed sampler ensures each GPU sees a unique subset
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=batch_size, 
        drop_last=True
    )

    # 6. Optimizer + LR scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 7. Train
    global_step = 0
    model.train()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # ensures a different shuffle each epoch
        for batch in dataloader:
            global_step += 1

            # Move data to GPU
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            # Llama uses standard causal LM training: labels = input_ids
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            lm_loss = outputs.loss

            # Add your custom regularizer
            reg_loss = custom_regularizer(model, factor=regularizer_factor)
            total_loss = lm_loss + reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Simple logging
            if (global_step % 100 == 0) and (local_rank == 0):
                print(f"[Epoch {epoch}] step={global_step} | LM loss={lm_loss.item():.4f} | Reg loss={reg_loss.item():.4f}")

            # Optionally save every N steps
            if save_every > 0 and (global_step % save_every == 0) and (local_rank == 0):
                _save_checkpoint(model, output_dir, f"step_{global_step}")

    # 8. Final Save
    if local_rank == 0:
        _save_checkpoint(model, output_dir, "final")
    dist.barrier()


# ------------------------------------------------
# 3) Helper function to consolidate and save
# ------------------------------------------------
def _save_checkpoint(model, output_dir, tag="checkpoint"):
    """
    With FSDP, we must use the FSDP context to get a full, consolidated state_dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    full_sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    # Consolidate the full model weights on rank 0 CPU
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_sd_config):
        state_dict = model.state_dict()
    # Save
    ckpt_path = os.path.join(output_dir, f"llama3-8b-{tag}.pt")
    torch.save(state_dict, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")


# ------------------------------------------------
# 4) Script entry point
# ------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--regularizer_factor", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--save_every", type=int, default=500)
    args = parser.parse_args()

    train_llama_3_8b(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        regularizer_factor=args.regularizer_factor,
        max_seq_length=args.max_seq_length,
        save_every=args.save_every,
    )
