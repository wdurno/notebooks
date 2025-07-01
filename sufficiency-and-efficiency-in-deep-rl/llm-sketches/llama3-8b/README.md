# 🤖 LLM Model Upload Toolkit

This repo includes tools for uploading large language models (LLMs) to Azure Blob Storage — including full and quantized variants.

## 🧰 Contents

- `llm_keys.sh`: Template to configure secure Azure credentials
- `upload_models.py`: Python script to upload full and quantized models
- `models/`: Local directory where models are stored
  - `model_v0_full/`
  - `model_v0_quantized/`
  - ... 

## ✅ Quickstart

### 0. Fill in your credentials

Copy and edit `llm_keys.sh` with your Azure Blob credentials:

```bash
cp llm_keys.sh ~/llm_keys.sh
nano ~/llm_keys.sh  # or use your favorite editor
```

NEVER POPULATE `llm_keys.sh` IN THE REPO!
Otherwise, you risk running `git commit` on a key file.

### 1. Initialize Model 

This must run on a GPU node. 

```bash
. ~/llm_keys.sh ## use keys 
bash 1_init_fsdp_llama_8b.sh 
```

### 2. Generate text 

This can run on a cheaper node. 

```bash 
## TODO re-write to use `chat_pack.py` 
. ~/llm_keys.sh ## use keys 
python3 2_generate_data.py  
```

### 3. Update model 

This must run on a GPU node. 

```bash
## set keys 
. ~/llm_keys.sh ## use keys 
bash 3_fit.sh \
  --model-checkpoint models/model_v0_full \
  --output-dir models/model_v1 \
  --data-path chat_logs/replay_buffer.pt
```
