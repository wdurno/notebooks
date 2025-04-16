import os
import sys
from azure.storage.blob import BlobServiceClient, ContentSettings
from pathlib import Path

# Load from environment
account_name = os.environ.get("AZURE_STORAGE_ACCOUNT")
container_name = os.environ.get("AZURE_CONTAINER_NAME")
connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
prefix = os.environ.get("AZURE_BLOB_MODEL_PREFIX", "llm-uploads")

if not all([account_name, container_name, connection_string]):
    print("Missing one or more environment variables. Run `source llm_keys.sh` first.")
    sys.exit(1)

# Models to upload
local_model_paths = {
    "model_v0_full": Path("models/model_v0_full"),
    "model_v0_quantized": Path("models/model_v0_quantized")
}

# Connect to blob service
blob_service = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service.get_container_client(container_name)

# Upload function
def upload_dir(local_dir: Path, remote_dir: str):
    for path in local_dir.rglob("*"):
        if path.is_file():
            blob_name = f"{remote_dir}/{path.relative_to(local_dir)}"
            print(f"Uploading: {path} → {blob_name}")
            with open(path, "rb") as f:
                container_client.upload_blob(
                    name=blob_name,
                    data=f,
                    overwrite=True,
                    content_settings=ContentSettings(content_type="application/octet-stream"),
                )

# Main upload loop
for model_name, model_path in local_model_paths.items():
    remote_dir = f"{prefix}/{model_name}"
    upload_dir(model_path, remote_dir)

print("✅ Upload complete.")

