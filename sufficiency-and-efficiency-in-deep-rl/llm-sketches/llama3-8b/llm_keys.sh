#!/bin/bash

# Azure Blob Storage Access Keys
export AZURE_STORAGE_ACCOUNT="<your-storage-account-name>"
export AZURE_CONTAINER_NAME="<your-container-name>"
export AZURE_STORAGE_CONNECTION_STRING="<your-full-connection-string>"

# Optional: subdirectory to store models (e.g., versioned folder)
export AZURE_BLOB_MODEL_PREFIX="llm-uploads"

echo "Azure keys loaded."
