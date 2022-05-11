from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

connect_str = '[REDACTED]' ## todo: always cycle keys 
local_file_name = 'work/fake-data.txt'
remote_file_path = 'fake-data.txt'
container_name = 'data'

def upload_to_blob_store(data_as_bytes, remote_file_name, connect_str, container_name):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str) 
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=remote_file_path) 
    # Upload the created file
    blob_client.upload_blob(data_as_bytes) 
    pass

