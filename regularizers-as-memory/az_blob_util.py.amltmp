from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def upload_to_blob_store(data_as_bytes, remote_file_path, connect_str, container_name):
    '''
    data_as_bytes: upload these bytes as blob
    remote_file_path: remote destination in storage container
    connect_str: connection string key
    container_name: Azure storage container
    '''
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=remote_file_path)
    # Upload the created file
    blob_client.upload_blob(data_as_bytes, overwrite=True)
    pass
