from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

def create_container(connect_str, container_name):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_service_client.create_container(name=container_name)
    pass

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

def download_from_blob_store(remote_file_path, connect_str, container_name):
    '''
    remote_file_path: remote destination in storage container
    connect_str: connection string key
    container_name: Azure storage container
    '''
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=remote_file_path)
    # Upload the created file
    return blob_client.download_blob().readall() 

def ls_blob_store(remote_file_path, connect_str, container_name):
    '''
    remote_file_path: path prefix for listing files 
    connect_str: connection string key
    container_name: Azure storage container
    '''
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name) 
    # Upload the created file
    blobs = container_client.list_blobs(name_starts_with=remote_file_path) 
    out = [] 
    for blob in blobs: 
        out.append(blob['name']) 
        pass 
    return out 
