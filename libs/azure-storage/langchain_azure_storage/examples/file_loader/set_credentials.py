import os
from langchain_azure_storage import AzureBlobStorageFileLoader, AzureBlobStorageContainerLoader
from azure.core.credentials import AzureSasCredential

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
_CREDENTIAL = AzureSasCredential(os.getenv("AZURE_STORAGE_SAS_TOKEN"))


def set_credential_for_file_loader():
    loader = AzureBlobStorageFileLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        "test_csv.csv", # replace with your blob name
        _CREDENTIAL,
    )
    for doc in loader.lazy_load():
        print(doc.page_content)


if __name__ == "__main__":
    set_credential_for_file_loader()