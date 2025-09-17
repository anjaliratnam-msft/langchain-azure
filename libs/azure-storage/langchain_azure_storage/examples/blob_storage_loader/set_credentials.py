import os
from langchain_azure_storage import AzureBlobStorageLoader
from azure.core.credentials import AzureSasCredential

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
_CREDENTIAL = AzureSasCredential(os.getenv("AZURE_STORAGE_SAS_TOKEN"))


def set_credential_for_storage_loader():
    loader = AzureBlobStorageLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        ["test_json.json", "test_csv.csv", "pdf_test.pdf"], # replace with your list of blobs
        "test", # replace with your blob prefix
        _CREDENTIAL,
    )
    for doc in loader.lazy_load():
        print(doc.page_content)


if __name__ == "__main__":
    set_credential_for_storage_loader()