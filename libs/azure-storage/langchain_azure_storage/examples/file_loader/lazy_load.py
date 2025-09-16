import os
from langchain_azure_storage import AzureBlobStorageFileLoader

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


def lazy_load_from_file_loader():
    loader = AzureBlobStorageFileLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        "test_json.json", # replace with your blob name
    )
    for doc in loader.lazy_load():
        print(doc.page_content)

if __name__ == "__main__":
    lazy_load_from_file_loader()
  