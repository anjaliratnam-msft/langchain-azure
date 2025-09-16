import os
from langchain_azure_storage import AzureBlobStorageContainerLoader

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


def lazy_load_from_container_loader():
    loader = AzureBlobStorageContainerLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        "test", # replace with your blob prefix
    )
    for doc in loader.lazy_load():
        print(doc.page_content)


if __name__ == "__main__":
    lazy_load_from_container_loader()
  