import os
from langchain_azure_storage import AzureBlobStorageFileLoader
from langchain_community.document_loaders import CSVLoader

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


def custom_loader_factory_for_file_loader():
    loader = AzureBlobStorageFileLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        "test_csv.csv", # replace with your blob name
        loader_factory=CSVLoader,
    )
    for doc in loader.lazy_load():
        print(doc.page_content)


if __name__ == "__main__":
    custom_loader_factory_for_file_loader()
