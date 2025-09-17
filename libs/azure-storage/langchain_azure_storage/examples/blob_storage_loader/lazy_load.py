import os
from langchain_azure_storage import AzureBlobStorageLoader

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


def lazy_load_from_storage_loader():
    loader = AzureBlobStorageLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        ["test_json.json", "test_csv.csv", "pdf_test.pdf"], # replace with your list of blobs
        "test", # replace with your blob prefix
    )
    for doc in loader.lazy_load():
        print(doc.page_content)


if __name__ == "__main__":
    lazy_load_from_storage_loader()