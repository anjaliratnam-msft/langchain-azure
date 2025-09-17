import os
from langchain_azure_storage import AzureBlobStorageLoader
from langchain_community.document_loaders import JSONLoader

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


def json_loader_factory(file_path: str) -> JSONLoader:
    return JSONLoader(file_path=file_path, jq_schema=".[]", content_key="text")

def custom_loader_factory_for_storage_loader():
    loader = AzureBlobStorageLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        ["test_json.json", "test_csv.csv", "pdf_test.pdf"], # replace with your list of blobs
        "test_json", # replace with your blob prefix
        loader_factory=json_loader_factory
    )
    for doc in loader.lazy_load():
        print(doc.page_content)


if __name__ == "__main__":
    custom_loader_factory_for_storage_loader()