import os
import asyncio
from langchain_azure_storage import AzureBlobStorageLoader

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


async def alazy_load_from_storage_loader():
    loader = AzureBlobStorageLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        ["test_json.json", "test_csv.csv", "pdf_test.pdf"], # replace with your list of blobs
        "test", # replace with your blob prefix
    )
    async for doc in loader.alazy_load():
        print(doc.page_content)


if __name__ == "__main__":
    asyncio.run(alazy_load_from_storage_loader())