import os
import asyncio
from langchain_azure_storage import AzureBlobStorageFileLoader

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


async def alazy_load_from_file_loader():
    loader = AzureBlobStorageFileLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        "test_csv.csv", # replace with your blob name
    )
    async for doc in loader.alazy_load():
        print(doc.page_content)


if __name__ == "__main__":
    asyncio.run(alazy_load_from_file_loader())