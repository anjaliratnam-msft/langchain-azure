import os
import asyncio
from langchain_azure_storage import AzureBlobStorageContainerLoader

_ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")


async def alazy_load_from_container_loader():
    loader = AzureBlobStorageContainerLoader(
        _ACCOUNT_URL,
        _CONTAINER_NAME,
        "test", # replace with your blob prefix
    )
    async for doc in loader.alazy_load():
        print(doc.page_content)


if __name__ == "__main__":
    asyncio.run(alazy_load_from_container_loader())