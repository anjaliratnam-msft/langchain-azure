import os
import tempfile
import azure.identity
import azure.identity.aio
import azure.core.credentials
import azure.core.credentials_async
from typing import Optional, Union, Callable, Iterator, AsyncIterator, Iterable
from azure.storage.blob import BlobClient, ContainerClient
from langchain_core.document_loaders import BaseLoader, BaseBlobParser
from langchain_core.documents.base import Document, Blob
from azure.storage.blob.aio import BlobClient as AsyncBlobClient
from azure.storage.blob.aio import ContainerClient as AsyncContainerClient



SDK_CREDENTIAL_TYPE = Optional[
    Union[
        azure.core.credentials.AzureSasCredential,
        azure.core.credentials.TokenCredential,
        azure.core.credentials_async.AsyncTokenCredential,
    ]
]

def _get_client_kwargs(credential=None) -> dict:
    return {"credential": credential, "connection_data_block_size": 256 * 1024}


def _lazy_load_documents_from_blob(
    blob_client: BlobClient, parser: Optional[BaseBlobParser] = None
) -> Iterator[Document]:
    blob_data = blob_client.download_blob(max_concurrency=10)
    blob = Blob.from_data(
        data=blob_data.readall(),
        mime_type=blob_data.properties.content_settings.content_type,
        metadata={
            "source": blob_client.url,
        },
    )
    if parser:
        yield from parser.lazy_parse(blob)
    else:
        yield Document(page_content=blob.as_string(), metadata=blob.metadata)


async def _alazy_load_documents_from_blob(
    async_blob_client: AsyncBlobClient, parser: Optional[BaseBlobParser] = None
) -> AsyncIterator[Document]:
    blob_data = await async_blob_client.download_blob(max_concurrency=10)
    blob = Blob.from_data(
        data= await blob_data.readall(),
        mime_type=blob_data.properties.content_settings.content_type,
        metadata={
            "source": async_blob_client.url,
        },
    )
    if parser:
        for doc in parser.lazy_parse(blob):
            yield doc
    else:
        yield Document(page_content=blob.as_string(), metadata=blob.metadata)


def _download_blob_to_file(blob_client: BlobClient, file_path: str) -> None:
    with open(file_path, "wb") as file:
        download_stream = blob_client.download_blob()
        file.write(download_stream.readall())


def _lazy_load_from_custom_loader(blob_client: BlobClient, loader_factory) -> Iterator[Document]:
    with tempfile.NamedTemporaryFile() as temp_file:
        file_path = temp_file.name
     
    _download_blob_to_file(blob_client, file_path)
    loader = loader_factory(temp_file.name)
    for doc in loader.lazy_load():
        doc.metadata["source"] = blob_client.url
        yield doc
    os.remove(file_path)


async def _adownload_blob_to_file(blob_client, file_path: str) -> None:
    blob_data = await blob_client.download_blob()
    data = await blob_data.readall()
    with open(file_path, "wb") as file:
        file.write(data)


async def _alazy_load_from_custom_loader(blob_client, loader_factory) -> AsyncIterator[Document]:
    with tempfile.NamedTemporaryFile() as temp_file:
        file_path = temp_file.name
       
    await _adownload_blob_to_file(blob_client, file_path)
    loader = loader_factory(temp_file.name)
    async for doc in loader.alazy_load():
        doc.metadata["source"] = blob_client.url
        yield doc
    os.remove(file_path)


def _get_sdk_credential(credential: SDK_CREDENTIAL_TYPE) -> SDK_CREDENTIAL_TYPE:
    if credential is None:
        return {
            "sync": azure.identity.DefaultAzureCredential(), 
            "async": azure.identity.aio.DefaultAzureCredential(),
        }
    if isinstance(credential, (
        azure.core.credentials.TokenCredential, 
        azure.core.credentials.AzureSasCredential, 
        azure.core.credentials_async.AsyncTokenCredential
    )):
        return credential
    raise TypeError("Invalid credential type provided.")

class AzureBlobStorageLoader(BaseLoader):
    def __init__(
            self, 
            account_url: str,
            container_name: str,
            blob_names: Optional[Union[str, Iterable[str]]] = None,
            prefix: str = "",
            credential: Optional[
                Union[
                    azure.core.credentials.AzureSasCredential,
                    azure.core.credentials.TokenCredential,
                    azure.core.credentials_async.AsyncTokenCredential,
                ]
            ] = None,
            loader_factory: Optional[Callable[str, BaseLoader]] = None,
    ):
        self._account_url = account_url
        self._container_name = container_name
        self._blob_names = blob_names
        self._prefix = prefix
        self._credential = credential
        self._loader_factory = loader_factory

    def lazy_load(self) -> Iterator[Document]:
        self._credential = _get_sdk_credential(self._credential)
        if isinstance(self._credential, azure.core.credentials_async.AsyncTokenCredential):
            raise ValueError("Async credential provided to sync method.")
        container_url = f"{self._account_url}/{self._container_name}"
        container_client = ContainerClient.from_container_url(
            container_url, **_get_client_kwargs(self._credential["sync"] if isinstance(self._credential, dict) else self._credential)
        )
        blobs_with_prefix = list(container_client.list_blob_names(name_starts_with=self._prefix))
        for blob_name in self._blob_names:
            if blob_name in blobs_with_prefix:
                blob_client = container_client.get_blob_client(blob_name)
                if self._loader_factory:
                    yield from _lazy_load_from_custom_loader(blob_client, self._loader_factory)
                else:
                    yield from _lazy_load_documents_from_blob(blob_client, self._loader_factory)

    async def alazy_load(self) -> AsyncIterator[Document]:
        self._credential = _get_sdk_credential(self._credential)
        if isinstance(self._credential, (azure.core.credentials.TokenCredential, azure.core.credentials.AzureSasCredential)):
            raise ValueError("Sync credential provided to async method.")
        container_url = f"{self._account_url}/{self._container_name}"
        async with AsyncContainerClient.from_container_url(
            container_url, **_get_client_kwargs(self._credential["async"] if isinstance(self._credential, dict) else self._credential)
        ) as container_client:
            blobs_with_prefix = []
            async for blob_name in container_client.list_blob_names(name_starts_with=self._prefix):
                blobs_with_prefix.append(blob_name)
            for blob_name in self._blob_names:
                if blob_name in blobs_with_prefix:
                    blob_client = container_client.get_blob_client(blob_name)
                    if self._loader_factory:
                        async for doc in _alazy_load_from_custom_loader(blob_client, self._loader_factory):
                            yield doc
                    else: 
                        async for doc in _alazy_load_documents_from_blob(blob_client, self._loader_factory):
                            yield doc
            if self._credential and isinstance(self._credential, dict):
                await self._credential["async"].close()
