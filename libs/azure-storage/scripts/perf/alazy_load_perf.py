import asyncio
import gc
import os
import time
import tracemalloc
from typing import Any

from azure.core.credentials import AzureSasCredential
from azure.storage.blob import BlobServiceClient

from langchain_azure_storage.document_loaders import AzureBlobStorageLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader

try:
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# Configuration
ACCOUNT_URL = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
CONTAINER_NAME = "perf-test-container-64"
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
SAS_CRED = AzureSasCredential(os.getenv("AZURE_STORAGE_SAS_TOKEN"))
NUM_BLOBS = 10000
BLOB_SIZE_KB = 64
BLOB_PREFIX = "perf-test-blob-"


def create_test_data() -> bytes:
    return b"0" * (BLOB_SIZE_KB * 1024)


def setup_test_container() -> BlobServiceClient:
    """Create test container and upload test blobs."""
    if not ACCOUNT_URL:
        raise ValueError("AZURE_STORAGE_ACCOUNT_URL environment variable must be set")
    
    print(f"Setting up test container '{CONTAINER_NAME}'...")
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL, credential=SAS_CRED
    )
    
    # Create container
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    try:
        container_client.create_container()
        print(f"Created container '{CONTAINER_NAME}'")
    except Exception as e:
        print(f"Container already exists or error: {e}")
    
    # Check if blobs already exist
    blob_list = list(container_client.list_blobs(name_starts_with=BLOB_PREFIX))
    if len(blob_list) >= NUM_BLOBS:
        print(f"Found {len(blob_list)} existing test blobs, skipping upload...")
        return blob_service_client
    
    # Upload test blobs
    print(f"Uploading {NUM_BLOBS} blobs of {BLOB_SIZE_KB} KiB each...")
    test_data = create_test_data()
    
    start_time = time.time()
    for i in range(NUM_BLOBS):
        blob_name = f"{BLOB_PREFIX}{i}.txt"
        blob_client = container_client.get_blob_client(blob_name)
        try:
            blob_client.upload_blob(test_data)
        except Exception:
            pass  # Blob already exists, skip it
        
        if (i + 1) % 1000 == 0:
            print(f"  Uploaded {i + 1}/{NUM_BLOBS} blobs...")
    
    elapsed = time.time() - start_time
    print(f"Upload completed in {elapsed:.2f} seconds")
    
    return blob_service_client


def cleanup_test_container(blob_service_client: BlobServiceClient) -> None:
    """Delete test container and all blobs."""
    print(f"\nCleaning up test container '{CONTAINER_NAME}'...")
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    try:
        container_client.delete_container()
        print(f"Deleted container '{CONTAINER_NAME}'")
    except Exception as e:
        print(f"Error deleting container: {e}")


async def test_new_loader_alazy_load(use_loader_factory: bool = False) -> None:
    """Test the new AzureBlobStorageLoader with alazy_load."""
    print("\n" + "=" * 80)
    print("Testing NEW AzureBlobStorageLoader - alazy_load()")
    print("=" * 80)

    gc.collect()
    tracemalloc.start()
    
    if use_loader_factory:
        loader = loader_with_factory()
    else:
        loader = loader_without_factory()
    
    start_time = time.time()
    doc_count = 0
    peak_memory = 0
    
    async for doc in loader.alazy_load():
        doc_count += 1
        if doc_count % 1000 == 0:
            current_memory, peak = tracemalloc.get_traced_memory()
            peak_memory = max(peak_memory, peak)
            print(f"  Loaded {doc_count} documents... (current memory: {current_memory / 1024 / 1024:.2f} MB)")
    
    current_memory, peak = tracemalloc.get_traced_memory()
    peak_memory = max(peak_memory, peak)
    
    tracemalloc.stop()
    elapsed = time.time() - start_time
    
    print(f"\nCompleted:")
    print(f"  Documents loaded: {doc_count}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Throughput: {doc_count / elapsed:.2f} docs/sec")
    print(f"  Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory per doc: {peak_memory / doc_count / 1024:.2f} KB")
    

def loader_without_factory() -> AzureBlobStorageLoader:
    return AzureBlobStorageLoader(
        account_url=ACCOUNT_URL,
        container_name=CONTAINER_NAME,
        credential=AzureSasCredential(os.getenv("AZURE_STORAGE_SAS_TOKEN")),
    )

def loader_with_factory() -> AzureBlobStorageLoader:
    return AzureBlobStorageLoader(
        account_url=ACCOUNT_URL,
        container_name=CONTAINER_NAME,
        credential=AzureSasCredential(os.getenv("AZURE_STORAGE_SAS_TOKEN")),
        loader_factory=UnstructuredFileLoader,
    )



async def test_new_loader_alazy_load_gather(use_loader_factory: bool = False, batch_size: int = 1000) -> None:
    """Test the new AzureBlobStorageLoader with alazy_load using asyncio.gather on batches."""
    print("\n" + "=" * 80)
    print(f"Testing NEW AzureBlobStorageLoader - alazy_load() with asyncio.gather (batch_size={batch_size})")
    print("=" * 80)

    gc.collect()
    tracemalloc.start()
    
    # Generate all blob names
    all_blob_names = [f"{BLOB_PREFIX}{i}.txt" for i in range(NUM_BLOBS)]
    
    # Split into batches
    batches = [all_blob_names[i:i + batch_size] for i in range(0, len(all_blob_names), batch_size)]
    
    async def process_batch(blob_names_batch: list[str]) -> int:
        """Process one batch of blobs and return document count."""
        if use_loader_factory:
            loader = AzureBlobStorageLoader(
                account_url=ACCOUNT_URL,
                container_name=CONTAINER_NAME,
                blob_names=blob_names_batch,
                credential=AzureSasCredential(os.getenv("AZURE_STORAGE_SAS_TOKEN")),
                loader_factory=UnstructuredFileLoader,
            )
        else:
            loader = AzureBlobStorageLoader(
                account_url=ACCOUNT_URL,
                container_name=CONTAINER_NAME,
                blob_names=blob_names_batch,
                credential=AzureSasCredential(os.getenv("AZURE_STORAGE_SAS_TOKEN")),
            )
        
        count = 0
        async for doc in loader.alazy_load():
            count += 1
        return count
    
    start_time = time.time()
    peak_memory = 0
    
    tasks = [process_batch(batch) for batch in batches]
    
    batch_counts = await asyncio.gather(*tasks)
    doc_count = sum(batch_counts)
    
    elapsed = time.time() - start_time
    
    current_memory, peak = tracemalloc.get_traced_memory()
    peak_memory = max(peak_memory, peak)
    tracemalloc.stop()
    
    print(f"\nCompleted:")
    print(f"  Documents loaded: {doc_count}")
    print(f"  Batches processed: {len(batches)}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Throughput: {doc_count / elapsed:.2f} docs/sec")
    print(f"  Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory per doc: {peak_memory / doc_count / 1024:.2f} KB")


async def test_community_loader_alazy_load() -> None:
    """Test the legacy AzureBlobStorageContainerLoader with aload."""
    if not CONNECTION_STRING:
        print("\n" + "=" * 80)
        print("Skipping AzureBlobStorageContainerLoader - CONNECTION_STRING not set")
        print("=" * 80)
        return {
            "loader": "AzureBlobStorageContainerLoader",
            "method": "aload",
            "doc_count": 0,
            "elapsed_time": 0,
            "throughput": 0,
            "skipped": True,
        }
    
    from langchain_community.document_loaders import AzureBlobStorageContainerLoader
   
    
    print("\n" + "=" * 80)
    print("Testing LEGACY AzureBlobStorageContainerLoader - aload()")
    print("=" * 80)

    gc.collect()
    tracemalloc.start()
    
    loader = AzureBlobStorageContainerLoader(
        conn_str=CONNECTION_STRING,
        container=CONTAINER_NAME,
    )
    
    start_time = time.time()
    
    docs = await loader.aload()
    doc_count = len(docs)
    
    elapsed = time.time() - start_time
    
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nCompleted:")
    print(f"  Documents loaded: {doc_count}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Throughput: {doc_count / elapsed:.2f} docs/sec")
    print(f"  Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory per doc: {peak_memory / doc_count / 1024:.2f} KB")


async def test_community_loader_alazy_load_gather() -> None:
    """Test community loader using digit-based batching."""
    from langchain_community.document_loaders import AzureBlobStorageContainerLoader
    
    print("\n" + "=" * 80)
    print("Testing LEGACY AzureBlobStorageContainerLoader - aload() with digit batching (~1000 per batch)")
    print("=" * 80)

    gc.collect()
    tracemalloc.start()
    
    async def process_digit_batch(digit: int) -> int:
        """Process blobs starting with a specific digit."""
        prefix = f"{BLOB_PREFIX}{digit}"
        
        loader = AzureBlobStorageContainerLoader(
            conn_str=CONNECTION_STRING,
            container=CONTAINER_NAME,
            prefix=prefix,
        )
        
        docs = await loader.aload()
        return len(docs)
    
    start_time = time.time()
    
    # Create 10 digit-based batches (0-9)
    tasks = [process_digit_batch(digit) for digit in range(10)]
    
    batch_counts = await asyncio.gather(*tasks)
    doc_count = sum(batch_counts)
    
    elapsed = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nCompleted:")
    print(f"  Documents loaded: {doc_count}")
    print(f"  Digit batches processed: 10")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Throughput: {doc_count / elapsed:.2f} docs/sec")
    print(f"  Peak memory: {peak_memory / 1024 / 1024:.2f} MB")
    print(f"  Memory per doc: {peak_memory / doc_count / 1024:.2f} KB")


async def main() -> None:
    """Run asynchronous performance tests."""
    print("Azure Blob Storage Document Loader - ASYNCHRONOUS Performance Test")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Number of blobs: {NUM_BLOBS}")
    print(f"  Blob size: {BLOB_SIZE_KB} KiB")
    
    blob_service_client = setup_test_container() 
    try:
        # Test new loader - async
        for i in range(3):
            await test_new_loader_alazy_load(use_loader_factory=False)

        for i in range(3):
            await test_new_loader_alazy_load(use_loader_factory=True)
        
        # Test community loader - async
        for i in range(3):
            await test_community_loader_alazy_load()


        # for i in range(3):
        #     await test_new_loader_alazy_load_gather(use_loader_factory=False, batch_size=1000)

        # for i in range(3):
        #     await test_new_loader_alazy_load_gather(use_loader_factory=True, batch_size=1000)

        # for i in range(3):
        #     await test_community_loader_alazy_load_gather()
        
    finally:
        # Cleanup (optional - comment out if you want to keep data for testing)
        print(f"\nNote: Test container '{CONTAINER_NAME}' was kept for testing.")


if __name__ == "__main__":
    asyncio.run(main())