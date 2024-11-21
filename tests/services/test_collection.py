import pytest
from pathlib import Path
import shutil
import tempfile
import os

from minirag.services.collection_service import CollectionService
from minirag.models import Chunk

class TestCollectionService:
    @pytest.fixture
    def temp_dir(self):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after tests
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def collection_service(self, temp_dir):
        return CollectionService(storage_path=temp_dir)

    @pytest.fixture
    def sample_text_files(self, temp_dir):
        # Create sample text files for testing
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        
        file1_path = docs_dir / "test1.txt"
        file2_path = docs_dir / "test2.txt"
        
        file1_path.write_text("This is test document 1.")
        file2_path.write_text("This is test document 2.")
        
        return [str(file1_path), str(file2_path), str(docs_dir)]

    def test_init(self, collection_service):
        assert collection_service.active_collection is None
        assert isinstance(collection_service.storage_path, Path)

    def test_process_folder(self, collection_service, sample_text_files):
        docs_dir = sample_text_files[-1]
        chunks = collection_service._process_folder(docs_dir)
        
        assert isinstance(chunks, list)
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert len(chunks) > 0

    def test_create_and_load_collection(self, collection_service, sample_text_files):
        # Test creating collection
        collection_name = "test_collection"
        collection_service.create_collection(
            doc_paths=sample_text_files[:-1],  # Exclude the directory
            collection_name=collection_name
        )
        
        # Verify file exists
        collection_file = Path(collection_service.storage_path) / f"{collection_name}.npy"
        assert collection_file.exists()
        
        # Test loading collection
        collection_service.load_collection(collection_name)
        assert collection_service.active_collection is not None
        assert isinstance(collection_service.active_collection, list)
        assert all(isinstance(chunk, Chunk) for chunk in collection_service.active_collection)

    def test_load_nonexistent_collection(self, collection_service):
        collection_service.load_collection("nonexistent")
        assert collection_service.active_collection is None

    def test_list_collections(self, collection_service, sample_text_files):
        # Create a few collections first
        collection_service.create_collection(
            doc_paths=[sample_text_files[0]], 
            collection_name="test1"
        )
        collection_service.create_collection(
            doc_paths=[sample_text_files[1]], 
            collection_name="test2"
        )
        
        # Capture stdout to verify the output
        collections = [f.split(".")[0] for f in os.listdir(collection_service.storage_path)]
        assert "test1" in collections
        assert "test2" in collections