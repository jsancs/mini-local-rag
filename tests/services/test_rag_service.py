import pytest
from minirag.services.rag_service import RagService
from langchain_text_splitters import RecursiveCharacterTextSplitter

from minirag.models import Chunk


class TestRagService:
    def test_get_splitter(self):
        # Test with default parameters
        splitter = RagService.get_splitter()
        assert isinstance(splitter, RecursiveCharacterTextSplitter)
        assert splitter._chunk_size == 1000
        assert splitter._chunk_overlap == 20

        # Test with custom parameters
        custom_splitter = RagService.get_splitter(chunk_size=500, chunk_overlap=50)
        assert isinstance(custom_splitter, RecursiveCharacterTextSplitter)
        assert custom_splitter._chunk_size == 500
        assert custom_splitter._chunk_overlap == 50


    def test_generate_embeddings(self, mocker):
        # Mock ollama.embeddings
        mock_embeddings = mocker.patch('ollama.embeddings')
        mock_embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        # Test with default parameters
        text = "This is a test text"
        embeddings = RagService.generate_embeddings(text)
        
        # Assert embeddings are returned correctly
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        assert embeddings == [0.1, 0.2, 0.3]
        
        # Verify ollama.embeddings was called with correct parameters
        mock_embeddings.assert_called_once_with(
            model="all-minilm",
            prompt="This is a test text"
        )

        # Test with custom model
        mock_embeddings.reset_mock()
        embeddings = RagService.generate_embeddings(text, model_name="custom-model")
        
        # Verify ollama.embeddings was called with custom model
        mock_embeddings.assert_called_once_with(
            model="custom-model",
            prompt="This is a test text"
        )
        

    def test_similarity_search(self, mocker):
        # Mock generate_embeddings
        mock_generate_embeddings = mocker.patch.object(
            RagService, 'generate_embeddings',
            return_value=[1.0, 0.0, 0.0]  # Query embedding
        )

        # Create test chunks with different embeddings
        chunks = [
            Chunk(document_name="test_doc.txt", text="First chunk", embedding=[0.8, 0.1, 0.1]),    # Similarity: 0.8
            Chunk(document_name="test_doc.txt", text="Second chunk", embedding=[0.5, 0.5, 0.0]),   # Similarity: 0.5
            Chunk(document_name="test_doc.txt", text="Third chunk", embedding=[0.2, 0.7, 0.1]),    # Similarity: 0.2
        ]

        # Test similarity search with default top_k
        result = RagService.similarity_search("test query", chunks)

        # Verify generate_embeddings was called correctly
        mock_generate_embeddings.assert_called_once_with("test query")

        # Check if results are ordered by similarity and concatenated correctly
        expected_result = "First chunk Second chunk Third chunk"
        assert result == expected_result

        # Test with custom top_k
        result_top_2 = RagService.similarity_search("test query", chunks, top_k=2)
        expected_result_top_2 = "First chunk Second chunk"
        assert result_top_2 == expected_result_top_2
        