import pytest
from unittest.mock import mock_open, patch, MagicMock
from minirag.services.document_service import DocumentService
from minirag.models import Chunk


@pytest.fixture
def sample_text():
    return "This is a sample document text."


def test_read_document_success(sample_text):
    with patch("builtins.open", mock_open(read_data=sample_text)):
        result = DocumentService.read_document("fake/path.txt")
        assert result == sample_text


def test_read_document_file_not_found():
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = FileNotFoundError()
        result = DocumentService.read_document("nonexistent.txt")
        assert result == ""


def test_read_document_general_error():
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = Exception("Some error")
        result = DocumentService.read_document("error.txt")
        assert result == ""


def test_process_document_empty_file():
    with patch.object(DocumentService, "read_document", return_value=""):
        result = DocumentService.process_document("empty.txt")
        assert len(result) == 0
