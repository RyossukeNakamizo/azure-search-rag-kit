"""
pytest configuration and fixtures
"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_env():
    """Mock environment variables"""
    env_vars = {
        "AZURE_SEARCH_ENDPOINT": "https://test-search.search.windows.net",
        "AZURE_SEARCH_INDEX_NAME": "test-index",
        "AZURE_OPENAI_ENDPOINT": "https://test-openai.openai.azure.com",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-large",
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_credential():
    """Mock Azure credential"""
    credential = MagicMock()
    credential.get_token.return_value = MagicMock(token="mock-token")
    return credential


@pytest.fixture
def mock_search_client():
    """Mock SearchClient"""
    client = MagicMock()
    client.search.return_value = iter([
        {
            "id": "doc1",
            "title": "Test Document",
            "content": "Test content",
            "chunk": "Test chunk",
            "@search.score": 0.95,
            "category": "Test",
        }
    ])
    client.get_document_count.return_value = 100
    return client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client"""
    client = MagicMock()
    
    # Mock embedding response
    embedding_response = MagicMock()
    embedding_response.data = [MagicMock(embedding=[0.1] * 3072)]
    client.embeddings.create.return_value = embedding_response
    
    return client


@pytest.fixture
def sample_document():
    """Sample document for testing"""
    from search_client import Document, DocumentMetadata
    
    return Document(
        id="test-doc-1",
        document_id="test-doc",
        title="Test Document",
        content="This is test content for unit testing.",
        chunk="This is test content.",
        chunk_index=0,
        category="Test",
        tags=["test", "unit"],
        metadata=DocumentMetadata(
            file_type="txt",
            file_size=1024,
            language="ja",
            confidentiality_level="internal",
        ),
    )
