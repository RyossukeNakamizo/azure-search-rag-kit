"""
Azure Search Client Unit Tests
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDocument:
    """Document dataclass tests"""

    def test_document_to_dict(self, sample_document):
        """Test Document.to_dict() conversion"""
        doc_dict = sample_document.to_dict()

        assert doc_dict["id"] == "test-doc-1"
        assert doc_dict["documentId"] == "test-doc"
        assert doc_dict["title"] == "Test Document"
        assert doc_dict["chunk"] == "This is test content."
        assert doc_dict["category"] == "Test"
        assert doc_dict["tags"] == ["test", "unit"]
        assert doc_dict["metadata_fileType"] == "txt"
        assert doc_dict["metadata_language"] == "ja"

    def test_document_with_vector(self, sample_document):
        """Test Document with content vector"""
        sample_document.content_vector = [0.1] * 3072
        doc_dict = sample_document.to_dict()

        assert "contentVector" in doc_dict
        assert len(doc_dict["contentVector"]) == 3072


class TestSearchResult:
    """SearchResult dataclass tests"""

    def test_search_result_creation(self):
        """Test SearchResult creation"""
        from search_client import SearchResult

        result = SearchResult(
            id="doc1",
            title="Test",
            content="Content",
            chunk="Chunk",
            score=0.95,
            reranker_score=0.98,
            category="Test",
        )

        assert result.id == "doc1"
        assert result.score == 0.95
        assert result.reranker_score == 0.98


class TestCreateDocument:
    """create_document helper function tests"""

    def test_create_document_basic(self):
        """Test basic document creation"""
        from search_client import create_document

        doc = create_document(
            document_id="doc001",
            title="Test Title",
            content="Full content",
            chunk="Chunk content",
            chunk_index=0,
        )

        assert doc.id == "doc001_0"
        assert doc.document_id == "doc001"
        assert doc.title == "Test Title"
        assert doc.chunk_index == 0

    def test_create_document_with_metadata(self):
        """Test document creation with metadata"""
        from search_client import create_document

        doc = create_document(
            document_id="doc002",
            title="Test",
            content="Content",
            chunk="Chunk",
            chunk_index=1,
            category="Category",
            tags=["tag1", "tag2"],
            file_type="pdf",
            confidentiality_level="public",
        )

        assert doc.category == "Category"
        assert doc.tags == ["tag1", "tag2"]
        assert doc.metadata.file_type == "pdf"
        assert doc.metadata.confidentiality_level == "public"


class TestAzureSearchClient:
    """AzureSearchClient tests"""

    def test_client_initialization_missing_env(self):
        """Test client raises error when env vars missing"""
        from search_client import AzureSearchClient

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="must be set"):
                AzureSearchClient()

    def test_client_initialization_success(self, mock_env, mock_credential):
        """Test successful client initialization"""
        from search_client import AzureSearchClient

        with (
            patch("search_client.DefaultAzureCredential", return_value=mock_credential),
            patch("search_client.SearchIndexClient"),
            patch("search_client.SearchClient"),
        ):
            client = AzureSearchClient()
            assert client.endpoint == mock_env["AZURE_SEARCH_ENDPOINT"]
            assert client.index_name == mock_env["AZURE_SEARCH_INDEX_NAME"]

    def test_generate_embedding(self, mock_env, mock_credential, mock_openai_client):
        """Test embedding generation"""
        from search_client import AzureSearchClient

        with (
            patch("search_client.DefaultAzureCredential", return_value=mock_credential),
            patch("search_client.SearchIndexClient"),
            patch("search_client.SearchClient"),
            patch("search_client.AzureOpenAI", return_value=mock_openai_client),
        ):
            client = AzureSearchClient()
            embedding = client.generate_embedding("test text")
            assert len(embedding) == 3072
            mock_openai_client.embeddings.create.assert_called_once()

    def test_generate_embeddings_batch(self, mock_env, mock_credential, mock_openai_client):
        """Test batch embedding generation"""
        from search_client import AzureSearchClient

        with (
            patch("search_client.DefaultAzureCredential", return_value=mock_credential),
            patch("search_client.SearchIndexClient"),
            patch("search_client.SearchClient"),
            patch("search_client.AzureOpenAI", return_value=mock_openai_client),
        ):
            client = AzureSearchClient()
            texts = ["text1", "text2", "text3"]
            _ = client.generate_embeddings_batch(texts)
            # バッチサイズ16なので1回の呼び出し
            assert mock_openai_client.embeddings.create.call_count == 1


class TestSearchOperations:
    """Search operation tests"""

    def test_search_returns_results(self, mock_env, mock_credential, mock_search_client):
        """Test basic search returns SearchResult objects"""
        from search_client import AzureSearchClient

        with (
            patch("search_client.DefaultAzureCredential", return_value=mock_credential),
            patch("search_client.SearchIndexClient"),
            patch("search_client.SearchClient", return_value=mock_search_client),
        ):
            client = AzureSearchClient()
            results = client.search("test query")
            assert len(results) == 1
            assert results[0].id == "doc1"
            assert results[0].title == "Test Document"
            assert results[0].score == 0.95

    def test_vector_search_with_query(self, mock_env, mock_credential, mock_search_client, mock_openai_client):
        """Test vector search with text query"""
        from search_client import AzureSearchClient

        with (
            patch("search_client.DefaultAzureCredential", return_value=mock_credential),
            patch("search_client.SearchIndexClient"),
            patch("search_client.SearchClient", return_value=mock_search_client),
            patch("search_client.AzureOpenAI", return_value=mock_openai_client),
        ):
            client = AzureSearchClient()
            results = client.vector_search(query="test query")
            assert len(results) == 1

    def test_vector_search_requires_query_or_vector(self, mock_env, mock_credential, mock_search_client):
        """Test vector search raises error without query or vector"""
        from search_client import AzureSearchClient

        with (
            patch("search_client.DefaultAzureCredential", return_value=mock_credential),
            patch("search_client.SearchIndexClient"),
            patch("search_client.SearchClient", return_value=mock_search_client),
        ):
            client = AzureSearchClient()
            with pytest.raises(ValueError, match="Either 'query' or 'vector'"):
                client.vector_search()


class TestFilterExpressionBuilding:
    """Filter expression building tests"""

    def test_category_filter(self, mock_env, mock_credential, mock_search_client):
        """Test category filter expression"""
        from search_client import AzureSearchClient

        with (
            patch("search_client.DefaultAzureCredential", return_value=mock_credential),
            patch("search_client.SearchIndexClient"),
            patch("search_client.SearchClient", return_value=mock_search_client),
        ):
            client = AzureSearchClient()
            # search_with_filters内部でフィルター構築を確認
            client.search_with_filters(
                query="test",
                categories=["Cat1", "Cat2"],
                use_hybrid=False,
            )
            # searchが呼ばれたことを確認
            mock_search_client.search.assert_called()
            # フィルター引数を確認
            call_kwargs = mock_search_client.search.call_args[1]
            assert "filter" in call_kwargs
            assert "category eq 'Cat1'" in call_kwargs["filter"]
            assert "category eq 'Cat2'" in call_kwargs["filter"]
