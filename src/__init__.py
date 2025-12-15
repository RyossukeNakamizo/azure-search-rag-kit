"""
Azure AI Search RAG Toolkit

Enterprise-grade RAG system implementation using Azure AI Search.
"""

from .search_client import (
    AzureSearchClient,
    Document,
    DocumentMetadata,
    SearchResult,
    create_document,
)

__all__ = [
    "AzureSearchClient",
    "Document",
    "DocumentMetadata",
    "SearchResult",
    "create_document",
]

__version__ = "1.0.0"
