"""
Azure AI Search 操作クライアント

RAGシステム向けのAzure AI Search操作を提供するクラス。
インデックス管理、ハイブリッド検索、ドキュメントCRUD、バッチ処理に対応。

Requirements:
    pip install azure-search-documents azure-identity openai python-dotenv

Environment Variables:
    AZURE_SEARCH_ENDPOINT: Azure AI Search エンドポイント
    AZURE_SEARCH_INDEX_NAME: インデックス名
    AZURE_OPENAI_ENDPOINT: Azure OpenAI エンドポイント
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: 埋め込みモデルのデプロイ名
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from azure.core.credentials import TokenCredential
from azure.core.exceptions import (
    ResourceExistsError,
    ResourceNotFoundError,
)
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.models import (
    QueryType,
    VectorizableTextQuery,
    VectorizedQuery,
)
from openai import AzureOpenAI

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """検索結果を格納するデータクラス"""

    id: str
    title: str
    content: str
    chunk: str
    score: float
    reranker_score: float | None = None
    category: str | None = None
    source_url: str | None = None
    highlights: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    """ドキュメントメタデータを格納するデータクラス"""

    file_type: str | None = None
    file_size: int | None = None
    language: str = "ja"
    confidentiality_level: str = "internal"


@dataclass
class Document:
    """インデックス登録用ドキュメント"""

    id: str
    document_id: str
    title: str
    content: str
    chunk: str
    chunk_index: int
    content_vector: list[float] | None = None
    category: str | None = None
    subcategory: str | None = None
    tags: list[str] = field(default_factory=list)
    created_date: datetime | None = None
    last_modified_date: datetime | None = None
    author: str | None = None
    source_url: str | None = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)

    def to_dict(self) -> dict[str, Any]:
        """Azure AI Search用の辞書形式に変換"""
        doc = {
            "id": self.id,
            "documentId": self.document_id,
            "title": self.title,
            "content": self.content,
            "chunk": self.chunk,
            "chunkIndex": self.chunk_index,
            "category": self.category,
            "subcategory": self.subcategory,
            "tags": self.tags,
            "author": self.author,
            "sourceUrl": self.source_url,
            "metadata_fileType": self.metadata.file_type,
            "metadata_fileSize": self.metadata.file_size,
            "metadata_language": self.metadata.language,
            "metadata_confidentialityLevel": self.metadata.confidentiality_level,
        }

        if self.content_vector:
            doc["contentVector"] = self.content_vector

        if self.created_date:
            doc["createdDate"] = self.created_date.isoformat()

        if self.last_modified_date:
            doc["lastModifiedDate"] = self.last_modified_date.isoformat()

        return doc


class AzureSearchClient:
    """
    Azure AI Search 操作クライアント

    Attributes:
        endpoint: Azure AI Search エンドポイント
        index_name: 対象インデックス名
        credential: Azure認証情報
    """

    def __init__(
        self,
        endpoint: str | None = None,
        index_name: str | None = None,
        credential: TokenCredential | None = None,
        openai_endpoint: str | None = None,
        embedding_deployment: str | None = None,
    ):
        """
        クライアントを初期化

        Args:
            endpoint: Azure AI Search エンドポイント（省略時は環境変数から取得）
            index_name: インデックス名（省略時は環境変数から取得）
            credential: Azure認証情報（省略時はDefaultAzureCredential）
            openai_endpoint: Azure OpenAI エンドポイント（省略時は環境変数から取得）
            embedding_deployment: 埋め込みモデルのデプロイ名（省略時は環境変数から取得）
        """
        self.endpoint = endpoint or os.getenv("AZURE_SEARCH_ENDPOINT")
        self.index_name = index_name or os.getenv("AZURE_SEARCH_INDEX_NAME")
        self.credential = credential or DefaultAzureCredential()

        if not self.endpoint or not self.index_name:
            raise ValueError(
                "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX_NAME must be set"
            )

        # Search クライアント初期化
        self._index_client = SearchIndexClient(
            endpoint=self.endpoint, credential=self.credential
        )
        self._search_client = SearchClient(
            endpoint=self.endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )

        # OpenAI クライアント初期化（埋め込み用）
        self._openai_endpoint = openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self._embedding_deployment = embedding_deployment or os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        )

        if self._openai_endpoint:
            self._openai_client = AzureOpenAI(
                azure_endpoint=self._openai_endpoint,
                azure_ad_token_provider=self._get_openai_token,
                api_version="2024-10-01-preview",
            )
        else:
            self._openai_client = None
            logger.warning(
                "AZURE_OPENAI_ENDPOINT not set. Embedding generation disabled."
            )

    def _get_openai_token(self) -> str:
        """Azure AD トークンを取得"""
        return self.credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token

    # ==================== インデックス管理 ====================

    def create_index_from_schema(self, schema_path: str) -> SearchIndex:
        """
        JSONスキーマファイルからインデックスを作成

        Args:
            schema_path: スキーマJSONファイルのパス

        Returns:
            作成されたSearchIndexオブジェクト

        Raises:
            ResourceExistsError: インデックスが既に存在する場合
            FileNotFoundError: スキーマファイルが見つからない場合
        """
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        try:
            index = self._index_client.create_index(schema)
            logger.info(f"Index '{index.name}' created successfully")
            return index
        except ResourceExistsError:
            logger.warning(f"Index '{schema['name']}' already exists")
            raise

    def create_or_update_index(self, schema_path: str) -> SearchIndex:
        """
        JSONスキーマファイルからインデックスを作成または更新

        Args:
            schema_path: スキーマJSONファイルのパス

        Returns:
            作成/更新されたSearchIndexオブジェクト
        """
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

        index = self._index_client.create_or_update_index(schema)
        logger.info(f"Index '{index.name}' created/updated successfully")
        return index

    def delete_index(self, index_name: str | None = None) -> None:
        """
        インデックスを削除

        Args:
            index_name: 削除するインデックス名（省略時は初期化時の名前）
        """
        target_index = index_name or self.index_name
        self._index_client.delete_index(target_index)
        logger.info(f"Index '{target_index}' deleted successfully")

    def get_index(self, index_name: str | None = None) -> SearchIndex:
        """
        インデックス情報を取得

        Args:
            index_name: 取得するインデックス名（省略時は初期化時の名前）

        Returns:
            SearchIndexオブジェクト
        """
        target_index = index_name or self.index_name
        return self._index_client.get_index(target_index)

    def list_indexes(self) -> list[str]:
        """
        全インデックス名を取得

        Returns:
            インデックス名のリスト
        """
        return [index.name for index in self._index_client.list_indexes()]

    # ==================== 埋め込み生成 ====================

    def generate_embedding(self, text: str) -> list[float]:
        """
        テキストから埋め込みベクトルを生成

        Args:
            text: 埋め込み対象テキスト

        Returns:
            埋め込みベクトル（3072次元）

        Raises:
            RuntimeError: OpenAIクライアントが初期化されていない場合
        """
        if not self._openai_client:
            raise RuntimeError(
                "OpenAI client not initialized. Set AZURE_OPENAI_ENDPOINT."
            )

        response = self._openai_client.embeddings.create(
            model=self._embedding_deployment, input=text
        )
        return response.data[0].embedding

    def generate_embeddings_batch(
        self, texts: list[str], batch_size: int = 16
    ) -> list[list[float]]:
        """
        複数テキストの埋め込みをバッチ生成

        Args:
            texts: 埋め込み対象テキストのリスト
            batch_size: バッチサイズ（最大16推奨）

        Returns:
            埋め込みベクトルのリスト
        """
        if not self._openai_client:
            raise RuntimeError(
                "OpenAI client not initialized. Set AZURE_OPENAI_ENDPOINT."
            )

        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._openai_client.embeddings.create(
                model=self._embedding_deployment, input=batch
            )
            embeddings.extend([item.embedding for item in response.data])
            logger.debug(f"Generated embeddings for batch {i // batch_size + 1}")

        return embeddings

    # ==================== ドキュメント操作 ====================

    def upload_documents(
        self, documents: list[Document], generate_vectors: bool = True
    ) -> dict[str, Any]:
        """
        ドキュメントをアップロード

        Args:
            documents: アップロードするDocumentのリスト
            generate_vectors: 埋め込みベクトルを自動生成するか

        Returns:
            アップロード結果
        """
        if generate_vectors and self._openai_client:
            texts = [doc.chunk for doc in documents]
            vectors = self.generate_embeddings_batch(texts)
            for doc, vector in zip(documents, vectors):
                doc.content_vector = vector

        docs_dict = [doc.to_dict() for doc in documents]
        result = self._search_client.upload_documents(documents=docs_dict)

        success_count = sum(1 for r in result if r.succeeded)
        logger.info(f"Uploaded {success_count}/{len(documents)} documents successfully")

        return {
            "total": len(documents),
            "succeeded": success_count,
            "failed": len(documents) - success_count,
        }

    def merge_documents(self, documents: list[Document]) -> dict[str, Any]:
        """
        ドキュメントをマージ（部分更新）

        Args:
            documents: マージするDocumentのリスト

        Returns:
            マージ結果
        """
        docs_dict = [doc.to_dict() for doc in documents]
        result = self._search_client.merge_documents(documents=docs_dict)

        success_count = sum(1 for r in result if r.succeeded)
        logger.info(f"Merged {success_count}/{len(documents)} documents successfully")

        return {
            "total": len(documents),
            "succeeded": success_count,
            "failed": len(documents) - success_count,
        }

    def delete_documents(self, document_ids: list[str]) -> dict[str, Any]:
        """
        ドキュメントを削除

        Args:
            document_ids: 削除するドキュメントIDのリスト

        Returns:
            削除結果
        """
        documents = [{"id": doc_id} for doc_id in document_ids]
        result = self._search_client.delete_documents(documents=documents)

        success_count = sum(1 for r in result if r.succeeded)
        logger.info(
            f"Deleted {success_count}/{len(document_ids)} documents successfully"
        )

        return {
            "total": len(document_ids),
            "succeeded": success_count,
            "failed": len(document_ids) - success_count,
        }

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        """
        ドキュメントを取得

        Args:
            document_id: 取得するドキュメントID

        Returns:
            ドキュメント（見つからない場合はNone）
        """
        try:
            return self._search_client.get_document(key=document_id)
        except ResourceNotFoundError:
            logger.warning(f"Document '{document_id}' not found")
            return None

    def get_document_count(self) -> int:
        """
        インデックス内のドキュメント数を取得

        Returns:
            ドキュメント数
        """
        return self._search_client.get_document_count()

    # ==================== 検索操作 ====================

    def search(
        self,
        query: str,
        *,
        top: int = 10,
        filter_expression: str | None = None,
        select_fields: list[str] | None = None,
        include_total_count: bool = True,
    ) -> list[SearchResult]:
        """
        キーワード検索を実行

        Args:
            query: 検索クエリ
            top: 取得件数
            filter_expression: ODataフィルター式
            select_fields: 取得フィールド
            include_total_count: 総件数を含めるか

        Returns:
            SearchResultのリスト
        """
        results = self._search_client.search(
            search_text=query,
            top=top,
            filter=filter_expression,
            select=select_fields
            or ["id", "title", "content", "chunk", "category", "sourceUrl"],
            include_total_count=include_total_count,
            highlight_fields="chunk",
        )

        return [
            SearchResult(
                id=r["id"],
                title=r.get("title", ""),
                content=r.get("content", ""),
                chunk=r.get("chunk", ""),
                score=r["@search.score"],
                category=r.get("category"),
                source_url=r.get("sourceUrl"),
                highlights=r.get("@search.highlights", {}),
            )
            for r in results
        ]

    def vector_search(
        self,
        query: str | None = None,
        vector: list[float] | None = None,
        *,
        top: int = 10,
        filter_expression: str | None = None,
        select_fields: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        ベクトル検索を実行

        Args:
            query: 検索クエリ（自動ベクトル化）
            vector: 検索ベクトル（直接指定）
            top: 取得件数
            filter_expression: ODataフィルター式
            select_fields: 取得フィールド

        Returns:
            SearchResultのリスト

        Raises:
            ValueError: queryとvectorの両方が未指定の場合
        """
        if query is None and vector is None:
            raise ValueError("Either 'query' or 'vector' must be provided")

        if vector:
            vector_query = VectorizedQuery(
                vector=vector, k_nearest_neighbors=top, fields="contentVector"
            )
        else:
            vector_query = VectorizableTextQuery(
                text=query, k_nearest_neighbors=top, fields="contentVector"
            )

        results = self._search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            top=top,
            filter=filter_expression,
            select=select_fields
            or ["id", "title", "content", "chunk", "category", "sourceUrl"],
        )

        return [
            SearchResult(
                id=r["id"],
                title=r.get("title", ""),
                content=r.get("content", ""),
                chunk=r.get("chunk", ""),
                score=r["@search.score"],
                category=r.get("category"),
                source_url=r.get("sourceUrl"),
            )
            for r in results
        ]

    def hybrid_search(
        self,
        query: str,
        *,
        top: int = 10,
        filter_expression: str | None = None,
        select_fields: list[str] | None = None,
        use_semantic_reranker: bool = True,
        semantic_config_name: str = "semantic-config",
    ) -> list[SearchResult]:
        """
        ハイブリッド検索を実行（キーワード + ベクトル + セマンティックリランキング）

        Args:
            query: 検索クエリ
            top: 取得件数
            filter_expression: ODataフィルター式
            select_fields: 取得フィールド
            use_semantic_reranker: セマンティックリランカーを使用するか
            semantic_config_name: セマンティック設定名

        Returns:
            SearchResultのリスト
        """
        vector_query = VectorizableTextQuery(
            text=query, k_nearest_neighbors=top * 2, fields="contentVector"
        )

        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": top,
            "filter": filter_expression,
            "select": select_fields
            or ["id", "title", "content", "chunk", "category", "sourceUrl"],
            "highlight_fields": "chunk",
        }

        if use_semantic_reranker:
            search_kwargs.update(
                {
                    "query_type": QueryType.SEMANTIC,
                    "semantic_configuration_name": semantic_config_name,
                }
            )

        results = self._search_client.search(**search_kwargs)

        return [
            SearchResult(
                id=r["id"],
                title=r.get("title", ""),
                content=r.get("content", ""),
                chunk=r.get("chunk", ""),
                score=r["@search.score"],
                reranker_score=r.get("@search.reranker_score"),
                category=r.get("category"),
                source_url=r.get("sourceUrl"),
                highlights=r.get("@search.highlights", {}),
            )
            for r in results
        ]

    def search_with_filters(
        self,
        query: str,
        *,
        categories: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        authors: list[str] | None = None,
        confidentiality_levels: list[str] | None = None,
        top: int = 10,
        use_hybrid: bool = True,
    ) -> list[SearchResult]:
        """
        フィルター付き検索を実行

        Args:
            query: 検索クエリ
            categories: カテゴリフィルター
            date_from: 作成日（開始）
            date_to: 作成日（終了）
            authors: 著者フィルター
            confidentiality_levels: 機密レベルフィルター
            top: 取得件数
            use_hybrid: ハイブリッド検索を使用するか

        Returns:
            SearchResultのリスト
        """
        filters = []

        if categories:
            category_filter = " or ".join(
                [f"category eq '{cat}'" for cat in categories]
            )
            filters.append(f"({category_filter})")

        if date_from:
            filters.append(f"createdDate ge {date_from.isoformat()}Z")

        if date_to:
            filters.append(f"createdDate le {date_to.isoformat()}Z")

        if authors:
            author_filter = " or ".join([f"author eq '{auth}'" for auth in authors])
            filters.append(f"({author_filter})")

        if confidentiality_levels:
            level_filter = " or ".join(
                [
                    f"metadata_confidentialityLevel eq '{lvl}'"
                    for lvl in confidentiality_levels
                ]
            )
            filters.append(f"({level_filter})")

        filter_expression = " and ".join(filters) if filters else None

        if use_hybrid:
            return self.hybrid_search(
                query, top=top, filter_expression=filter_expression
            )
        else:
            return self.search(query, top=top, filter_expression=filter_expression)


# ==================== ユーティリティ関数 ====================


def create_document(
    document_id: str,
    title: str,
    content: str,
    chunk: str,
    chunk_index: int,
    **kwargs,
) -> Document:
    """
    Documentオブジェクトを作成するヘルパー関数

    Args:
        document_id: ドキュメントID
        title: タイトル
        content: 全文コンテンツ
        chunk: チャンクテキスト
        chunk_index: チャンクインデックス
        **kwargs: その他のオプション引数

    Returns:
        Documentオブジェクト
    """
    return Document(
        id=f"{document_id}_{chunk_index}",
        document_id=document_id,
        title=title,
        content=content,
        chunk=chunk,
        chunk_index=chunk_index,
        category=kwargs.get("category"),
        subcategory=kwargs.get("subcategory"),
        tags=kwargs.get("tags", []),
        created_date=kwargs.get("created_date"),
        last_modified_date=kwargs.get("last_modified_date"),
        author=kwargs.get("author"),
        source_url=kwargs.get("source_url"),
        metadata=DocumentMetadata(
            file_type=kwargs.get("file_type"),
            file_size=kwargs.get("file_size"),
            language=kwargs.get("language", "ja"),
            confidentiality_level=kwargs.get("confidentiality_level", "internal"),
        ),
    )


# ==================== 使用例 ====================

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # クライアント初期化
    client = AzureSearchClient()

    # インデックス作成
    # client.create_or_update_index("index_schema.json")

    # ドキュメント作成
    doc = create_document(
        document_id="doc001",
        title="Azure AI Search 入門",
        content="Azure AI Searchはフルマネージドの検索サービスです...",
        chunk="Azure AI Searchはフルマネージドの検索サービスです。",
        chunk_index=0,
        category="Azure",
        tags=["search", "ai", "rag"],
        author="Ryo",
        created_date=datetime.now(),
    )

    # アップロード
    # client.upload_documents([doc])

    # ハイブリッド検索
    results = client.hybrid_search(
        query="Azure AI Searchとは",
        top=5,
        use_semantic_reranker=True,
    )

    for r in results:
        print(f"[{r.score:.4f}] {r.title}: {r.chunk[:100]}...")
