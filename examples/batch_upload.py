"""
バッチアップロードサンプル

大量ドキュメントを効率的にインデックス登録するサンプルコード。
チャンキング、埋め込み生成、バッチ処理の実装パターンを解説。

Usage:
    python examples/batch_upload.py
"""

import sys
from datetime import datetime
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from search_client import AzureSearchClient, create_document

# 環境変数読み込み
load_dotenv()


# サンプルドキュメントデータ
SAMPLE_DOCUMENTS = [
    {
        "document_id": "azure-search-001",
        "title": "Azure AI Search 概要",
        "content": """Azure AI Search（旧 Azure Cognitive Search）は、Microsoft Azure が提供する
        フルマネージドのクラウド検索サービスです。Web サイト、アプリケーション、
        エンタープライズデータに対して、高度な検索機能を簡単に統合できます。
        主な特徴として、フルテキスト検索、ベクトル検索、セマンティック検索、
        AI エンリッチメント機能があります。""",
        "category": "Azure",
        "subcategory": "AI Services",
        "tags": ["search", "ai", "fulltext", "vector"],
        "author": "Azure Documentation Team",
    },
    {
        "document_id": "azure-search-002",
        "title": "ベクトル検索の仕組み",
        "content": """ベクトル検索は、テキストを高次元のベクトル空間に埋め込み、
        類似度に基づいて関連するドキュメントを検索する手法です。
        Azure AI Search では、HNSW（Hierarchical Navigable Small World）
        アルゴリズムを使用して、高速かつ高精度なベクトル検索を実現しています。
        text-embedding-3-large モデルを使用することで、3072次元の
        高品質な埋め込みベクトルを生成できます。""",
        "category": "Azure",
        "subcategory": "Vector Search",
        "tags": ["vector", "embedding", "hnsw", "similarity"],
        "author": "Azure Documentation Team",
    },
    {
        "document_id": "azure-search-003",
        "title": "セマンティック検索とリランキング",
        "content": """セマンティック検索は、クエリとドキュメントの意味的な関連性を
        理解して検索結果を改善する機能です。Azure AI Search の
        セマンティックランカーは、Microsoft の大規模言語モデルを使用して、
        検索結果を意味的な関連度で再ランキングします。
        これにより、キーワードの完全一致がなくても、
        ユーザーの意図に最も適した結果を上位に表示できます。""",
        "category": "Azure",
        "subcategory": "Semantic Search",
        "tags": ["semantic", "reranking", "llm", "relevance"],
        "author": "Azure Documentation Team",
    },
    {
        "document_id": "rag-pattern-001",
        "title": "RAG アーキテクチャパターン",
        "content": """RAG（Retrieval-Augmented Generation）は、外部知識ベースから
        関連情報を取得し、大規模言語モデルの回答生成を強化するパターンです。
        Classic RAG では、単一のクエリで検索を実行し、結果を LLM に渡します。
        Agentic RAG では、複雑なクエリを自動的に分解し、
        複数のサブクエリを並列実行して、より包括的な回答を生成します。""",
        "category": "Architecture",
        "subcategory": "RAG",
        "tags": ["rag", "llm", "retrieval", "generation"],
        "author": "Solution Architecture Team",
    },
    {
        "document_id": "security-001",
        "title": "Azure AI Search のセキュリティ",
        "content": """Azure AI Search では、複数のセキュリティレイヤーを提供しています。
        認証には、Managed Identity を使用した RBAC 認証を推奨します。
        API キー認証も利用可能ですが、本番環境では Managed Identity が
        よりセキュアです。ネットワークセキュリティとして、
        Private Endpoint を使用して、VNet 内からのみアクセスを許可できます。
        データは保存時と転送時に暗号化されます。""",
        "category": "Security",
        "subcategory": "Authentication",
        "tags": ["security", "rbac", "managed-identity", "private-endpoint"],
        "author": "Security Team",
    },
]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    テキストをチャンクに分割
    
    Args:
        text: 分割対象テキスト
        chunk_size: チャンクサイズ（文字数）
        overlap: オーバーラップ（文字数）
    
    Returns:
        チャンクのリスト
    """
    text = " ".join(text.split())  # 空白正規化
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 文の境界で分割（句点を探す）
        if end < len(text):
            boundary = text.rfind("。", start, end)
            if boundary > start:
                end = boundary + 1
        
        chunks.append(text[start:end].strip())
        start = end - overlap
    
    return chunks


def main():
    """バッチアップロードのデモンストレーション"""
    
    print("=" * 60)
    print("Azure AI Search RAG Toolkit - Batch Upload Demo")
    print("=" * 60)
    
    # クライアント初期化
    client = AzureSearchClient()
    
    # インデックス作成（既存の場合は更新）
    schema_path = Path(__file__).parent.parent / "schemas" / "index_schema.json"
    print(f"\n📋 Creating/updating index from: {schema_path}")
    
    try:
        client.create_or_update_index(str(schema_path))
        print("✅ Index ready")
    except Exception as e:
        print(f"⚠️  Index operation: {e}")
    
    # ドキュメント準備
    print(f"\n📄 Preparing {len(SAMPLE_DOCUMENTS)} documents...")
    
    documents = []
    for doc_data in SAMPLE_DOCUMENTS:
        # チャンキング
        chunks = chunk_text(doc_data["content"])
        
        for idx, chunk in enumerate(chunks):
            doc = create_document(
                document_id=doc_data["document_id"],
                title=doc_data["title"],
                content=doc_data["content"],
                chunk=chunk,
                chunk_index=idx,
                category=doc_data.get("category"),
                subcategory=doc_data.get("subcategory"),
                tags=doc_data.get("tags", []),
                author=doc_data.get("author"),
                created_date=datetime.now(),
                language="ja",
                confidentiality_level="internal",
            )
            documents.append(doc)
    
    print(f"   Total chunks: {len(documents)}")
    
    # アップロード（自動ベクトル化）
    print("\n🚀 Uploading documents with auto-vectorization...")
    
    result = client.upload_documents(documents, generate_vectors=True)
    
    print(f"\n📊 Upload Results:")
    print(f"   Total: {result['total']}")
    print(f"   Succeeded: {result['succeeded']}")
    print(f"   Failed: {result['failed']}")
    
    # 確認
    doc_count = client.get_document_count()
    print(f"\n✅ Index now contains {doc_count} documents")
    
    print("\n" + "=" * 60)
    print("Batch upload completed! Run basic_search.py to test.")


if __name__ == "__main__":
    main()
