"""
基本検索サンプル

Azure AI Search の基本的な検索操作を示すサンプルコード。
キーワード検索、ベクトル検索、ハイブリッド検索の使用方法を解説。

Usage:
    python examples/basic_search.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from search_client import AzureSearchClient

# 環境変数読み込み
load_dotenv()


def main():
    """基本検索のデモンストレーション"""

    # クライアント初期化
    print("=" * 60)
    print("Azure AI Search RAG Toolkit - Basic Search Demo")
    print("=" * 60)

    client = AzureSearchClient()

    # ドキュメント数確認
    doc_count = client.get_document_count()
    print(f"\n📊 Index contains {doc_count} documents")

    if doc_count == 0:
        print("⚠️  No documents in index. Run batch_upload.py first.")
        return

    query = "Azure AI Search の特徴"

    # 1. キーワード検索
    print(f"\n🔍 Keyword Search: '{query}'")
    print("-" * 40)

    keyword_results = client.search(query, top=3)
    for i, r in enumerate(keyword_results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title}")
        print(f"   {r.chunk[:80]}...")

    # 2. ベクトル検索
    print(f"\n🧮 Vector Search: '{query}'")
    print("-" * 40)

    vector_results = client.vector_search(query=query, top=3)
    for i, r in enumerate(vector_results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title}")
        print(f"   {r.chunk[:80]}...")

    # 3. ハイブリッド検索（セマンティックリランキング付き）
    print(f"\n🚀 Hybrid Search with Semantic Reranking: '{query}'")
    print("-" * 40)

    hybrid_results = client.hybrid_search(
        query=query,
        top=3,
        use_semantic_reranker=True,
    )
    for i, r in enumerate(hybrid_results, 1):
        reranker = f" | Reranker: {r.reranker_score:.4f}" if r.reranker_score else ""
        print(f"{i}. [Score: {r.score:.4f}{reranker}] {r.title}")
        print(f"   {r.chunk[:80]}...")
        if r.highlights.get("chunk"):
            print(f"   💡 Highlight: {r.highlights['chunk'][0][:60]}...")

    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    main()
