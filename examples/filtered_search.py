"""
フィルター検索サンプル

カテゴリ、日付、著者などの条件でフィルタリングする検索サンプル。
OData フィルター式の構築パターンを解説。

Usage:
    python examples/filtered_search.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from search_client import AzureSearchClient

# 環境変数読み込み
load_dotenv()


def main():
    """フィルター検索のデモンストレーション"""
    
    print("=" * 60)
    print("Azure AI Search RAG Toolkit - Filtered Search Demo")
    print("=" * 60)
    
    client = AzureSearchClient()
    
    # ドキュメント数確認
    doc_count = client.get_document_count()
    print(f"\n📊 Index contains {doc_count} documents")
    
    if doc_count == 0:
        print("⚠️  No documents in index. Run batch_upload.py first.")
        return
    
    query = "Azure"
    
    # 1. カテゴリフィルター
    print(f"\n🏷️  Category Filter: 'Azure'")
    print("-" * 40)
    
    results = client.search_with_filters(
        query=query,
        categories=["Azure"],
        top=3,
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.category}] {r.title}")
    
    # 2. 複数カテゴリフィルター
    print(f"\n🏷️  Multiple Categories: 'Azure', 'Security'")
    print("-" * 40)
    
    results = client.search_with_filters(
        query=query,
        categories=["Azure", "Security"],
        top=5,
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.category}] {r.title}")
    
    # 3. 日付範囲フィルター
    print(f"\n📅 Date Range: Last 30 days")
    print("-" * 40)
    
    date_from = datetime.now() - timedelta(days=30)
    
    results = client.search_with_filters(
        query=query,
        date_from=date_from,
        top=3,
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}")
    
    # 4. 著者フィルター
    print(f"\n👤 Author Filter: 'Azure Documentation Team'")
    print("-" * 40)
    
    results = client.search_with_filters(
        query=query,
        authors=["Azure Documentation Team"],
        top=3,
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}")
    
    # 5. 複合フィルター
    print(f"\n🔧 Combined Filters:")
    print("   - Category: Azure")
    print("   - Date: Last 30 days")
    print("   - Use Hybrid Search: Yes")
    print("-" * 40)
    
    results = client.search_with_filters(
        query="検索機能",
        categories=["Azure"],
        date_from=date_from,
        use_hybrid=True,
        top=3,
    )
    for i, r in enumerate(results, 1):
        reranker = f" | Reranker: {r.reranker_score:.4f}" if r.reranker_score else ""
        print(f"{i}. [{r.score:.4f}{reranker}] {r.title}")
        print(f"   Category: {r.category}")
    
    # 6. 機密レベルフィルター
    print(f"\n🔒 Confidentiality Level: 'internal'")
    print("-" * 40)
    
    results = client.search_with_filters(
        query=query,
        confidentiality_levels=["internal"],
        top=3,
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}")
    
    # 7. 直接 OData フィルター
    print(f"\n📝 Direct OData Filter: category eq 'Architecture'")
    print("-" * 40)
    
    results = client.hybrid_search(
        query="RAG パターン",
        filter_expression="category eq 'Architecture'",
        top=3,
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.category}] {r.title}")
    
    print("\n" + "=" * 60)
    print("Filtered search demo completed!")


if __name__ == "__main__":
    main()
