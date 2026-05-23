"""
Medical Knowledge Base — PubMed Edition
Carga papers científicos reales desde pubmed_knowledge_base.json
Reemplaza la KB hardcodeada por datos reales de PubMed
"""

import json
import os
from typing import List, Dict



KB_FILE = os.path.join(os.path.dirname(__file__), "pubmed_knowledge_base.json")




def load_pubmed_knowledge_base() -> List[Dict]:
    """
    Carga los papers de PubMed desde el JSON generado por pubmed_fetcher.py
    Devuelve lista de papers con estructura compatible con RAGAgent
    """
    if not os.path.exists(KB_FILE):
        print(f"⚠️  KB file not found: {KB_FILE}")
        print("   Run: python agents/pubmed_fetcher.py")
        return []

    with open(KB_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    papers = data.get("papers", [])
    print(f"✅ Knowledge base cargada: {len(papers)} papers científicos de PubMed")
    return papers




MEDICAL_KNOWLEDGE_BASE = load_pubmed_knowledge_base()




def get_kb_stats() -> Dict:
    """Estadísticas de la knowledge base"""
    if not MEDICAL_KNOWLEDGE_BASE:
        return {"error": "KB vacía — ejecuta pubmed_fetcher.py"}

    years = {}
    journals = {}
    sources = {}

    for paper in MEDICAL_KNOWLEDGE_BASE:
        
        year = paper.get("year", "Unknown")
        years[year] = years.get(year, 0) + 1

        
        journal = paper.get("journal", "Unknown")
        journals[journal] = journals.get(journal, 0) + 1

        
        source = paper.get("source_query", "Unknown")
        sources[source] = sources.get(source, 0) + 1

    
    top_journals = sorted(journals.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_papers": len(MEDICAL_KNOWLEDGE_BASE),
        "source": "PubMed (NCBI)",
        "years_distribution": dict(sorted(years.items())),
        "top_journals": top_journals,
        "queries_distribution": sources
    }




if __name__ == "__main__":
    print("=" * 60)
    print("KNOWLEDGE BASE — PubMed Edition")
    print("=" * 60)

    stats = get_kb_stats()
    print(f"\nTotal papers: {stats['total_papers']}")
    print(f"Source: {stats['source']}")

    print(f"\nDistribución por query:")
    for q, count in stats['queries_distribution'].items():
        print(f"  {q}: {count} papers")

    print(f"\nTop 5 journals:")
    for journal, count in stats['top_journals']:
        print(f"  [{count}] {journal}")

    print(f"\nDistribución por año:")
    for year, count in stats['years_distribution'].items():
        print(f"  {year}: {count} papers")

    print("\n--- PREVIEW (3 papers) ---")
    for i, paper in enumerate(MEDICAL_KNOWLEDGE_BASE[:3], 1):
        print(f"\n[{i}] {paper.get('title', '')[:70]}...")
        print(f"    {paper.get('authors_str', '')} ({paper.get('year', '')})")
        print(f"    {paper.get('journal', '')}")
        print(f"    Abstract: {paper.get('abstract', '')[:100]}...")