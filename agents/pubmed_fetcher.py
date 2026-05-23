"""
PubMed Fetcher
Recupera papers científicos de PubMed y los guarda en JSON
para usar como knowledge base del RAG agent
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import os
from typing import List, Dict



NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = ""  
OUTPUT_FILE = "agents/pubmed_knowledge_base.json"
MAX_PAPERS_PER_QUERY = 200  


PUBMED_QUERIES = [
    {
        "id": "query_1",
        "description": "Menopausia - Calidad de vida, sueño, estrés, fatiga, cognición",
        "query": (
            '("Menopause"[MeSH]) AND '
            '("Quality of Life"[MeSH] OR "Sleep Wake Disorders"[MeSH] OR '
            '"Stress, Psychological"[MeSH] OR "Fatigue"[MeSH] OR "Anxiety"[MeSH] OR '
            '"Cognition Disorders"[MeSH] OR "Cognitive Dysfunction"[MeSH] OR '
            '"Executive Function"[MeSH] OR "Memory"[MeSH] OR cognit*[tiab] OR fatigue[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]) AND (middleaged[Filter]))'
        )
    },
    {
        "id": "query_2",
        "description": "Menopausia - Psicología",
        "query": (
            '("menopause/psychology"[MeSH Terms]) AND '
            '((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]) AND (middleaged[Filter]))'
        )
    },
    {
        "id": "query_3",
        "description": "Menopausia - Fisiología",
        "query": (
            '("menopause/physiology"[MeSH Terms]) AND '
            '((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]) AND (middleaged[Filter]))'
        )
    }
]




def search_pubmed(query: str, max_results: int = MAX_PAPERS_PER_QUERY) -> List[str]:
    """
    Busca en PubMed y devuelve lista de PMIDs
    """
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    }
    if API_KEY:
        params["api_key"] = API_KEY

    try:
        response = requests.get(f"{NCBI_BASE}/esearch.fcgi", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        print(f"  → Encontrados {len(ids)} papers")
        return ids
    except Exception as e:
        print(f"  ✗ Error en esearch: {e}")
        return []




def fetch_paper_details(pmids: List[str]) -> List[Dict]:
    """
    Dado una lista de PMIDs, devuelve los detalles completos
    (título, abstract, autores, año, journal)
    """
    if not pmids:
        return []

   
    all_papers = []
    batch_size = 100

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        print(f"  → Fetching batch {i//batch_size + 1} ({len(batch)} papers)...")

        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "abstract",
            "retmode": "xml",
        }
        if API_KEY:
            params["api_key"] = API_KEY

        try:
            response = requests.get(f"{NCBI_BASE}/efetch.fcgi", params=params, timeout=60)
            response.raise_for_status()

            papers = parse_pubmed_xml(response.text)
            all_papers.extend(papers)

            # Respetar rate limit (3 req/sec sin API key, 10 con API key)
            sleep_time = 0.15 if API_KEY else 0.4
            time.sleep(sleep_time)

        except Exception as e:
            print(f"  ✗ Error en efetch batch {i//batch_size + 1}: {e}")

    return all_papers




def parse_pubmed_xml(xml_text: str) -> List[Dict]:
    """
    Parsea el XML de PubMed y extrae los campos relevantes
    """
    papers = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  ✗ Error parsing XML: {e}")
        return []

    for article in root.findall(".//PubmedArticle"):
        paper = {}

        
        pmid_el = article.find(".//PMID")
        if pmid_el is not None:
            paper["pmid"] = pmid_el.text
        else:
            continue  

        
        title_el = article.find(".//ArticleTitle")
        paper["title"] = title_el.text if title_el is not None else ""
       
        if paper["title"]:
            paper["title"] = ET.tostring(title_el, encoding="unicode", method="text") if title_el is not None else ""

        
        abstract_texts = article.findall(".//AbstractText")
        if abstract_texts:
            
            abstract_parts = []
            for at in abstract_texts:
                label = at.get("Label", "")
                text = ET.tostring(at, encoding="unicode", method="text")
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            paper["abstract"] = " ".join(abstract_parts).strip()
        else:
            paper["abstract"] = ""

        
        if not paper["abstract"]:
            continue

        
        authors = []
        for author in article.findall(".//Author"):
            lastname = author.find("LastName")
            forename = author.find("ForeName")
            if lastname is not None:
                name = lastname.text
                if forename is not None:
                    name += f" {forename.text[0]}."  
                authors.append(name)
        paper["authors"] = authors[:6]  
        paper["authors_str"] = ", ".join(paper["authors"][:3])
        if len(paper["authors"]) > 3:
            paper["authors_str"] += " et al."

        
        year_el = article.find(".//PubDate/Year")
        if year_el is None:
            year_el = article.find(".//PubDate/MedlineDate")
        paper["year"] = year_el.text[:4] if year_el is not None else "Unknown"

        
        journal_el = article.find(".//Journal/Title")
        if journal_el is None:
            journal_el = article.find(".//MedlineTA")
        paper["journal"] = journal_el.text if journal_el is not None else "Unknown"

        
        doi_el = article.find(".//ArticleId[@IdType='doi']")
        paper["doi"] = doi_el.text if doi_el is not None else ""

        
        paper["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"

        
        paper["citation"] = f"{paper['authors_str']} ({paper['year']}). {paper['title']}. {paper['journal']}."

        papers.append(paper)

    return papers




def deduplicate_papers(papers: List[Dict]) -> List[Dict]:
    """
    Elimina papers duplicados (mismo PMID de queries diferentes)
    """
    seen_pmids = set()
    unique_papers = []
    for paper in papers:
        if paper["pmid"] not in seen_pmids:
            seen_pmids.add(paper["pmid"])
            unique_papers.append(paper)
    return unique_papers




def save_knowledge_base(papers: List[Dict], output_file: str = OUTPUT_FILE):
    """
    Guarda los papers en JSON para usar como knowledge base
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    kb = {
        "metadata": {
            "total_papers": len(papers),
            "queries_used": [q["description"] for q in PUBMED_QUERIES],
            "source": "PubMed (NCBI)",
            "filters": "Last 5 years, humans, female, middle-aged"
        },
        "papers": papers
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Knowledge base guardada: {output_file}")
    print(f"   Total papers: {len(papers)}")




def fetch_all():
    print("=" * 60)
    print("PUBMED FETCHER — Construyendo Knowledge Base")
    print("=" * 60)

    all_papers = []

    for i, query_config in enumerate(PUBMED_QUERIES, 1):
        print(f"\n[Query {i}/3] {query_config['description']}")
        print(f"  Buscando en PubMed...")

        
        pmids = search_pubmed(query_config["query"])

        if not pmids:
            print(f"  ✗ No se encontraron resultados")
            continue

        
        papers = fetch_paper_details(pmids)
        print(f"  ✓ Papers con abstract: {len(papers)}")

        
        for paper in papers:
            paper["source_query"] = query_config["id"]

        all_papers.extend(papers)

        
        time.sleep(1)

    
    print(f"\n📊 Total antes de deduplicar: {len(all_papers)}")
    unique_papers = deduplicate_papers(all_papers)
    print(f"📊 Total después de deduplicar: {len(unique_papers)}")

    
    save_knowledge_base(unique_papers)

    
    print("\n--- PREVIEW (primer paper) ---")
    if unique_papers:
        p = unique_papers[0]
        print(f"PMID: {p['pmid']}")
        print(f"Título: {p['title'][:80]}...")
        print(f"Autores: {p['authors_str']}")
        print(f"Año: {p['year']}")
        print(f"Journal: {p['journal']}")
        print(f"Abstract: {p['abstract'][:150]}...")
        print(f"URL: {p['url']}")

    print("\n" + "=" * 60)
    print("✅ FETCH COMPLETO")
    print("=" * 60)

    return unique_papers


if __name__ == "__main__":
    fetch_all()