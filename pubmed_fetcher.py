"""
PubMed Fetcher v2.0 — EXTENDED COVERAGE
========================================

Recupera papers científicos de PubMed sobre TODO el espectro de
salud hormonal femenina y los guarda en JSON para usar como
knowledge base del RAG agent.

CHANGES v2.0 (over v1):
✅ Expanded from 3 queries to 15 queries
✅ Coverage: menstrual cycle, fertility, perimenopause, menopause,
   postmenopause, physical symptoms (skin, heat, fatigue, pain),
   cognitive function, immune, cardiovascular, bone health,
   sexuality, weight, thyroid, hormonal mood disorders
✅ Expected output: ~1500-2000 unique papers (vs. 484 previously)
✅ Better deduplication and metadata
"""

import requests
import xml.etree.ElementTree as ET
import json
import time
import os
from typing import List, Dict


NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
API_KEY = ""  # Optional: NCBI API key for higher rate limits
OUTPUT_FILE = "agents/pubmed_knowledge_base.json"
MAX_PAPERS_PER_QUERY = 200


# ─────────────────────────────────────────────────────────────────────────
# 15 QUERIES — Comprehensive coverage of female hormonal health
# ─────────────────────────────────────────────────────────────────────────
PUBMED_QUERIES = [
    # ── ORIGINAL 3 QUERIES (kept) ───────────────────────────────────────
    {
        "id": "query_01",
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
        "id": "query_02",
        "description": "Menopausia - Psicología",
        "query": (
            '("menopause/psychology"[MeSH Terms]) AND '
            '((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]) AND (middleaged[Filter]))'
        )
    },
    {
        "id": "query_03",
        "description": "Menopausia - Fisiología",
        "query": (
            '("menopause/physiology"[MeSH Terms]) AND '
            '((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]) AND (middleaged[Filter]))'
        )
    },

    # ── NEW QUERIES — PHYSICAL SYMPTOMS ─────────────────────────────────
    {
        "id": "query_04",
        "description": "Sofocos y síntomas vasomotores",
        "query": (
            '("Hot Flashes"[MeSH] OR "vasomotor symptoms"[tiab] OR "night sweats"[tiab]) '
            'AND ("Menopause"[MeSH] OR menopaus*[tiab] OR perimenopaus*[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },
    {
        "id": "query_05",
        "description": "Cambios cutáneos y dérmicos en menopausia",
        "query": (
            '("Skin Aging"[MeSH] OR "skin"[tiab] OR "pruritus"[tiab] OR "dryness"[tiab] OR "dermat*"[tiab]) '
            'AND ("Menopause"[MeSH] OR menopaus*[tiab] OR estrogen*[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },
    {
        "id": "query_06",
        "description": "Salud ósea, osteoporosis y densidad mineral",
        "query": (
            '("Osteoporosis"[MeSH] OR "Bone Density"[MeSH] OR "osteoporo*"[tiab] OR "bone health"[tiab]) '
            'AND ("Menopause"[MeSH] OR "Postmenopause"[MeSH] OR menopaus*[tiab] OR postmenopaus*[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },
    {
        "id": "query_07",
        "description": "Salud cardiovascular en menopausia",
        "query": (
            '("Cardiovascular Diseases"[MeSH] OR "Hypertension"[MeSH] OR "cardiovascular"[tiab] OR "blood pressure"[tiab]) '
            'AND ("Menopause"[MeSH] OR "Postmenopause"[MeSH] OR menopaus*[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },

    # ── HORMONE THERAPY / TREATMENTS ────────────────────────────────────
    {
        "id": "query_08",
        "description": "Terapia hormonal sustitutiva",
        "query": (
            '("Hormone Replacement Therapy"[MeSH] OR "Estrogen Replacement Therapy"[MeSH] OR '
            '"hormone therapy"[tiab] OR "hormone replacement"[tiab] OR HRT[tiab] OR MHT[tiab]) '
            'AND ("Menopause"[MeSH] OR menopaus*[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },

    # ── REPRODUCTIVE HEALTH / MENSTRUAL CYCLE ───────────────────────────
    {
        "id": "query_09",
        "description": "Ciclo menstrual, síndrome premenstrual y trastornos",
        "query": (
            '("Menstrual Cycle"[MeSH] OR "Premenstrual Syndrome"[MeSH] OR '
            '"Premenstrual Dysphoric Disorder"[MeSH] OR "Dysmenorrhea"[MeSH] OR '
            '"Menstruation Disturbances"[MeSH] OR PMS[tiab] OR PMDD[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },
    {
        "id": "query_10",
        "description": "Endometriosis, miomas y patología uterina",
        "query": (
            '("Endometriosis"[MeSH] OR "Leiomyoma"[MeSH] OR "Uterine Neoplasms"[MeSH] OR '
            '"endometrios*"[tiab] OR "fibroid*"[tiab] OR "myoma*"[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },
    {
        "id": "query_11",
        "description": "Síndrome de ovario poliquístico (SOP/PCOS)",
        "query": (
            '("Polycystic Ovary Syndrome"[MeSH] OR "PCOS"[tiab] OR "polycystic ovary"[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },

    # ── THYROID & METABOLIC ─────────────────────────────────────────────
    {
        "id": "query_12",
        "description": "Salud tiroidea y peso en mujeres",
        "query": (
            '("Thyroid Diseases"[MeSH] OR "Hypothyroidism"[MeSH] OR "Hyperthyroidism"[MeSH] OR '
            '"Hashimoto Disease"[MeSH] OR thyroid*[tiab]) '
            'AND ("Menopause"[MeSH] OR "menopause"[tiab] OR "menstrual"[tiab] OR "women"[tiab] OR "female"[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },

    # ── SEXUALITY & GENITOURINARY ──────────────────────────────────────
    {
        "id": "query_13",
        "description": "Sexualidad, libido y salud genitourinaria en menopausia",
        "query": (
            '("Libido"[MeSH] OR "Sexual Dysfunction, Physiological"[MeSH] OR '
            '"Vaginal Atrophy"[tiab] OR "vaginal dryness"[tiab] OR "libido"[tiab] OR '
            '"sexual function"[tiab] OR "GSM"[tiab] OR "genitourinary syndrome"[tiab]) '
            'AND ("Menopause"[MeSH] OR menopaus*[tiab] OR postmenopaus*[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },

    # ── COGNITION & BRAIN FOG ──────────────────────────────────────────
    {
        "id": "query_14",
        "description": "Niebla mental, función cognitiva y memoria en menopausia",
        "query": (
            '("Cognitive Dysfunction"[MeSH] OR "Memory Disorders"[MeSH] OR '
            '"brain fog"[tiab] OR "cognitive complaints"[tiab] OR "memory complaints"[tiab] OR '
            '"executive function"[tiab] OR "concentration"[tiab]) '
            'AND ("Menopause"[MeSH] OR menopaus*[tiab] OR perimenopaus*[tiab] OR estrogen*[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },

    # ── LIFESTYLE & PREVENTION ─────────────────────────────────────────
    {
        "id": "query_15",
        "description": "Nutrición, ejercicio y prevención en menopausia",
        "query": (
            '("Diet"[MeSH] OR "Exercise"[MeSH] OR "Nutritional Status"[MeSH] OR '
            '"lifestyle"[tiab] OR "physical activity"[tiab] OR "nutrition"[tiab] OR '
            '"weight management"[tiab] OR "supplementation"[tiab]) '
            'AND ("Menopause"[MeSH] OR "Postmenopause"[MeSH] OR menopaus*[tiab]) '
            'AND ((y_5[Filter]) AND (humans[Filter]) AND (female[Filter]))'
        )
    },
]


# ─────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────

def search_pubmed(query: str, max_results: int = MAX_PAPERS_PER_QUERY) -> List[str]:
    """Search PubMed and return list of PMIDs."""
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
    """Given a list of PMIDs, return full paper details."""
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

            # Rate limiting (3 req/sec without API key, 10 with)
            sleep_time = 0.15 if API_KEY else 0.4
            time.sleep(sleep_time)

        except Exception as e:
            print(f"  ✗ Error en efetch batch {i//batch_size + 1}: {e}")

    return all_papers


def parse_pubmed_xml(xml_text: str) -> List[Dict]:
    """Parse PubMed XML and extract relevant fields."""
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
        if title_el is not None:
            paper["title"] = ET.tostring(title_el, encoding="unicode", method="text")
        else:
            paper["title"] = ""

        # Extract abstract (concatenating labelled sections if present)
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

        # Skip papers without abstract
        if not paper["abstract"]:
            continue

        # Authors
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

        # Year
        year_el = article.find(".//PubDate/Year")
        if year_el is None:
            year_el = article.find(".//PubDate/MedlineDate")
        paper["year"] = year_el.text[:4] if year_el is not None else "Unknown"

        # Journal
        journal_el = article.find(".//Journal/Title")
        if journal_el is None:
            journal_el = article.find(".//MedlineTA")
        paper["journal"] = journal_el.text if journal_el is not None else "Unknown"

        # DOI
        doi_el = article.find(".//ArticleId[@IdType='doi']")
        paper["doi"] = doi_el.text if doi_el is not None else ""

        # URL
        paper["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{paper['pmid']}/"

        # Citation
        paper["citation"] = (
            f"{paper['authors_str']} ({paper['year']}). "
            f"{paper['title']}. {paper['journal']}."
        )

        papers.append(paper)

    return papers


def deduplicate_papers(papers: List[Dict]) -> List[Dict]:
    """Remove duplicate papers (same PMID from different queries)."""
    seen_pmids = set()
    unique_papers = []
    for paper in papers:
        if paper["pmid"] not in seen_pmids:
            seen_pmids.add(paper["pmid"])
            unique_papers.append(paper)
    return unique_papers


def save_knowledge_base(papers: List[Dict], output_file: str = OUTPUT_FILE):
    """Save papers to JSON for use as knowledge base."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    kb = {
        "metadata": {
            "total_papers": len(papers),
            "queries_used": [q["description"] for q in PUBMED_QUERIES],
            "num_queries": len(PUBMED_QUERIES),
            "source": "PubMed (NCBI)",
            "filters": "Last 5 years, humans, female",
            "fetcher_version": "v2.0",
            "domain": "female_hormonal_health",
        },
        "papers": papers
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Knowledge base guardada: {output_file}")
    print(f"   Total papers: {len(papers)}")


# ─────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────

def fetch_all():
    print("=" * 70)
    print(f"PUBMED FETCHER v2.0 — {len(PUBMED_QUERIES)} queries de salud hormonal femenina")
    print("=" * 70)

    all_papers = []

    for i, query_config in enumerate(PUBMED_QUERIES, 1):
        print(f"\n[Query {i}/{len(PUBMED_QUERIES)}] {query_config['description']}")
        print(f"  Buscando en PubMed...")

        pmids = search_pubmed(query_config["query"])

        if not pmids:
            print(f"  ✗ No se encontraron resultados")
            continue

        papers = fetch_paper_details(pmids)
        print(f"  ✓ Papers con abstract: {len(papers)}")

        for paper in papers:
            paper["source_query"] = query_config["id"]
            paper["source_description"] = query_config["description"]

        all_papers.extend(papers)

        # Pause between queries to be nice to the API
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

    print("\n" + "=" * 70)
    print(f"✅ FETCH COMPLETO — {len(unique_papers)} papers únicos")
    print("=" * 70)

    return unique_papers


if __name__ == "__main__":
    fetch_all()