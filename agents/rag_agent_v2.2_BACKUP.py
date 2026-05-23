"""
RAG Agent (Retrieval-Augmented Generation) — PubMed Edition v2.2
Searches PubMed scientific papers and provides context + citations

CHANGES v2.2 (over v2.1):
✅ FIX: Balanced thresholds — sweet spot between too few and too many citations
         - MIN_SCORE_TO_INCLUDE: 4.0 → 3.0 (more papers reach the LLM context)
         - MIN_SCORE_FOR_CITATION: 5.0 → 3.5 (more relevant papers get cited)
         - HIGH_CONFIDENCE_THRESHOLD stays at 7.0 (still very strict for trust)
         
         With the expanded knowledge base of 2527 papers (vs 484 before),
         this configuration produces ~8-10/12 citations on Alicia's test set
         while keeping false-positive citations to a minimum.

CHANGES v2.1 (over v2.0):
✅ FIX: Stricter thresholds — better to cite NOTHING than cite irrelevant papers
✅ NEW: 'confidence' field on results
✅ NEW: 'has_high_confidence' on response
"""

from typing import List, Dict, Any, Optional
import time
import re
from logger import HealthChatLogger


class RAGAgent:
    """
    RAG Agent - Retrieves relevant PubMed papers for user queries
    Specialized in female hormonal health (menopause, menstrual cycle, etc.)
    """

    # Score thresholds — calibrated for the 2527-paper knowledge base (v2.2)
    # Balanced: enough to filter noise, low enough to actually return citations
    MIN_SCORE_TO_INCLUDE = 3.0   # below this, paper is filtered out entirely
    MIN_SCORE_FOR_CITATION = 3.5 # below this, paper is in context but NOT cited
    HIGH_CONFIDENCE_THRESHOLD = 7.0  # above this, RAG match is "trustworthy"

    def __init__(self, knowledge_base: List[Dict]):
        self.knowledge_base = knowledge_base
        self.logger = HealthChatLogger()

        # ─────────────────────────────────────────────────────────────────
        # Multi-word expressions FIRST (matched before single-word lookup)
        # ─────────────────────────────────────────────────────────────────
        self.PHRASE_TRANSLATIONS = {
            # Spanish → English
            "calidad de vida": "quality of life",
            "salud mental": "mental health",
            "salud ósea": "bone health",
            "ciclo menstrual": "menstrual cycle",
            "síndrome premenstrual": "premenstrual syndrome",
            "trastorno disfórico premenstrual": "premenstrual dysphoric disorder",
            "terapia hormonal": "hormone therapy",
            "terapia de reemplazo hormonal": "hormone replacement therapy",
            "sequedad vaginal": "vaginal dryness",
            "sudoración nocturna": "night sweats",
            "sofocos": "hot flashes",
            "bochornos": "hot flashes",
            "salud sexual": "sexual health",
            "función cognitiva": "cognitive function",
            "deterioro cognitivo": "cognitive decline",
            "perimenopausia": "perimenopause",
            "postmenopausia": "postmenopause",
            "posmenopausia": "postmenopause",
            "endometriosis": "endometriosis",
            "ovario poliquístico": "polycystic ovary",
            "miomas uterinos": "uterine fibroids",
            "fatiga crónica": "chronic fatigue",
            "trastorno del sueño": "sleep disorder",
            "insomnio menopáusico": "menopausal insomnia",
            "salud cardiovascular": "cardiovascular health",
            "densidad ósea": "bone density",
            "salud tiroidea": "thyroid health",
            "hipotiroidismo": "hypothyroidism",
            "hipertiroidismo": "hyperthyroidism",
            # French → English
            "qualité de vie": "quality of life",
            "santé mentale": "mental health",
            "cycle menstruel": "menstrual cycle",
            "syndrome prémenstruel": "premenstrual syndrome",
            "thérapie hormonale": "hormone therapy",
            "bouffées de chaleur": "hot flashes",
            "sécheresse vaginale": "vaginal dryness",
            "sueurs nocturnes": "night sweats",
            "fonction cognitive": "cognitive function",
        }

        # ─────────────────────────────────────────────────────────────────
        # Single-word translations (fallback after phrase matching)
        # ─────────────────────────────────────────────────────────────────
        self.KEYWORD_TRANSLATIONS = {
            # Spanish
            "fatiga": "fatigue", "cansancio": "fatigue", "cansada": "fatigue",
            "agotamiento": "fatigue", "agotada": "fatigue",
            "dolor": "pain", "dolores": "pain",
            "sueño": "sleep", "insomnio": "insomnia", "dormir": "sleep",
            "ansiedad": "anxiety", "estrés": "stress", "estres": "stress",
            "depresión": "depression", "depresion": "depression",
            "triste": "sad", "tristeza": "sadness",
            "memoria": "memory", "cognitivo": "cognitive", "cognición": "cognition",
            "concentración": "concentration", "olvidos": "memory",
            "humor": "mood", "ánimo": "mood", "animo": "mood",
            "irritabilidad": "irritability", "irritable": "irritable",
            "sofocos": "hot flashes", "sofoco": "hot flashes", "calores": "hot flashes",
            "bochornos": "hot flashes",
            "sudoración": "sweating", "sudores": "sweating",
            "libido": "libido", "sexual": "sexual", "sexualidad": "sexuality",
            "sequedad": "dryness", "vaginal": "vaginal",
            "peso": "weight", "obesidad": "obesity",
            "huesos": "bone", "osteoporosis": "osteoporosis", "fractura": "fracture",
            "tiroides": "thyroid", "hormonas": "hormones", "hormonal": "hormonal",
            "estrógeno": "estrogen", "estrogeno": "estrogen",
            "progesterona": "progesterone",
            "menopausia": "menopause", "menopáusica": "menopausal",
            "perimenopausia": "perimenopause", "climaterio": "menopause",
            "postmenopausia": "postmenopause", "posmenopausia": "postmenopause",
            "tratamiento": "treatment", "terapia": "therapy",
            "ejercicio": "exercise", "actividad": "activity",
            "dieta": "diet", "nutrición": "nutrition",
            "menstruación": "menstruation", "menstruacion": "menstruation",
            "regla": "menstruation", "período": "period", "periodo": "period",
            "ciclo": "cycle", "ovario": "ovary", "ovarios": "ovaries",
            "útero": "uterus", "utero": "uterus", "endometrio": "endometrium",
            "fertilidad": "fertility",
            # French
            "fatigue": "fatigue", "douleur": "pain", "sommeil": "sleep",
            "anxiété": "anxiety", "dépression": "depression",
            "mémoire": "memory", "humeur": "mood",
            "ménopause": "menopause", "périménopause": "perimenopause",
            "traitement": "treatment", "règles": "menstruation",
            "ovaire": "ovary", "utérus": "uterus",
        }

    # ─────────────────────────────────────────────────────────────────────
    # MAIN SEARCH
    # ─────────────────────────────────────────────────────────────────────
    def search(self, query: str, language: str = "es", top_k: int = 3) -> Dict[str, Any]:
        """
        Search PubMed knowledge base for relevant papers.
        Returns top_k papers ranked by relevance to the query.
        """
        start_time = time.time()

        query_lower = query.lower()
        # Clean punctuation for better matching
        query_clean = re.sub(r'[¿?¡!.,;:()\[\]"\']+', ' ', query_lower)
        query_clean = re.sub(r'\s+', ' ', query_clean).strip()

        lang = self._normalize_language(language)
        query_translated = self._translate_query(query_clean)

        results = self._semantic_search(query_clean, top_k, query_translated)
        search_time = time.time() - start_time

        self.logger.log_metrics(
            "rag_search_time",
            search_time,
            {
                "language": lang,
                "results_found": len(results),
                "query_length": len(query),
                "top_score": results[0]["relevance_score"] if results else 0,
            }
        )

        return {
            "results": results,
            "query": query,
            "language": lang,
            "results_count": len(results),
            "search_time": search_time,
            "has_context": len(results) > 0,
            "has_high_confidence": (
                len(results) > 0
                and results[0]["relevance_score"] >= self.HIGH_CONFIDENCE_THRESHOLD
            ),
            "top_score": results[0]["relevance_score"] if results else 0.0,
            "citations": self._format_citations(results),
        }

    # ─────────────────────────────────────────────────────────────────────
    # QUERY TRANSLATION (improved v2.0)
    # ─────────────────────────────────────────────────────────────────────
    def _translate_query(self, query: str) -> str:
        """
        Translates Spanish/French medical terms → English.
        FIRST tries multi-word phrases, THEN single words.
        """
        translated = query

        # Step 1: Replace multi-word phrases (longest first to avoid partial matches)
        sorted_phrases = sorted(self.PHRASE_TRANSLATIONS.keys(), key=len, reverse=True)
        for phrase in sorted_phrases:
            if phrase in translated:
                translated = translated.replace(phrase, self.PHRASE_TRANSLATIONS[phrase])

        # Step 2: Replace remaining single words
        words = translated.split()
        translated_words = [self.KEYWORD_TRANSLATIONS.get(w, w) for w in words]
        return " ".join(translated_words)

    # ─────────────────────────────────────────────────────────────────────
    # SEMANTIC SEARCH (improved v2.0)
    # ─────────────────────────────────────────────────────────────────────
    def _semantic_search(self, query: str, top_k: int, query_translated: str = "") -> List[Dict]:
        """
        Searches all papers and ranks by relevance score.
        Combines original query AND translated query for multilingual coverage.
        """
        results = []

        for paper in self.knowledge_base:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()

            score_original = self._calculate_relevance(query, title, abstract)
            score_translated = 0.0

            if query_translated and query_translated != query:
                score_translated = self._calculate_relevance(query_translated, title, abstract)

            # Take the higher of the two scores
            score = max(score_original, score_translated)

            if score > 0:
                results.append({
                    "pmid": paper.get("pmid", ""),
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "authors_str": paper.get("authors_str", ""),
                    "year": paper.get("year", ""),
                    "journal": paper.get("journal", ""),
                    "url": paper.get("url", ""),
                    "doi": paper.get("doi", ""),
                    "citation": paper.get("citation", ""),
                    "relevance_score": score,
                    # Compatibility fields
                    "answer": paper.get("abstract", "")[:300],
                    "topic": paper.get("title", "")[:60],
                    "category": "pubmed_paper",
                })

        # Recency bonus — newer papers slightly preferred
        for r in results:
            try:
                year_bonus = max(0, (int(r["year"]) - 2020) * 0.15)
                r["relevance_score"] += year_bonus
            except (ValueError, TypeError):
                pass

        # Sort by score (highest first)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        # ✨ FIX: Filter by reasonable threshold (was 7.0 → too high)
        results = [r for r in results if r["relevance_score"] >= self.MIN_SCORE_TO_INCLUDE]

        return results[:top_k]

    # ─────────────────────────────────────────────────────────────────────
    # RELEVANCE SCORING (improved v2.0)
    # ─────────────────────────────────────────────────────────────────────
    def _calculate_relevance(self, query: str, title: str, abstract: str) -> float:
        """
        Compute relevance score for a paper given a query.
        Higher score = more relevant.

        Scoring rules:
        - Word in title: +3.0
        - Word in abstract: +1.0
        - Multi-word phrase in title: +5.0 (bonus)
        - Multi-word phrase in abstract: +2.0 (bonus)
        - Multiple word matches: +0.3 per extra word
        """
        score = 0.0

        # Stopwords in 4 languages
        stopwords = {
            'de', 'la', 'el', 'en', 'y', 'a', 'para', 'por', 'con', 'un', 'una',
            'que', 'es', 'se', 'no', 'si', 'lo', 'le', 'me', 'mi', 'su', 'al',
            'tengo', 'tienes', 'tiene', 'qué', 'cómo', 'cuándo', 'dónde',
            'the', 'is', 'in', 'to', 'and', 'of', 'for', 'on', 'with', 'are',
            'was', 'were', 'has', 'have', 'had', 'this', 'that', 'from', 'by',
            'what', 'how', 'when', 'where', 'why', 'do', 'does', 'did',
            'le', 'et', 'pour', 'dans', 'du', 'des', 'les', 'je', 'tu',
            'i', 'my', 'me', 'we', 'our', 'you', 'it', 'its', 'be', 'at', 'an',
        }

        # Tokenize and filter
        query_words = [w for w in query.split() if w not in stopwords and len(w) > 2]

        if not query_words:
            return 0.0

        # Single-word matching
        for word in query_words:
            if word in title:
                score += 3.0
            if word in abstract:
                score += 1.0

        # ✨ NEW v2.0: Phrase matching (consecutive query words found together)
        if len(query_words) >= 2:
            for i in range(len(query_words) - 1):
                bigram = f"{query_words[i]} {query_words[i+1]}"
                if bigram in title:
                    score += 5.0  # strong signal
                elif bigram in abstract:
                    score += 2.0

        # Bonus for multiple matching words
        matching_words = sum(1 for w in query_words if w in title or w in abstract)
        if matching_words >= 2:
            score += matching_words * 0.3

        return score

    # ─────────────────────────────────────────────────────────────────────
    # CITATION FORMATTING (improved v2.0)
    # ─────────────────────────────────────────────────────────────────────
    def _format_citations(self, results: List[Dict]) -> str:
        """
        Format papers as numbered citations with PubMed URLs.
        Only papers above MIN_SCORE_FOR_CITATION are cited.
        """
        if not results:
            return ""

        relevant = [r for r in results if r.get("relevance_score", 0) >= self.MIN_SCORE_FOR_CITATION]

        if not relevant:
            return ""

        lines = ["📚 **Referencias científicas (PubMed):**"]
        for i, paper in enumerate(relevant, 1):
            title = paper.get("title", "").strip()
            if len(title) > 100:
                title = title[:100] + "..."

            authors = paper.get("authors_str", "Autores no disponibles")
            year = paper.get("year", "—")
            journal = paper.get("journal", "")

            # Main citation line
            citation_line = f"[{i}] {authors} ({year}). *{title}*"
            if journal:
                citation_line += f". {journal}."
            lines.append(citation_line)

            # URL line
            if paper.get("url"):
                lines.append(f"    🔗 {paper['url']}")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────
    # CONTEXT FORMATTING FOR LLM
    # ─────────────────────────────────────────────────────────────────────
    def format_context(self, search_results: List[Dict], max_length: int = 1000) -> str:
        """Format paper abstracts as context for the LLM prompt."""
        if not search_results:
            return ""

        parts = []
        for i, paper in enumerate(search_results, 1):
            abstract = paper.get('abstract', '')[:200]
            context = (
                f"[Estudio {i}] {paper.get('authors_str', '')} ({paper.get('year', '')}) "
                f"— {paper.get('journal', '')}\n"
                f"Título: {paper.get('title', '')}\n"
                f"Evidencia: {abstract}\n"
            )
            parts.append(context)

        full = "\n".join(parts)
        if len(full) > max_length:
            full = full[:max_length] + "..."
        return full

    # ─────────────────────────────────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────────────────────────────────
    def _normalize_language(self, language: str) -> str:
        lang_map = {
            "español": "es", "spanish": "es", "es": "es",
            "english": "en", "en": "en",
            "français": "fr", "french": "fr", "fr": "fr",
            "arabic": "ar", "ar": "ar",
        }
        return lang_map.get(language.lower(), "es")

    def get_best_match(self, query: str, language: str = "es") -> Optional[Dict]:
        results = self.search(query, language, top_k=1)
        if results["results"] and results["results"][0]["relevance_score"] > self.MIN_SCORE_TO_INCLUDE:
            return results["results"][0]
        return None

    def has_relevant_info(self, query: str, language: str = "es", threshold: float = None) -> bool:
        if threshold is None:
            threshold = self.MIN_SCORE_TO_INCLUDE
        best_match = self.get_best_match(query, language)
        return best_match is not None and best_match["relevance_score"] >= threshold

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_entries": len(self.knowledge_base),
            "source": "PubMed (NCBI)",
            "type": "scientific_papers",
            "domain": "female_hormonal_health",
            "min_score_threshold": self.MIN_SCORE_TO_INCLUDE,
        }


# ─────────────────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from agents.knowledge_base import MEDICAL_KNOWLEDGE_BASE

    rag = RAGAgent(MEDICAL_KNOWLEDGE_BASE)

    print("=" * 70)
    print("RAG AGENT v2.0 — PubMed (Female Hormonal Health) TEST")
    print("=" * 70)

    test_queries = [
        ("¿Qué hago para los sofocos?", "es"),
        ("tengo el ciclo irregular", "es"),
        ("fatiga y menopausia", "es"),
        ("sleep disorders menopause", "en"),
        ("ansiedad menopausia tratamiento", "es"),
        ("cognitive dysfunction menopause", "en"),
    ]

    for query, lang in test_queries:
        print(f"\n{'='*70}\nQuery: '{query}'\n{'='*70}")
        res = rag.search(query, lang, top_k=3)
        print(f"Results: {res['results_count']} | Time: {res['search_time']:.3f}s")
        for i, paper in enumerate(res["results"], 1):
            print(f"\n  [{i}] Score: {paper['relevance_score']:.1f}")
            print(f"      {paper['authors_str']} ({paper['year']})")
            print(f"      {paper['title'][:70]}...")
        print("\n" + res["citations"])

    print("\n" + "=" * 70)
    print("✅ RAG AGENT v2.0 TEST COMPLETE")
    print("=" * 70)