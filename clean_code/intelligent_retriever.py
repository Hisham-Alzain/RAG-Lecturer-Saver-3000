import re
import streamlit as st
from config import Config
from sentence_transformers import SentenceTransformer
from llm_client import LLMClient
from loaders import load_reranker
from languageUtils import LanguageUtils
from collections import defaultdict
from typing import List, Dict, Tuple


class IntelligentRetriever:
    """
    Smart retrieval with query understanding and multi-strategy search.

    Key features:
    1. Query analysis - understands question type (factual, comparison, etc.)
    2. Query expansion - generates related queries for better coverage
    3. Multi-vector search - searches with multiple query formulations
    4. Cross-encoder reranking - ensures relevance
    5. Parent context retrieval - returns richer context
    """

    def __init__(
        self, embedder: SentenceTransformer, collection, llm_client: LLMClient
    ):
        self.embedder = embedder
        self.collection = collection
        self.llm_client = llm_client
        self.reranker = None

        if Config.ENABLE_RERANKING:
            try:
                self.reranker = load_reranker()
            except Exception as e:
                st.warning(f"Reranker not loaded: {e}")

    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query to understand what information is needed.
        Returns query type and extracted entities/topics.
        """
        query_lower = query.lower()
        lang = LanguageUtils.detect_language(query)

        analysis = {
            "lang": lang,
            "type": "factual",  # default
            "topics": [],
            "is_comparison": False,
            "is_definition": False,
            "is_explanation": False,
            "is_list": False,
        }

        # Detect comparison questions
        comparison_patterns_en = [
            r"compare\s+(.+?)\s+(?:and|with|to|vs\.?|versus)\s+(.+)",
            r"difference(?:s)?\s+between\s+(.+?)\s+(?:and|&)\s+(.+)",
            r"(.+?)\s+vs\.?\s+(.+)",
            r"contrast\s+(.+?)\s+(?:and|with)\s+(.+)",
            r"how\s+(?:does|do|is|are)\s+(.+?)\s+differ\s+from\s+(.+)",
        ]
        comparison_patterns_ar = [
            r"قارن\s+(?:بين\s+)?(.+?)\s+(?:و|مع)\s+(.+)",
            r"الفرق\s+بين\s+(.+?)\s+(?:و|&)\s+(.+)",
            r"ما\s+(?:هو\s+)?الفرق\s+بين\s+(.+?)\s+(?:و|&)\s+(.+)",
            r"مقارنة\s+(?:بين\s+)?(.+?)\s+(?:و|مع)\s+(.+)",
        ]

        patterns = comparison_patterns_ar if lang == "ar" else comparison_patterns_en

        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                analysis["is_comparison"] = True
                analysis["type"] = "comparison"
                topic1 = match.group(1).strip().strip("?.,")
                topic2 = match.group(2).strip().strip("?.,")
                # Clean articles
                topic1 = re.sub(r"^(the|a|an)\s+", "", topic1, flags=re.IGNORECASE)
                topic2 = re.sub(r"^(the|a|an)\s+", "", topic2, flags=re.IGNORECASE)
                if topic1 and topic2:
                    analysis["topics"] = [topic1, topic2]
                break

        # Detect definition questions
        if not analysis["is_comparison"]:
            definition_patterns = [
                r"what\s+is\s+(?:a\s+|an\s+|the\s+)?(.+)",
                r"define\s+(.+)",
                r"ما\s+(?:هو|هي)\s+(.+)",
                r"تعريف\s+(.+)",
                r"عرف\s+(.+)",
            ]
            for pattern in definition_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    analysis["is_definition"] = True
                    analysis["type"] = "definition"
                    topic = match.group(1).strip().strip("?.,")
                    if topic:
                        analysis["topics"] = [topic]
                    break

        # Detect explanation questions
        if not analysis["topics"]:
            explanation_patterns = [
                r"how\s+(?:does|do|can|to)\s+(.+)",
                r"explain\s+(.+)",
                r"why\s+(?:does|do|is|are)\s+(.+)",
                r"كيف\s+(.+)",
                r"اشرح\s+(.+)",
                r"لماذا\s+(.+)",
            ]
            for pattern in explanation_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    analysis["is_explanation"] = True
                    analysis["type"] = "explanation"
                    topic = match.group(1).strip().strip("?.,")
                    if topic:
                        analysis["topics"] = [topic]
                    break

        # Detect list questions
        list_patterns = [
            r"(?:what|list|name)\s+(?:are\s+)?(?:the\s+)?(?:different\s+)?(?:types|kinds|methods|ways|steps)",
            r"ما\s+(?:هي\s+)?(?:أنواع|طرق|خطوات)",
        ]
        for pattern in list_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                analysis["is_list"] = True
                analysis["type"] = "list"
                break

        # Extract keywords if no topics found
        if not analysis["topics"]:
            analysis["topics"] = LanguageUtils.extract_keywords(query)[:5]

        return analysis

    def generate_search_queries(self, query: str, analysis: Dict) -> List[str]:
        """Generate multiple search queries for comprehensive retrieval."""
        queries = [query]  # Always include original
        lang = analysis["lang"]

        # For comparison questions, search each topic separately
        if analysis["is_comparison"] and len(analysis["topics"]) >= 2:
            for topic in analysis["topics"]:
                if lang == "ar":
                    queries.extend(
                        [
                            f"ما هو {topic}",
                            f"تعريف {topic}",
                            topic,
                        ]
                    )
                else:
                    queries.extend(
                        [
                            f"what is {topic}",
                            f"{topic} definition",
                            f"{topic} explanation",
                            topic,
                        ]
                    )

        # For definition questions
        elif analysis["is_definition"] and analysis["topics"]:
            topic = analysis["topics"][0]
            if lang == "ar":
                queries.extend(
                    [
                        topic,
                        f"تعريف {topic}",
                        f"شرح {topic}",
                    ]
                )
            else:
                queries.extend(
                    [
                        topic,
                        f"{topic} definition",
                        f"{topic} meaning",
                        f"define {topic}",
                    ]
                )

        # For other questions, add keyword-based queries
        else:
            keywords = analysis["topics"][:3]
            if keywords:
                queries.append(" ".join(keywords))

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)

        return unique_queries[:8]  # Max 8 queries

    def vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform vector similarity search."""
        # Format for E5 model
        query_embedding = self.embedder.encode(
            [f"query: {query}"], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding, n_results=top_k
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        return [
            {
                "id": doc_id,
                "document": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1.0 - dist,  # Convert distance to similarity
            }
            for doc_id, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def merge_results(self, results_list: List[List[Dict]]) -> List[Dict]:
        """Merge results from multiple queries using RRF."""
        scores = defaultdict(float)
        doc_data = {}
        k = 60  # RRF constant

        for results in results_list:
            for rank, result in enumerate(results):
                doc_id = result["id"]
                scores[doc_id] += 1.0 / (k + rank + 1)
                if doc_id not in doc_data:
                    doc_data[doc_id] = result

        # Sort by RRF score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [
            {**doc_data[doc_id], "rrf_score": scores[doc_id]}
            for doc_id in sorted_ids
            if doc_id in doc_data
        ]

    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Rerank candidates using cross-encoder."""
        if not self.reranker or not candidates:
            return candidates[:top_k]

        # Prepare pairs for reranking
        pairs = []
        for c in candidates[: Config.RERANK_CANDIDATES]:
            # Use parent text for richer context
            text = c["metadata"].get("parent_text", c["document"])
            # Truncate if too long
            if len(text) > 512:
                text = text[:512]
            pairs.append([query, text])

        # Get reranker scores
        scores = self.reranker.predict(pairs)

        # Attach scores and sort
        for i, c in enumerate(candidates[: len(scores)]):
            c["rerank_score"] = float(scores[i])

        candidates[: len(scores)] = sorted(
            candidates[: len(scores)],
            key=lambda x: x.get("rerank_score", -float("inf")),
            reverse=True,
        )

        return candidates[:top_k]

    def deduplicate(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks from same parent."""
        seen_parents = set()
        unique = []

        for r in results:
            parent_id = f"{r['metadata'].get('source', '')}_{r['metadata'].get('page', '')}_{r['metadata'].get('parent_idx', '')}"
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                unique.append(r)

        return unique

    def retrieve(self, query: str) -> Tuple[List[Dict], Dict]:
        """
        Main retrieval pipeline.
        Returns (results, debug_info)
        """
        debug = {
            "query_type": "unknown",
            "topics": [],
            "num_queries": 0,
            "num_candidates": 0,
        }

        # Step 1: Analyze query
        analysis = self.analyze_query(query)
        debug["query_type"] = analysis["type"]
        debug["topics"] = analysis["topics"]

        # Step 2: Generate search queries
        search_queries = self.generate_search_queries(query, analysis)
        debug["num_queries"] = len(search_queries)

        # Step 3: Search with each query
        all_results = []
        for q in search_queries:
            results = self.vector_search(q, Config.RETRIEVAL_K)
            if results:
                all_results.append(results)

        if not all_results:
            return [], debug

        # Step 4: Merge results
        merged = self.merge_results(all_results)
        debug["num_candidates"] = len(merged)

        # Step 5: Deduplicate
        deduped = self.deduplicate(merged)

        # Step 6: Rerank
        reranked = self.rerank(query, deduped, Config.TOP_K)

        return reranked, debug