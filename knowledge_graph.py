"""
Knowledge Graph builder and query engine for CogniCove.

Uses NetworkX to build an in-memory graph (~200 nodes) from kg_data.py.
Provides symptom matching (cosine similarity with the same MiniLM embeddings),
comorbidity look-ups, exclusion checks, and differential retrieval.
"""

import networkx as nx
import numpy as np
from functools import lru_cache
from typing import List, Dict, Tuple, Optional
from langchain_huggingface import HuggingFaceEmbeddings

from kg_data import (
    CATEGORY_MAP,
    COMORBIDITY_EDGES,
    DISORDER_CRITERIA,
    EXCLUSION_LINKS,
    DIFFERENTIAL_PAIRS,
)
from constants import EMBEDDING_MODEL_NAME


# ───────────────────────── Graph Construction ─────────────────────────

@lru_cache(maxsize=1)
def build_knowledge_graph() -> nx.DiGraph:
    """Build the full KG from kg_data.py definitions.

    Node types:
      - category   (e.g. "Depressive Disorders")
      - disorder   (e.g. "Major Depressive Disorder")
      - symptom    (e.g. "depressed mood")

    Edge types:
      - belongs_to         disorder → category
      - has_symptom        disorder → symptom
      - comorbid_with      disorder ↔ disorder
      - excludes           disorder → disorder / condition string
      - differential_with  disorder ↔ disorder
    """
    G = nx.DiGraph()

    # -- Categories --------------------------------------------------------
    categories = set(CATEGORY_MAP.values())
    for cat in categories:
        G.add_node(cat, node_type="category")

    # -- Disorders & belongs_to --------------------------------------------
    for disorder, cat in CATEGORY_MAP.items():
        G.add_node(disorder, node_type="disorder", category=cat)
        G.add_edge(disorder, cat, relation="belongs_to")

    # -- Symptoms & has_symptom --------------------------------------------
    for disorder, criteria in DISORDER_CRITERIA.items():
        for symptom in criteria.get("symptoms", []):
            sym_key = symptom.lower().strip()
            if not G.has_node(sym_key):
                G.add_node(sym_key, node_type="symptom")
            G.add_edge(disorder, sym_key, relation="has_symptom")

    # -- Comorbidity edges (bidirectional) ---------------------------------
    for src, tgt, weight, note in COMORBIDITY_EDGES:
        if src in G.nodes and tgt in G.nodes:
            G.add_edge(src, tgt, relation="comorbid_with", weight=weight, evidence=note)
            if not G.has_edge(tgt, src):
                G.add_edge(tgt, src, relation="comorbid_with", weight=weight, evidence=note)

    # -- Exclusion edges ---------------------------------------------------
    for disorder, exc_type, condition in EXCLUSION_LINKS:
        if not G.has_node(condition):
            G.add_node(condition, node_type="condition_string")
        G.add_edge(disorder, condition, relation="excludes", exc_type=exc_type)

    # -- Differential pairs (bidirectional) --------------------------------
    for a, b in DIFFERENTIAL_PAIRS:
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b, relation="differential_with")
            if not G.has_edge(b, a):
                G.add_edge(b, a, relation="differential_with")

    return G


# ───────────────────────── Embedding Cache ────────────────────────────

@lru_cache(maxsize=1)
def _get_embedding_model():
    """Return the same HuggingFace embedding model used by the vectorstore."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=2)
def _get_symptom_embeddings(_graph_nodes: tuple) -> Dict[str, np.ndarray]:
    """Pre-compute embeddings for every symptom node.

    _graph_nodes is a frozen tuple used only as a cache key so that
    Streamlit doesn't try to hash the whole graph.
    """
    model = _get_embedding_model()
    symptom_nodes = list(_graph_nodes)
    if not symptom_nodes:
        return {}
    vecs = model.embed_documents(symptom_nodes)
    return {s: np.array(v) for s, v in zip(symptom_nodes, vecs)}


def _symptom_nodes(G: nx.DiGraph) -> tuple:
    """Return a sorted tuple of all symptom node keys (hashable cache key)."""
    return tuple(sorted(n for n, d in G.nodes(data=True) if d.get("node_type") == "symptom"))


# ───────────────────────── Symptom Matching ───────────────────────────

def match_symptoms(
    G: nx.DiGraph,
    patient_symptoms: List[str],
    threshold: float = 0.65,
) -> Dict[str, List[Tuple[str, float]]]:
    """Match patient-reported symptoms to KG symptom nodes.

    Returns:
        dict mapping each patient symptom to a list of (kg_symptom, similarity)
        pairs that exceed *threshold*.
    """
    if not patient_symptoms:
        return {}

    model = _get_embedding_model()
    sym_nodes = _symptom_nodes(G)
    sym_embs = _get_symptom_embeddings(sym_nodes)

    if not sym_embs:
        return {}

    # Stack reference vectors once
    ref_keys = list(sym_embs.keys())
    ref_matrix = np.stack([sym_embs[k] for k in ref_keys])  # (N, D)
    ref_norms = np.linalg.norm(ref_matrix, axis=1, keepdims=True) + 1e-10
    ref_normed = ref_matrix / ref_norms

    # Encode patient symptoms
    patient_vecs = model.embed_documents(patient_symptoms)
    results: Dict[str, List[Tuple[str, float]]] = {}

    for psym, pvec in zip(patient_symptoms, patient_vecs):
        pvec = np.array(pvec)
        pnorm = np.linalg.norm(pvec) + 1e-10
        cosines = (ref_normed @ pvec) / pnorm  # (N,)
        matches = [
            (ref_keys[i], float(cosines[i]))
            for i in range(len(ref_keys))
            if cosines[i] >= threshold
        ]
        matches.sort(key=lambda x: x[1], reverse=True)
        if matches:
            results[psym] = matches

    return results


# ───────────────────────── Comorbidity ────────────────────────────────

def get_comorbid_disorders(
    G: nx.DiGraph,
    disorder: str,
    min_weight: float = 0.3,
) -> List[Tuple[str, float, str]]:
    """Return comorbid disorders for *disorder* with weight >= min_weight.

    Returns list of (target_disorder, weight, evidence_note).
    """
    results = []
    if disorder not in G:
        return results
    for _, tgt, data in G.out_edges(disorder, data=True):
        if data.get("relation") == "comorbid_with" and data.get("weight", 0) >= min_weight:
            results.append((tgt, data["weight"], data.get("evidence", "")))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ───────────────────────── Exclusions ─────────────────────────────────

def check_exclusions(
    G: nx.DiGraph,
    disorder: str,
    detected_disorders: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Check whether any exclusion criteria fire for *disorder*.

    Args:
        G: knowledge graph
        disorder: disorder to check exclusions for
        detected_disorders: list of other disorders the patient may meet criteria for

    Returns:
        list of (excluding_condition, exc_type) tuples that fire.
    """
    detected_set = set(detected_disorders or [])
    fired = []
    if disorder not in G:
        return fired
    for _, tgt, data in G.out_edges(disorder, data=True):
        if data.get("relation") != "excludes":
            continue
        exc_type = data.get("exc_type", "excludes_if")
        # Fire if the excluding condition is a disorder we actually detected
        if tgt in detected_set:
            fired.append((tgt, exc_type))
    return fired


# ───────────────────────── Differentials ──────────────────────────────

def get_differentials(
    G: nx.DiGraph,
    disorder: str,
) -> List[str]:
    """Return disorders that should be considered in differential with *disorder*."""
    if disorder not in G:
        return []
    return [
        tgt
        for _, tgt, data in G.out_edges(disorder, data=True)
        if data.get("relation") == "differential_with"
    ]


# ───────────────────────── Disorder → Symptom Lookup ──────────────────

def get_disorder_symptoms(G: nx.DiGraph, disorder: str) -> List[str]:
    """Return the list of canonical symptom strings for a disorder."""
    if disorder not in G:
        return []
    return [
        tgt
        for _, tgt, data in G.out_edges(disorder, data=True)
        if data.get("relation") == "has_symptom"
    ]


def get_criteria(disorder: str) -> Optional[Dict]:
    """Return the DISORDER_CRITERIA entry for a disorder, or None."""
    return DISORDER_CRITERIA.get(disorder)


# ───────────────────────── Formatting ─────────────────────────────────

def format_kg_evidence(
    matched_symptoms: Dict[str, List[Tuple[str, float]]],
    comorbidities: List[Tuple[str, float, str]],
    exclusions: List[Tuple[str, str]],
    differentials: List[str],
    disorder: Optional[str] = None,
) -> str:
    """Format KG evidence into a human-readable string for the LLM prompt.

    This is injected into classification and diagnosis prompt placeholders.
    """
    lines: List[str] = []

    if disorder:
        lines.append(f"=== Knowledge-Graph Evidence for {disorder} ===")

    # Symptom matches
    if matched_symptoms:
        lines.append("\n-- Matched Symptoms (patient → KG symptom, similarity) --")
        for psym, matches in matched_symptoms.items():
            top = matches[:3]
            match_str = "; ".join(f"{m[0]} ({m[1]:.2f})" for m in top)
            lines.append(f"  • {psym}  →  {match_str}")

    # Comorbidities
    if comorbidities:
        lines.append("\n-- Comorbid Disorders (from KG edges) --")
        for tgt, w, note in comorbidities:
            lines.append(f"  • {tgt}  (weight={w:.2f})  – {note}")

    # Exclusions
    if exclusions:
        lines.append("\n-- ⚠ Exclusion Alerts --")
        for cond, exc_type in exclusions:
            lines.append(f"  • {exc_type}: {cond}")

    # Differentials
    if differentials:
        lines.append("\n-- Differential Diagnoses to Consider --")
        for d in differentials:
            lines.append(f"  • {d}")

    return "\n".join(lines) if lines else "(No KG evidence available)"
