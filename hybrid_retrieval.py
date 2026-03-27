"""
Hybrid retrieval: combines FAISS vector search with Knowledge-Graph reasoning.

Two public entry-points used by ui.py:
  • hybrid_classify  – enriches the classification step with KG symptom matching
  • hybrid_diagnose  – enriches per-category diagnosis with comorbidity, exclusion,
                        and differential evidence from the KG
"""

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage
from pydantic import ValidationError

from chains import get_classification_chain, get_diagnosis_chain
from constants import (
    DISORDER_CATEGORIES,
    PRIMARY_CONFIDENCE_THRESHOLD,
    COMORBIDITY_CONFIDENCE_THRESHOLD,
    MAX_COMORBIDITIES,
    KG_COMORBIDITY_THRESHOLD,
    SYMPTOM_MATCH_THRESHOLD,
)
from knowledge_graph import (
    build_knowledge_graph,
    match_symptoms,
    get_comorbid_disorders,
    check_exclusions,
    get_differentials,
    format_kg_evidence,
    get_criteria,
)
from kg_data import CATEGORY_MAP
from models import ClassificationModel, DiagnosticResults, KGEvidence
from vectorstore import get_vectorstore, retrieve_overview_chunks, retrieve_category_chunks, retrieve_comorbidity_chunks


# ───────────────────────── helpers ────────────────────────────────────

def _parse_symptoms_from_summary(summary_json: str) -> list:
    """Extract a flat list of symptom strings from the clinical summary JSON."""
    try:
        data = json.loads(summary_json)
    except Exception:
        return []
    symptoms = data.get("symptom_list", [])
    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in symptoms.split(",") if s.strip()]
    return symptoms


def _safe_json_parse(text: str, model_cls):
    """Try to parse *text* as *model_cls*; fall back to regex extraction."""
    text = text.strip()
    try:
        return model_cls.model_validate_json(text)
    except (ValidationError, Exception):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return model_cls.model_validate_json(m.group(0))
            except Exception:
                return None
    return None


# ───────────────────── hybrid_classify ────────────────────────────────

def hybrid_classify(summary_json: str) -> Optional[ClassificationModel]:
    """Run classification with FAISS retrieval + KG symptom evidence.

    Steps:
      1. FAISS overview retrieval (same as before).
      2. KG symptom matching on patient symptom list.
      3. Inject KG evidence into the classifier prompt.
      4. Parse and return ClassificationModel.
    """
    db = get_vectorstore()
    G = build_knowledge_graph()

    # 1 — FAISS overview chunks
    ref_docs = retrieve_overview_chunks(db, summary_json, k=10)
    ref_text = "\n\n---\n\n".join(
        f"**Category: {d.metadata.get('category', 'N/A')}**\n{d.page_content}"
        for d in ref_docs
    )

    # 2 — KG symptom matching
    patient_symptoms = _parse_symptoms_from_summary(summary_json)
    sym_matches = match_symptoms(G, patient_symptoms, threshold=SYMPTOM_MATCH_THRESHOLD)
    kg_block = format_kg_evidence(
        matched_symptoms=sym_matches,
        comorbidities=[],
        exclusions=[],
        differentials=[],
    )

    # 3 — Call classification chain (KG evidence appended to reference_chunks)
    enriched_refs = ref_text + "\n\n## Knowledge-Graph Symptom Evidence\n" + kg_block
    classification_chain = get_classification_chain()
    try:
        cls_resp = classification_chain.invoke({
            "categories": ", ".join(DISORDER_CATEGORIES + ["None"]),
            "summary_json": summary_json,
            "reference_chunks": enriched_refs,
        })
        cls_text = cls_resp.content.strip()
    except Exception:
        cls_text = ""

    # 4 — Parse and validate
    cls_result = _safe_json_parse(cls_text, ClassificationModel)
    
    if cls_result:
        allowed_primary = set(DISORDER_CATEGORIES) | {"None"}

        # Validate primary category is from allowed list (or "None")
        if cls_result.predicted_category not in allowed_primary:
            return None  # Reject invalid primary prediction

        # Enforce primary confidence threshold; allow a valid "None" outcome
        if cls_result.predicted_category != "None" and cls_result.confidence <= PRIMARY_CONFIDENCE_THRESHOLD:
            cls_result.predicted_category = "None"
            cls_result.comorbid_categories = []
            cls_result.comorbidity_scores = {}
            return cls_result

        if cls_result.predicted_category == "None":
            cls_result.comorbid_categories = []
            cls_result.comorbidity_scores = {}
            return cls_result

        # Filter comorbidities: allowed list AND confidence > threshold AND not the primary
        valid_comorbidities_with_scores = []
        for cat in cls_result.comorbid_categories:
            if cat not in DISORDER_CATEGORIES:
                continue
            if cat == cls_result.predicted_category:
                continue
            score = float(cls_result.comorbidity_scores.get(cat, 0.0) or 0.0)
            if score > COMORBIDITY_CONFIDENCE_THRESHOLD:
                valid_comorbidities_with_scores.append((cat, score))

        # Sort by confidence and keep only top N
        valid_comorbidities_with_scores.sort(key=lambda x: x[1], reverse=True)
        valid_comorbidities = [cat for cat, _ in valid_comorbidities_with_scores[:MAX_COMORBIDITIES]]

        cls_result.comorbid_categories = valid_comorbidities
        cls_result.comorbidity_scores = {cat: cls_result.comorbidity_scores.get(cat, 0.0) for cat in valid_comorbidities}
    
    return cls_result


# ───────────────────── hybrid_diagnose ────────────────────────────────

def hybrid_diagnose(
    summary_json: str,
    classification: ClassificationModel,
) -> list:
    """Run per-category diagnosis enriched with KG comorbidity / exclusion data.

    Returns a list of serialized DiagnosticResults JSON strings (same format
    the UI already consumes).
    """
    db = get_vectorstore()
    G = build_knowledge_graph()

    patient_symptoms = _parse_symptoms_from_summary(summary_json)

    primary = classification.predicted_category
    if primary not in DISORDER_CATEGORIES:
        no_dx = DiagnosticResults(
            category="None",
            disorder_name="No diagnosis",
            matched_criteria=0,
            total_required=0,
            duration_met=False,
            impairment_met=False,
            exclusion_triggered=False,
            alignment_score=0.0,
            confidence_level="low",
            recommendation="No disorder category met the required confidence threshold for diagnosis.",
        )
        return [no_dx.model_dump_json(indent=2)]

    # Diagnose only for detected categories (primary + comorbid)
    sorted_cats = [primary]
    for cat in (classification.comorbid_categories or []):
        if cat in DISORDER_CATEGORIES and cat != primary and cat not in sorted_cats:
            sorted_cats.append(cat)

    results = []

    for category in sorted_cats:
        # FAISS retrieval for this category
        cat_docs = retrieve_category_chunks(db, summary_json, category, k=1000)
        cat_text = "\n\n---\n\n".join(d.page_content for d in cat_docs)
        if not cat_text.strip():
            continue

        # --- KG enrichment per category ---
        # Identify disorders in this category
        disorders_in_cat = [d for d, c in CATEGORY_MAP.items() if c == category]

        # Aggregate KG evidence across disorders in the category
        all_comorbidities = []
        all_exclusions = []
        all_differentials = []
        detected_disorder_names = []  # collect names for exclusion checks

        # First pass: symptom matching → figure out which disorders match
        for disorder in disorders_in_cat:
            criteria = get_criteria(disorder)
            if not criteria:
                continue
            # Check if patient symptoms match this disorder's criteria
            sym_matches = match_symptoms(G, patient_symptoms, threshold=SYMPTOM_MATCH_THRESHOLD)
            # Count how many KG symptoms of this disorder are matched
            disorder_syms = set(s.lower().strip() for s in criteria.get("symptoms", []))
            matched_kg_syms = set()
            for _psym, kg_hits in sym_matches.items():
                for kg_sym, _score in kg_hits:
                    if kg_sym in disorder_syms:
                        matched_kg_syms.add(kg_sym)
            if len(matched_kg_syms) >= max(1, criteria.get("threshold", 1) // 2):
                detected_disorder_names.append(disorder)

        # Second pass: gather comorbidities, exclusions, differentials
        for disorder in disorders_in_cat:
            comorbids = get_comorbid_disorders(G, disorder, min_weight=KG_COMORBIDITY_THRESHOLD)
            all_comorbidities.extend(comorbids)

            exclusions = check_exclusions(G, disorder, detected_disorders=detected_disorder_names)
            all_exclusions.extend(exclusions)

            diffs = get_differentials(G, disorder)
            all_differentials.extend(diffs)

        # Deduplicate
        seen_comorbid = set()
        unique_comorbidities = []
        for item in all_comorbidities:
            if item[0] not in seen_comorbid:
                seen_comorbid.add(item[0])
                unique_comorbidities.append(item)

        unique_differentials = list(dict.fromkeys(all_differentials))
        unique_exclusions = list({(c, t) for c, t in all_exclusions})

        # Format KG block
        kg_block = format_kg_evidence(
            matched_symptoms=match_symptoms(G, patient_symptoms, threshold=SYMPTOM_MATCH_THRESHOLD),
            comorbidities=unique_comorbidities,
            exclusions=unique_exclusions,
            differentials=unique_differentials,
            disorder=category,
        )

        # Retrieve comorbidity chunks from FAISS for detected disorders
        comorbidity_text_parts = []
        for disorder in detected_disorder_names:
            comorb_docs = retrieve_comorbidity_chunks(db, disorder, k=5)
            for doc in comorb_docs:
                comorbidity_text_parts.append(doc.page_content)

        # Build enriched reference text
        enriched_cat_text = cat_text
        if comorbidity_text_parts:
            enriched_cat_text += "\n\n## Comorbidity References\n" + "\n\n---\n\n".join(comorbidity_text_parts)
        enriched_cat_text += "\n\n## Knowledge-Graph Evidence\n" + kg_block

        # Call diagnosis chain
        diag_chain = get_diagnosis_chain()
        try:
            diag_resp = diag_chain.invoke({
                "summary_json": summary_json,
                "references": enriched_cat_text,
                "category": category,
            })
            diag_text = diag_resp.content.strip()
        except Exception:
            diag_text = ""

        # Parse
        diag_valid = _safe_json_parse(diag_text, DiagnosticResults)
        if diag_valid:
            results.append(diag_valid.model_dump_json(indent=2))
        elif diag_text:
            results.append(diag_text)

    return results
