"""Vectorstore utilities for FAISS loading and metadata-only retrieval."""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from constants import DB_FAISS_PATH, EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_vectorstore():
    """Load and cache the FAISS vectorstore with HuggingFace embeddings."""
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    db = FAISS.load_local(
        folder_path=DB_FAISS_PATH,
        embeddings=embedding,
        index_name="index",
        allow_dangerous_deserialization=True
    )
    return db


def _iter_documents_in_deterministic_order(db):
    """Yield stored documents in a stable order based on FAISS index order."""
    docstore = getattr(db, "docstore", None)
    docstore_dict = getattr(docstore, "_dict", {}) if docstore is not None else {}
    index_to_docstore_id = getattr(db, "index_to_docstore_id", None)

    if isinstance(index_to_docstore_id, dict) and index_to_docstore_id:
        for index in sorted(index_to_docstore_id):
            doc_id = index_to_docstore_id[index]
            doc = docstore_dict.get(doc_id)
            if doc is not None:
                yield doc
        return

    for doc_id in sorted(docstore_dict):
        doc = docstore_dict.get(doc_id)
        if doc is not None:
            yield doc


def _filter_documents_by_metadata(db, *, required_metadata, k=None):
    """Select documents whose metadata matches all required labels."""
    matches = []
    for doc in _iter_documents_in_deterministic_order(db):
        metadata = getattr(doc, "metadata", {}) or {}
        if all(metadata.get(key) == value for key, value in required_metadata.items()):
            matches.append(doc)
            if k is not None and len(matches) >= k:
                break
    return matches


def retrieve_overview_chunks(db, query, k=10):
    """Retrieve overview/criteria chunks by metadata only.
    
    Args:
        db: FAISS vectorstore
        query: Preserved for compatibility; not used for retrieval
        k: Maximum number of matching results to return after filtering
    
    Returns:
        List of overview documents whose metadata.section matches the overview label
    """
    return _filter_documents_by_metadata(
        db,
        required_metadata={"section": "overview_summary_criteria"},
        k=k,
    )


def retrieve_category_chunks(db, query, category, k=1000):
    """Retrieve all chunks for a specific disorder category by metadata only.
    
    Args:
        db: FAISS vectorstore
        query: Preserved for compatibility; not used for retrieval
        category: Disorder category name
        k: Maximum number of matching results to return after filtering
    
    Returns:
        List of documents whose metadata.category matches the requested category
    """
    return _filter_documents_by_metadata(
        db,
        required_metadata={"category": category},
        k=k,
    )


def retrieve_comorbidity_chunks(db, disorder_name, k=10):
    """Retrieve comorbidity-specific chunks for a given disorder by metadata only.

    Args:
        db: FAISS vectorstore
        disorder_name: Exact disorder name as stored in metadata
        k: Maximum number of matching results to return after filtering

    Returns:
        List of comorbidity documents whose metadata.disorder and metadata.section match
    """
    return _filter_documents_by_metadata(
        db,
        required_metadata={"disorder": disorder_name, "section": "comorbidity"},
        k=k,
    )
