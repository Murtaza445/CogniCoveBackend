"""Vectorstore utilities for FAISS embedding and retrieval."""

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


def retrieve_overview_chunks(db, query, k=10):
    """Retrieve overview/criteria chunks for classification.
    
    Args:
        db: FAISS vectorstore
        query: Search query text
        k: Number of results to retrieve
    
    Returns:
        List of retrieved documents
    """
    try:
        # Server-side filter by section
        docs = db.similarity_search(query, k=k, filter={"section": "overview_summary_criteria"})
    except TypeError:
        # Fallback: client-side filter
        docs = db.similarity_search(query, k=k*5)
        docs = [d for d in docs if d.metadata.get("section") == "overview_summary_criteria"]
    return docs


def retrieve_category_chunks(db, query, category, k=1000):
    """Retrieve all chunks for a specific disorder category.
    
    Args:
        db: FAISS vectorstore
        query: Search query text
        category: Disorder category name
        k: Maximum number of results to retrieve
    
    Returns:
        List of retrieved documents filtered by category
    """
    try:
        # Server-side filter by category only
        docs = db.similarity_search(query, k=k, filter={"category": category})
    except TypeError:
        # Fallback: client-side filter
        all_docs = db.similarity_search(query, k=k)
        docs = [d for d in all_docs if d.metadata.get("category") == category]
    return docs


def retrieve_comorbidity_chunks(db, disorder_name, k=10):
    """Retrieve comorbidity-specific chunks for a given disorder.

    Args:
        db: FAISS vectorstore
        disorder_name: Exact disorder name as stored in metadata
        k: Number of results to retrieve

    Returns:
        List of comorbidity documents for the disorder
    """
    query = f"{disorder_name} comorbidity"
    try:
        docs = db.similarity_search(
            query, k=k,
            filter={"disorder": disorder_name, "section": "comorbidity"},
        )
    except TypeError:
        all_docs = db.similarity_search(query, k=k * 5)
        docs = [
            d for d in all_docs
            if d.metadata.get("disorder") == disorder_name
            and d.metadata.get("section") == "comorbidity"
        ]
    return docs
