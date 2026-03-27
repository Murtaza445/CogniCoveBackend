"""Pydantic models for data validation and serialization."""

from pydantic import BaseModel, Field
from typing import List


class SummaryModel(BaseModel):
    """Clinical summary extracted from therapy conversation."""
    overview: str = Field("", description="Brief overview of patient's presentation")
    symptom_list: List[str] = Field(default_factory=list, description="List of reported symptoms")
    duration_requirements: str = Field("", description="Reported duration of symptoms")
    impairment_requirement: str = Field("", description="Functional impairment description")
    differential_diagnosis: str = Field("", description="Possible differential diagnoses")
    associated_suicide_risk: str = Field("", description="Suicide risk summary")
    conversation_summary: str = Field("", description="Free-text conversation summary")


class ClassificationModel(BaseModel):
    """Disorder category classification result with comorbidity detection."""
    predicted_category: str = Field("", description="The most likely disorder category")
    confidence: float = Field(default=0.0, description="Confidence level 0.0-1.0 for primary diagnosis")
    reasoning: str = Field("", description="Brief reasoning for classification")
    matched_symptoms: List[str] = Field(default_factory=list, description="Symptoms that matched the category criteria")
    comorbid_categories: List[str] = Field(default_factory=list, description="List of comorbid disorder categories")
    comorbidity_scores: dict = Field(default_factory=dict, description="Confidence scores for each detected category (0.0-1.0)")


class KGEvidence(BaseModel):
    """Knowledge-graph evidence passed to classification / diagnosis prompts."""
    matched_symptoms: dict = Field(default_factory=dict, description="Patient symptom → [(KG symptom, score)] mapping")
    comorbid_disorders: list = Field(default_factory=list, description="(disorder, weight, note) tuples from KG")
    exclusion_alerts: list = Field(default_factory=list, description="(condition, exc_type) tuples that fired")
    differentials: list = Field(default_factory=list, description="Disorder names to consider in differential")
    formatted_text: str = Field("", description="Pre-formatted string for LLM prompt injection")


class DiagnosticResults(BaseModel):
    """Structured diagnostic results matching clinical assessment schema."""
    category: str = Field("", description="Disorder category (e.g., Depressive Disorders)")
    disorder_name: str = Field("", description="Specific disorder name within category")
    matched_criteria: int = Field(0, description="Number of criteria met")
    total_required: int = Field(0, description="Total criteria required for diagnosis")
    duration_met: bool = Field(False, description="Whether duration requirements are met")
    impairment_met: bool = Field(False, description="Whether functional impairment criteria are met")
    exclusion_triggered: bool = Field(False, description="Whether exclusion criteria are triggered")
    alignment_score: float = Field(0.0, description="Alignment score 0.0-1.0 based on symptom match")
    confidence_level: str = Field("low", description="Confidence: low, moderate, or high")
    recommendation: str = Field("", description="Recommended next steps and clinical notes")
