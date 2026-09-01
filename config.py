from typing import Dict, List, Set

APP_TITLE = "MedPehchaan AI+ - Intelligent Clinical Text Intelligence System"
APP_TAGLINE = (
    "Biomedical entity extraction, explainable triage cues, and readable clinical summaries "
    "for education and research workflows."
)
APP_DISCLAIMER = (
    "This is an educational AI system and must not be used for medical diagnosis, treatment, "
    "or emergency decision-making."
)

INPUT_MODES = [
    "Use typed text only",
    "Use uploaded file only",
    "Use uploaded image only",
    "Combine all inputs",
]

UPLOAD_FILE_TYPES = ["txt", "pdf", "csv", "tsv", "xlsx", "jsonl"]
MAX_DATASET_PREVIEW_ROWS = 10
LARGE_DATASET_THRESHOLD = 1000
VERY_LARGE_DATASET_THRESHOLD = 10000
EXTREME_DATASET_THRESHOLD = 50000
DEFAULT_PROCESSING_CHUNK_SIZE = 256
LARGE_DATASET_PROCESSING_CHUNK_SIZE = 512
VERY_LARGE_DATASET_PROCESSING_CHUNK_SIZE = 1024
EXTREME_DATASET_PROCESSING_CHUNK_SIZE = 2048
DEFAULT_NER_BATCH_SIZE = 16
LARGE_DATASET_NER_BATCH_SIZE = 32
VERY_LARGE_DATASET_NER_BATCH_SIZE = 64
EXTREME_DATASET_NER_BATCH_SIZE = 128
MAX_RENDERED_PATIENTS = 250
# Memory optimization settings
ENABLE_MEMORY_OPTIMIZATION = True
ENABLE_STREAMING_MODE = True
STREAMING_FLUSH_INTERVAL = 100
ENABLE_MULTIPROCESSING = True
MAX_WORKERS = 4
FALLBACK_CONFIDENCE_BASE = 58.0
FALLBACK_CONFIDENCE_EXACT_MATCH_BONUS = 6.0
FALLBACK_CONFIDENCE_CANONICAL_BONUS = 4.0
FALLBACK_CONFIDENCE_MULTIWORD_BONUS = 4.0
FALLBACK_CONFIDENCE_MAX = 74.0

ENTITY_LABELS = ["Disease", "Symptom", "Medication", "Procedure"]
ENTITY_COLORS: Dict[str, str] = {
    "Disease": "#f6d2d2",
    "Symptom": "#f9e2c2",
    "Medication": "#d7efda",
    "Procedure": "#d8e7fb",
}

LABEL_PRIORITY = {
    "Medication": 4,
    "Disease": 3,
    "Symptom": 2,
    "Procedure": 1,
}

MIN_ENTITY_CONFIDENCE = 0.50
HIGH_CONFIDENCE_THRESHOLD = 0.85
MEDIUM_CONFIDENCE_THRESHOLD = 0.60
MAX_ENTITY_WORDS = 4

MODEL_CANDIDATES: List[Dict[str, str]] = [
    {"name": "d4data/biomedical-ner-all", "role": "primary_biomedical_ner"},
    {"name": "samrawal/bert-base-uncased_clinical-ner", "role": "clinical_ner_fallback"},
]

CANONICAL_ENTITY_LABELS = {
    "asthma": "Disease",
    "diabetes": "Disease",
    "fatigue": "Symptom",
    "headache": "Symptom",
    "beta blockers": "Medication",
    "beta blocker": "Medication",
    "chest pain": "Symptom",
}

SUPPORT_DICTIONARY: Dict[str, Set[str]] = {
    "Disease": {
        "anemia",
        "asthma",
        "cold",
        "diabetes",
        "heart disease",
        "hypertension",
        "infection",
        "pneumonia",
        "stroke",
    },
    "Symptom": {
        "chest pain",
        "cough",
        "fatigue",
        "fever",
        "headache",
        "nausea",
        "shortness of breath",
    },
    "Medication": {
        "amoxicillin",
        "aspirin",
        "beta blockers",
        "insulin",
        "metformin",
        "paracetamol",
    },
    "Procedure": {
        "blood test",
        "ct scan",
        "ecg",
        "mri",
        "x-ray",
    },
}

CHRONIC_DISEASE_TERMS = {
    "asthma",
    "diabetes",
    "heart disease",
    "hypertension",
}

RISK_RULES = {
    "High": ["chest pain", "heart disease", "stroke"],
    "Medium": ["fever", "infection"],
    "Low": ["headache", "cold"],
}

MODEL_LABEL_HINTS = {
    "disease": "Disease",
    "disorder": "Disease",
    "condition": "Disease",
    "problem": "Disease",
    "diagnosis": "Disease",
    "dx": "Disease",
    "sign": "Symptom",
    "symptom": "Symptom",
    "symptoms": "Symptom",
    "drug": "Medication",
    "medication": "Medication",
    "treatment": "Medication",
    "brand": "Medication",
    "procedure": "Procedure",
    "test": "Procedure",
    "exam": "Procedure",
    "investigation": "Procedure",
    "therapeutic": "Procedure",
}
