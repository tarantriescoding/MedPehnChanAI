import html
import io
import traceback
from typing import Dict, List, Tuple

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from image_utils import detect_edges, preprocess_image, read_image, extract_text_from_image

from config import (
    APP_DISCLAIMER,
    APP_TAGLINE,
    APP_TITLE,
    DEFAULT_NER_BATCH_SIZE,
    DEFAULT_PROCESSING_CHUNK_SIZE,
    INPUT_MODES,
    LARGE_DATASET_NER_BATCH_SIZE,
    LARGE_DATASET_PROCESSING_CHUNK_SIZE,
    LARGE_DATASET_THRESHOLD,
    VERY_LARGE_DATASET_NER_BATCH_SIZE,
    VERY_LARGE_DATASET_PROCESSING_CHUNK_SIZE,
    VERY_LARGE_DATASET_THRESHOLD,
    EXTREME_DATASET_NER_BATCH_SIZE,
    EXTREME_DATASET_PROCESSING_CHUNK_SIZE,
    EXTREME_DATASET_THRESHOLD,
    MAX_DATASET_PREVIEW_ROWS,
    MAX_RENDERED_PATIENTS,
    UPLOAD_FILE_TYPES,
    ENABLE_STREAMING_MODE,
)
from evaluation import LABEL_FIELD_ALIASES, compute_metrics, compute_metrics_for_patient_results
from intelligence import process_dataset
from pdf_utils import extract_text_from_uploaded_file
from preprocessing import normalize_patient_id, split_text_into_patient_records
from report_utils import generate_patient_csv, generate_patient_pdf
from utils import build_entity_table, resolve_input_text


# ===== HELPER FUNCTIONS (Defined early for Streamlit caching) =====

def _normalize_search_keywords(search_term: str) -> List[str]:
    """
    Normalize search term into a list of keywords for searching.
    
    Args:
        search_term: Raw search term from user input
        
    Returns:
        List of lowercase keywords
    """
    if not search_term or not isinstance(search_term, str):
        return []
    
    # Split by spaces and convert to lowercase, filter empty strings
    keywords = [keyword.strip().lower() for keyword in search_term.split() if keyword.strip()]
    return keywords


TEXT_COLUMN_CANDIDATES = [
    "text",
    "note",
    "notes",
    "clinical_note",
    "clinical_notes",
    "report",
    "description",
    "discharge_summary",
    "summary",
]
PATIENT_ID_CANDIDATES = ["patient_id", "subject_id", "hadm_id", "mrn", "record_id", "id"]


@st.cache_data(show_spinner=False)
def load_evaluation_metrics() -> Dict[str, object]:
    return compute_metrics(run_inference=True)


def _read_uploaded_table(uploaded_file) -> pd.DataFrame:
    file_bytes = uploaded_file.getvalue()
    file_name = str(getattr(uploaded_file, "name", "") or "").lower()

    if file_name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(file_bytes))

    if file_name.endswith(".jsonl"):
        return pd.read_json(io.BytesIO(file_bytes), lines=True)

    csv_kwargs = {"low_memory": True}
    if file_name.endswith(".tsv"):
        csv_kwargs["sep"] = "\t"

    try:
        return pd.read_csv(io.BytesIO(file_bytes), **csv_kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", **csv_kwargs)


def _build_row_metadata(row_dict: Dict[str, object], row_number: int, id_column: str | None) -> Dict[str, object]:
    metadata: Dict[str, object] = {"row_number": row_number}
    if id_column:
        metadata[id_column] = row_dict.get(id_column)

    label_columns = {alias for aliases in LABEL_FIELD_ALIASES.values() for alias in aliases}
    for column in label_columns:
        if column in row_dict:
            metadata[column] = row_dict.get(column)
    return metadata


def _candidate_text_columns(df: pd.DataFrame) -> List[str]:
    available_columns = {column.lower(): column for column in df.columns}
    matches = [available_columns[name] for name in TEXT_COLUMN_CANDIDATES if name in available_columns]
    if matches:
        return matches

    excluded_columns = set(PATIENT_ID_CANDIDATES)
    for aliases in LABEL_FIELD_ALIASES.values():
        excluded_columns.update(alias.lower() for alias in aliases)

    object_columns = [
        column
        for column in df.columns
        if df[column].dtype == "object" and column.lower() not in excluded_columns
    ]
    return object_columns[:3]


def dataframe_to_patient_records(df: pd.DataFrame, source_name: str = "uploaded_csv") -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if df.empty:
        return [], {"text_columns": [], "patient_id_column": None}

    text_columns = _candidate_text_columns(df)
    id_column = next((column for column in df.columns if column.lower() in PATIENT_ID_CANDIDATES), None)
    records: List[Dict[str, object]] = []

    for index, row in df.iterrows():
        row_dict = row.to_dict()
        patient_id = normalize_patient_id(row_dict.get(id_column) if id_column else None, index + 1)
        text_parts = []
        for column in text_columns:
            value = row_dict.get(column)
            if pd.isna(value):
                continue
            text = str(value).strip()
            if text:
                text_parts.append(text)

        raw_text = "\n".join(text_parts).strip()
        if not raw_text:
            continue

        records.append(
            {
                "patient_id": patient_id,
                "record_index": len(records) + 1,
                "source": source_name,
                "raw_text": raw_text,
                "metadata": _build_row_metadata(row_dict, index + 1, id_column),
            }
        )

    return records, {"text_columns": text_columns, "patient_id_column": id_column}


def _resolve_processing_settings(record_count: int) -> Tuple[int, int, bool]:
    """
    Resolve optimal processing settings based on dataset size.
    
    Returns: (batch_size, chunk_size, use_streaming_mode)
    """
    if record_count >= EXTREME_DATASET_THRESHOLD:
        return EXTREME_DATASET_NER_BATCH_SIZE, EXTREME_DATASET_PROCESSING_CHUNK_SIZE, ENABLE_STREAMING_MODE and record_count >= 50000
    if record_count >= VERY_LARGE_DATASET_THRESHOLD:
        return VERY_LARGE_DATASET_NER_BATCH_SIZE, VERY_LARGE_DATASET_PROCESSING_CHUNK_SIZE, ENABLE_STREAMING_MODE and record_count >= 20000
    if record_count >= LARGE_DATASET_THRESHOLD:
        return LARGE_DATASET_NER_BATCH_SIZE, LARGE_DATASET_PROCESSING_CHUNK_SIZE, False
    return DEFAULT_NER_BATCH_SIZE, DEFAULT_PROCESSING_CHUNK_SIZE, False


def build_selected_records(
    typed_text: str,
    uploaded_file,
    uploaded_preview_text: str,
    uploaded_dataframe: pd.DataFrame | None,
    uploaded_image_text: str,
    input_mode: str,
) -> Tuple[List[Dict[str, object]], str, Dict[str, object]]:
    typed = (typed_text or "").strip()
    typed_records = split_text_into_patient_records(typed, source_name="typed_text") if typed else []
    uploaded_records: List[Dict[str, object]] = []
    image_records: List[Dict[str, object]] = []
    upload_meta: Dict[str, object] = {"text_columns": [], "patient_id_column": None}

    if uploaded_file is not None:
        if (uploaded_file.name or "").lower().endswith((".csv", ".tsv", ".xlsx", ".jsonl")):
            uploaded_records, upload_meta = dataframe_to_patient_records(
                uploaded_dataframe if uploaded_dataframe is not None else pd.DataFrame()
            )
        else:
            uploaded_records = (
                split_text_into_patient_records(uploaded_preview_text, source_name="uploaded_file")
                if uploaded_preview_text.strip()
                else []
            )

    # Process extracted image text
    if (uploaded_image_text or "").strip():
        image_records = split_text_into_patient_records(uploaded_image_text, source_name="uploaded_image")

    selected_text, input_source = resolve_input_text(typed, uploaded_preview_text, uploaded_image_text, input_mode)
    if input_mode == "Use typed text only":
        return typed_records, input_source, upload_meta
    if input_mode == "Use uploaded file only":
        return uploaded_records, input_source, upload_meta
    if input_mode == "Use uploaded image only":
        return image_records, input_source, upload_meta
    if not selected_text.strip():
        return [], "combined_inputs", upload_meta
    return typed_records + uploaded_records + image_records, "combined_inputs", upload_meta


def _render_model_status(model_meta: Dict[str, object]) -> None:
    st.subheader("Model Status")
    if model_meta.get("available"):
        st.success(
            f"Model loaded: {model_meta['model_name']} ({model_meta['role']}) on {model_meta.get('device', 'cpu')}.",
            icon="✅",
        )
    else:
        st.warning(
            "No biomedical model checkpoint could be loaded in this environment. Dictionary fallback is active.",
            icon="⚠️",
        )
    if model_meta.get("auth_enabled"):
        st.caption("Hugging Face authentication detected. Higher Hub rate limits are enabled.")
    else:
        st.caption("Set the `HF_TOKEN` environment variable to avoid unauthenticated Hugging Face Hub limits.")
    if model_meta.get("failures"):
        with st.expander("Model loading details", expanded=False):
            st.json(model_meta["failures"])


def _render_patient_section(index: int, patient_result: Dict[str, object]) -> None:
    risk_level = patient_result["risk"]["risk_level"]

    # Modern risk color coding
    risk_colors = {
        "High": ("#e53e3e", "#fed7d7", "🚨"),
        "Medium": ("#dd6b20", "#feebc8", "⚠️"),
        "Low": ("#38a169", "#c6f6d5", "✅")
    }
    risk_color, risk_bg, risk_icon = risk_colors.get(risk_level, ("#718096", "#f7fafc", "ℹ️"))

    title = f"{risk_icon} Patient {index}: {patient_result['patient_id']} | Risk: {risk_level}"

    with st.expander(title, expanded=index == 1):
        # Modern metrics cards
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin: 20px 0;">
            <div style="background: linear-gradient(135deg, {risk_bg} 0%, rgba(255,255,255,0.9) 100%); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-left: 4px solid {risk_color};">
                <div style="font-size: 2rem; margin-bottom: 8px;">🏥</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {risk_color};">{len(patient_result["entities"])}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Entities</div>
            </div>
            <div style="background: rgba(255,255,255,0.95); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                <div style="font-size: 2rem; margin-bottom: 8px;">📝</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2d3748;">{patient_result["preprocessing"]["token_count"]}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Tokens</div>
            </div>
            <div style="background: rgba(255,255,255,0.95); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                <div style="font-size: 2rem; margin-bottom: 8px;">📏</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2d3748;">{patient_result["preprocessing"]["char_count"]}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Characters</div>
            </div>
            <div style="background: rgba(255,255,255,0.95); border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #2d3748;">{str(patient_result["source"]).replace("_", " ").title()}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Source</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander(title, expanded=index == 1):
        # Metrics cards
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin: 20px 0;">
            <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-left: 4px solid {risk_color};">
                <div style="font-size: 2rem; margin-bottom: 8px;">🏥</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: {risk_color};">{len(patient_result["entities"])}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Entities</div>
            </div>
            <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="font-size: 2rem; margin-bottom: 8px;">📝</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1E3A8A;">{patient_result["preprocessing"]["token_count"]}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Tokens</div>
            </div>
            <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="font-size: 2rem; margin-bottom: 8px;">📏</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1E3A8A;">{patient_result["preprocessing"]["char_count"]}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Characters</div>
            </div>
            <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #1E3A8A;">{str(patient_result["source"]).replace("_", " ").title()}</div>
                <div style="color: #4a5568; font-size: 0.9rem;">Source</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Download buttons
        st.markdown('<div style="display: flex; gap: 16px; margin: 24px 0;">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 Download PDF Report",
                data=generate_patient_pdf(patient_result),
                file_name=f"patient_{patient_result['patient_id']}_report.pdf",
                mime="application/pdf",
                key=f"pdf_report_{patient_result['patient_id']}_{index}",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                label="📊 Download CSV Report",
                data=generate_patient_csv(patient_result),
                file_name=f"patient_{patient_result['patient_id']}_report.csv",
                mime="text/csv",
                key=f"csv_report_{patient_result['patient_id']}_{index}",
                use_container_width=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # Extracted Medical Entities
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Extracted Medical Entities")
        if not patient_result["entities"]:
            st.warning("⚠️ No medical entities detected in this patient record.")
        else:
            st.markdown(_render_entity_badges(patient_result["entities"]), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Risk Assessment
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Risk Assessment")
        risk_explanation = patient_result["risk"]["explanation"]
        risk_class = f'risk-{risk_level.lower()}'
        st.markdown(f'<div class="{risk_class}" style="padding: 16px; border-radius: 8px; margin: 10px 0;">{risk_explanation}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Clinical Insights
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 💡 Clinical Insights")
        for insight in patient_result["insights"]:
            st.markdown(f'<div class="insight-item">💡 {insight}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Patient Summary
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📋 Patient Summary")
        st.markdown(f'<div class="summary-box">{patient_result["summary"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 🎨 Highlighted Clinical Notes")
        st.markdown(patient_result["highlighted_html"], unsafe_allow_html=True)

        with st.expander("🔧 Structured JSON Data", expanded=False):
            st.json(
                {
                    "patient_id": patient_result["patient_id"],
                    "risk": patient_result["risk"],
                    "insights": patient_result["insights"],
                    "summary": patient_result["summary"],
                    "structured_data": patient_result["structured_data"],
                }
            )


def _render_entity_badges(entities: List[Dict[str, object]]) -> str:
    if not entities:
        return "<p>No entities detected.</p>"
    
    badges = []
    for entity in entities:
        entity_type = entity.get("label", "").lower()
        text = entity.get("text", "")
        confidence = entity.get("confidence", 0)
        css_class = {
            "disease": "disease",
            "symptom": "symptom", 
            "medication": "medication",
            "procedure": "procedure"
        }.get(entity_type, "disease")
        badges.append(f'<span class="entity-badge {css_class}" title="Confidence: {confidence:.2f}">{text}</span>')
    
    return "".join(badges)


def _patient_search_blob(patient_result: Dict[str, object]) -> str:
    entity_names = [str(entity.get("text", "")).strip() for entity in patient_result.get("entities", [])]
    parts = [
        str(patient_result.get("patient_id", "")),
        str(patient_result.get("raw_text", "")),
        str(patient_result.get("summary", "")),
        " ".join(entity_names),
    ]
    return " ".join(part for part in parts if part).lower()


def _matched_keywords(patient_result: Dict[str, object], keywords: List[str]) -> List[str]:
    if not keywords:
        return []

    searchable_text = _patient_search_blob(patient_result)
    return [keyword for keyword in keywords if keyword in searchable_text]


def _highlight_keyword_matches(keywords: List[str]) -> str:
    if not keywords:
        return ""
    badges = "".join(
        f"<span style='background:#eef5ff;border:1px solid #c9daf8;border-radius:999px;padding:2px 8px;margin-right:6px;'>{html.escape(keyword)}</span>"
        for keyword in keywords
    )
    return f"<div style='margin:6px 0 10px 0;'>Matched keywords: {badges}</div>"


def _ensure_session_defaults() -> None:
    defaults = {
        "analysis_ready": False,
        "analysis_results": None,
        "analysis_input_source": "",
        "analysis_upload_meta": {"text_columns": [], "patient_id_column": None},
        "analysis_uploaded_eval_metrics": None,
        "analysis_signature": None,
        "patient_search_term": "",
        "patient_risk_filter": "All",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _build_input_signature(
    typed_text: str,
    uploaded_file,
    uploaded_text: str,
    uploaded_dataframe: pd.DataFrame | None,
    uploaded_image_text: str,
    input_mode: str,
) -> Tuple[object, ...]:
    file_name = ""
    file_size = 0
    if uploaded_file is not None:
        file_name = str(getattr(uploaded_file, "name", "") or "")
        file_size = int(getattr(uploaded_file, "size", 0) or 0)

    dataframe_shape = tuple(uploaded_dataframe.shape) if uploaded_dataframe is not None else None
    return (
        (typed_text or "").strip(),
        file_name,
        file_size,
        (uploaded_text or "").strip(),
        dataframe_shape,
        (uploaded_image_text or "").strip(),
        input_mode,
    )


def run_app() -> None:
    st.set_page_config(
        page_title="MedPehchaan AI+",
        layout="wide"
    )

    # Custom CSS for exceptionally beautiful and modern look
    st.markdown("""
    <style>
    /* Import Google Fonts for modern typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Light mode - default */
    /* Main background with clean light */
    .main {
        background: #F8FAFC;
    }

    .stAppViewContainer {
        background: #F8FAFC;
    }

    /* Title styling - light mode */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }

    h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    /* Modern button styling */
    .stButton>button {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.4);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 58, 138, 0.6);
    }

    /* Input fields styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea,
    .stSelectbox>div>div>div>div>select {
        border-radius: 12px;
        border: 2px solid #E5E7EB;
        padding: 16px 20px;
        background: white;
        color: #1a202c;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus,
    .stSelectbox>div>div>div>div>select:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }

    /* File uploader styling */
    .stFileUploader {
        border-radius: 12px;
        border: 2px dashed #CBD5E1;
        background: white;
        padding: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .stFileUploader:hover {
        border-color: #3B82F6;
        background: #F8FAFC;
    }

    /* Card styling */
    .card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin: 20px 0;
        border: 1px solid #E5E7EB;
    }

    /* Expander styling */
    .stExpander {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E5E7EB;
        margin: 10px 0;
        overflow: hidden;
    }

    .stExpander > div:first-child {
        background: #1E3A8A;
        color: white;
        font-weight: 600;
        padding: 16px 20px;
        border-radius: 12px 12px 0 0;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        border-radius: 10px;
    }

    /* Alert styling */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 12px 12px 0 0;
        padding: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        margin: 0 4px;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #1E3A8A;
        color: white;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3);
    }

    /* Success/Warning/Error styling */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 12px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #F1F5F9;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: #1E3A8A;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #3B82F6;
    }

    /* Download button styling */
    .stDownloadButton>button {
        background: #22C55E;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4);
    }

    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(34, 197, 94, 0.6);
    }

    /* Entity badges */
    .entity-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 2px 4px;
        color: white;
    }

    .disease { background-color: #DC2626; }
    .symptom { background-color: #EA580C; }
    .medication { background-color: #16A34A; }
    .procedure { background-color: #7C3AED; }

    /* Risk boxes */
    .risk-high { background-color: #FEE2E2; border-left: 4px solid #DC2626; color: #991B1B; }
    .risk-medium { background-color: #FEF3C7; border-left: 4px solid #D97706; color: #92400E; }
    .risk-low { background-color: #D1FAE5; border-left: 4px solid #059669; color: #065F46; }

    /* Insights styling */
    .insight-item {
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid #3B82F6;
    }

    /* Summary styling */
    .summary-box {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E5E7EB;
        font-size: 1.1rem;
        line-height: 1.6;
    }

    /* ===== DARK MODE SUPPORT ===== */
    @media (prefers-color-scheme: dark) {
        .main, .stAppViewContainer {
            background: #0F172A;
        }

        h1 {
            color: #60A5FA;
        }

        h2, h3 {
            color: #93C5FD;
        }

        p, span, div, body {
            color: #E2E8F0;
        }

        /* Input fields - dark mode */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>div>div>select {
            background: #1E293B;
            color: #E2E8F0;
            border-color: #334155;
        }

        .stTextInput>div>div>input::placeholder,
        .stTextArea>div>div>textarea::placeholder {
            color: #94A3B8;
        }

        /* File uploader - dark mode */
        .stFileUploader {
            background: #1E293B;
            border-color: #334155;
        }

        .stFileUploader:hover {
            background: #0F172A;
            border-color: #3B82F6;
        }

        /* Card - dark mode */
        .card {
            background: #1E293B;
            border-color: #334155;
            color: #E2E8F0;
        }

        /* Expander - dark mode */
        .stExpander {
            background: #1E293B;
            border-color: #334155;
        }

        .stExpander > div:first-child {
            background: #3B82F6;
            color: white;
        }

        /* Tab styling - dark mode */
        .stTabs [data-baseweb="tab-list"] {
            background: #1E293B;
            border-color: #334155;
        }

        /* Scrollbar - dark mode */
        ::-webkit-scrollbar-track {
            background: #0F172A;
        }

        ::-webkit-scrollbar-thumb {
            background: #3B82F6;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #60A5FA;
        }

        /* Risk boxes - dark mode */
        .risk-high { 
            background-color: rgba(220, 38, 38, 0.2); 
            border-left-color: #EF4444; 
            color: #FCA5A5; 
        }

        .risk-medium { 
            background-color: rgba(217, 119, 6, 0.2); 
            border-left-color: #F59E0B; 
            color: #FBBF24; 
        }

        .risk-low { 
            background-color: rgba(34, 197, 94, 0.2); 
            border-left-color: #4ADE80; 
            color: #86EFAC; 
        }

        /* Insights - dark mode */
        .insight-item {
            background: rgba(59, 130, 246, 0.15);
            border-left-color: #60A5FA;
        }

        /* Summary box - dark mode */
        .summary-box {
            background: #1E293B;
            border-color: #334155;
            color: #E2E8F0;
        }

        /* Alert styling - dark mode */
        .stAlert {
            background: #1E293B;
            color: #E2E8F0;
        }

        /* Success/Warning/Error - dark mode */
        .stSuccess {
            background-color: rgba(34, 197, 94, 0.15) !important;
            color: #86EFAC !important;
        }

        .stWarning {
            background-color: rgba(249, 115, 22, 0.15) !important;
            color: #FDBA74 !important;
        }

        .stError {
            background-color: rgba(239, 68, 68, 0.15) !important;
            color: #FCA5A5 !important;
        }

        .stInfo {
            background-color: rgba(59, 130, 246, 0.15) !important;
            color: #93C5FD !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    _ensure_session_defaults()

    # Modern hero section
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; margin-bottom: 40px;">
        <h1>🏥 MedPehchaan AI+</h1>
        <p style="font-size: 1.3rem; color: #4a5568; margin-top: 10px; font-weight: 300;">
            Clinical Text Intelligence System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.9); border-radius: 20px; padding: 24px; margin: 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
        <h3 style="color: #2d3748; margin-top: 0;">🔬 How It Works</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;">
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 16px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📝</div>
                <strong>Split & Clean</strong><br>
                <small>Process patient records individually</small>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(72, 187, 120, 0.1) 0%, rgba(56, 161, 105, 0.1) 100%); border-radius: 16px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">🔍</div>
                <strong>Extract Entities</strong><br>
                <small>Biomedical NER with AI</small>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(237, 137, 54, 0.1) 0%, rgba(221, 107, 32, 0.1) 100%); border-radius: 16px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
                <strong>Analyze & Risk Assess</strong><br>
                <small>Generate insights & summaries</small>
            </div>
            <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(229, 62, 62, 0.1) 0%, rgba(197, 48, 48, 0.1) 100%); border-radius: 16px;">
                <div style="font-size: 2rem; margin-bottom: 10px;">📋</div>
                <strong>Report Generation</strong><br>
                <small>Comprehensive patient reports</small>
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h2 style="text-align: center; margin-bottom: 30px;">📥 Input Your Clinical Data</h2>', unsafe_allow_html=True)

        left, right = st.columns([2, 1])

        with left:
            st.markdown("### ✍️ Type or Paste Text")
            typed_text = st.text_area(
                "",
                key="typed_text_input",
                height=250,
                placeholder=(
                    "Patient ID: 1001\nDiabetes with chest pain. Aspirin prescribed.\n\n"
                    "Patient ID: 1002\nAsthma with fatigue. ECG advised."
                ),
                label_visibility="collapsed"
            )

        uploaded_text = ""
        uploaded_image_text = ""
        uploaded_dataframe = None
        uploaded_file = None
        with right:
            st.markdown("### 📎 Upload File")
            uploaded_file = st.file_uploader(
                "",
                key="uploaded_file_input",
                type=UPLOAD_FILE_TYPES,
                help="Supports: TXT, PDF, CSV, TSV, XLSX, JSONL",
                label_visibility="collapsed"
            )

            if uploaded_file is not None:
                file_name = (uploaded_file.name or "").lower()
                try:
                    if file_name.endswith((".csv", ".tsv", ".xlsx", ".jsonl")):
                        uploaded_dataframe = _read_uploaded_table(uploaded_file)
                        st.success(f"✅ Dataset loaded: {len(uploaded_dataframe)} rows")
                        if len(uploaded_dataframe) >= LARGE_DATASET_THRESHOLD:
                            st.info("💡 Large dataset detected. Optimized processing enabled.")
                        with st.expander("👀 Preview Dataset", expanded=False):
                            st.dataframe(
                                uploaded_dataframe.head(MAX_DATASET_PREVIEW_ROWS),
                                use_container_width=True,
                            )
                    else:
                        uploaded_text = extract_text_from_uploaded_file(uploaded_file)
                        st.success("✅ File processed successfully")
                        with st.expander("👀 Preview Text", expanded=False):
                            st.write((uploaded_text[:1200] + "...") if len(uploaded_text) > 1200 else uploaded_text)
                except Exception as exc:
                    st.error(f"❌ Error reading file: {exc}")

            st.markdown("### 🖼️ Upload Image")
            uploaded_image = st.file_uploader(
                "",
                key="uploaded_image_input",
                type=["png", "jpg", "jpeg", "bmp"],
                help="Upload prescription or medical document images. Text will be extracted automatically.",
                label_visibility="collapsed"
            )

            if uploaded_image is not None:
                try:
                    image = read_image(uploaded_image)
                    processed = preprocess_image(image)
                    edges = detect_edges(processed)
                    
                    st.success("✅ Image loaded and processed")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Original image")
                    with col2:
                        st.image(edges, use_container_width=True, caption="Edge detection", clamp=True, channels="GRAY")
                    
                    # Extract text from image using OCR
                    with st.spinner("Extracting text from image..."):
                        uploaded_image_text = extract_text_from_image(image)
                    
                    if uploaded_image_text.strip():
                        with st.expander("👁️ Extracted Text from Image", expanded=False):
                            st.write((uploaded_image_text[:1200] + "...") if len(uploaded_image_text) > 1200 else uploaded_image_text)
                        st.success("✅ Text extracted from image successfully")
                    else:
                        st.info("ℹ️ No text detected in image - it may be too blurry or contain only medical imagery")
                except Exception as exc:
                    st.error(f"❌ Image processing failed: {exc}")

        # Input mode selection with better styling
        has_typed = typed_text.strip()
        has_file = uploaded_text.strip() or uploaded_dataframe is not None
        has_image = (uploaded_image_text or "").strip()
        
        input_sources = sum([bool(has_typed), bool(has_file), bool(has_image)])
        
        if input_sources >= 2:
            st.markdown("### 🎯 Input Mode")
            input_mode = st.radio(
                "",
                INPUT_MODES,
                key="input_mode_selection",
                help="Choose how to combine your inputs",
                label_visibility="collapsed"
            )
        elif has_typed:
            input_mode = "Use typed text only"
            st.info("📝 Using typed text only")
        elif has_file:
            input_mode = "Use uploaded file only"
            st.info("📎 Using uploaded file only")
        elif has_image:
            input_mode = "Use uploaded image only"
            st.info("🖼️ Using uploaded image only")
        else:
            input_mode = "Use typed text only"

        # Modern analyze button
        st.markdown('<div style="text-align: center; margin-top: 30px;">', unsafe_allow_html=True)
        analyze = st.button("🚀 Analyze Patients", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    current_input_signature = _build_input_signature(
        typed_text=typed_text,
        uploaded_file=uploaded_file,
        uploaded_text=uploaded_text,
        uploaded_dataframe=uploaded_dataframe,
        uploaded_image_text=uploaded_image_text,
        input_mode=input_mode,
    )

    if analyze:
        try:
            patient_records, input_source, upload_meta = build_selected_records(
                typed_text=typed_text,
                uploaded_file=uploaded_file,
                uploaded_preview_text=uploaded_text,
                uploaded_dataframe=uploaded_dataframe,
                uploaded_image_text=uploaded_image_text,
                input_mode=input_mode,
            )

            if not patient_records:
                st.session_state["analysis_ready"] = False
                st.session_state["analysis_results"] = None
                st.warning("No patient records found. Please enter text or upload a valid text taken from image, TXT, PDF, or CSV file.")
                return

            batch_size, processing_chunk_size, use_streaming = _resolve_processing_settings(len(patient_records))
            with st.spinner("Processing patient records independently..."):
                progress_bar = st.progress(0.0)
                progress_text = st.empty()

                def _on_progress(chunk_index: int, total_chunks: int, processed_count: int) -> None:
                    progress_bar.progress(
                        min(chunk_index / max(total_chunks, 1), 1.0),
                        text=f"Processed {processed_count} / {len(patient_records)} patient records",
                    )
                    progress_text.caption(
                        f"Chunk {chunk_index} of {total_chunks} completed using NER batch size {batch_size}."
                    )

                # Use streaming mode for very large datasets
                if use_streaming:
                    from intelligence import process_dataset_streaming
                    patient_list = []
                    aggregate_state = {}
                    model_meta = {}
                    
                    for patient_result in process_dataset_streaming(
                        patient_records,
                        batch_size=batch_size,
                        processing_chunk_size=processing_chunk_size,
                        progress_callback=_on_progress,
                    ):
                        patient_list.append(patient_result)
                    
                    # This will contain the final aggregate report
                    results = {
                        "patients": patient_list,
                        "aggregate_report": {"total_patients_processed": len(patient_list)},
                        "model_meta": model_meta,
                    }
                else:
                    from intelligence import process_dataset
                    results = process_dataset(
                        patient_records,
                        batch_size=batch_size,
                        processing_chunk_size=processing_chunk_size,
                        progress_callback=_on_progress,
                    )
                progress_bar.empty()
                progress_text.empty()
                uploaded_eval_metrics = compute_metrics_for_patient_results(results["patients"])

            st.session_state["analysis_ready"] = True
            st.session_state["analysis_results"] = results
            st.session_state["analysis_input_source"] = input_source
            st.session_state["analysis_upload_meta"] = upload_meta
            st.session_state["analysis_uploaded_eval_metrics"] = uploaded_eval_metrics
            st.session_state["analysis_signature"] = current_input_signature
            st.session_state["patient_search_term"] = ""
            st.session_state["patient_risk_filter"] = "All"
        except Exception as exc:
            st.session_state["analysis_ready"] = False
            st.session_state["analysis_results"] = None
            st.error("The app hit an unexpected error while processing this input.")
            st.code(f"{type(exc).__name__}: {exc}")
            st.text(traceback.format_exc())
            return

    if not st.session_state["analysis_ready"]:
        return

    if st.session_state["analysis_signature"] != current_input_signature:
        st.info("Inputs changed after the last analysis. Click `Analyze Patients` again to refresh the results.")
        return

    try:
        results = st.session_state["analysis_results"]
        input_source = st.session_state["analysis_input_source"]
        upload_meta = st.session_state["analysis_upload_meta"]
        uploaded_eval_metrics = st.session_state["analysis_uploaded_eval_metrics"]

        st.markdown('<h2 style="text-align: center; margin-bottom: 30px;">📊 Processing Overview</h2>', unsafe_allow_html=True)
        overview_cols = st.columns(4)
        metrics_data = [
            (results["aggregate_report"]["total_patients_processed"], "👥 Patients", "#1E3A8A"),
            (results["aggregate_report"]["total_diseases_detected"], "🦠 Diseases", "#DC2626"),
            (results["aggregate_report"]["patients_with_no_entities"], "📋 No-Entity Patients", "#6B7280"),
            (input_source.replace("_", " ").title(), "📥 Input Source", "#22C55E")
        ]

        for i, (value, label, color) in enumerate(metrics_data):
            with overview_cols[i]:
                st.markdown(f"""
                <div style="background: white; border-radius: 12px; padding: 24px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #E5E7EB;">
                    <div style="font-size: 2.5rem; margin-bottom: 12px;">{label.split()[0]}</div>
                    <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 8px;">{value}</div>
                    <div style="color: #4a5568; font-size: 0.9rem; font-weight: 500;">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        if upload_meta["text_columns"]:
            st.caption(
                f"📋 CSV text columns used: {', '.join(upload_meta['text_columns'])}"
                + (f" | 🆔 Patient ID column: {upload_meta['patient_id_column']}" if upload_meta["patient_id_column"] else "")
            )

        _render_model_status(results["model_meta"])

        # Modern tabs
        patient_tab, aggregate_tab = st.tabs(["👤 Patient-wise View", "📈 Aggregate Analysis"])

        with patient_tab:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Search & Filter")
            search_term = st.text_input(
                "",
                key="patient_search_term",
                placeholder="Search by patient ID, symptoms, or conditions...",
                label_visibility="collapsed"
            )
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All", "High", "Medium", "Low"],
                key="patient_risk_filter"
            )
            st.markdown('</div>', unsafe_allow_html=True)

            filtered_results = []
            keywords = _normalize_search_keywords(search_term)
            for patient_result in results["patients"]:
                matched_keywords = _matched_keywords(patient_result, keywords)
                matches_query = not keywords or bool(matched_keywords)
                matches_risk = risk_filter == "All" or patient_result["risk"]["risk_level"] == risk_filter
                if matches_query and matches_risk:
                    patient_with_match_meta = dict(patient_result)
                    patient_with_match_meta["search_matches"] = matched_keywords
                    patient_with_match_meta["highlighted_html"] = (
                        _highlight_keyword_matches(matched_keywords) + patient_result["highlighted_html"]
                    )
                    filtered_results.append(patient_with_match_meta)

            if not filtered_results:
                st.warning("🔍 No matching patients found with the current filters.")
            else:
                if len(filtered_results) > MAX_RENDERED_PATIENTS:
                    st.info(
                        f"📋 Showing first {MAX_RENDERED_PATIENTS} of {len(filtered_results)} matching patients."
                    )
                for index, patient_result in enumerate(filtered_results[:MAX_RENDERED_PATIENTS], start=1):
                    _render_patient_section(index, patient_result)

        with aggregate_tab:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Comprehensive Analysis Dashboard")

            # Risk distribution section
            risk_df = pd.DataFrame(
                [
                    {"Risk Level": level, "Patients": count}
                    for level, count in results["aggregate_report"]["overall_risk_distribution"].items()
                ]
            )
            symptom_df = pd.DataFrame(results["aggregate_report"]["most_common_symptoms"])

            risk_cols = st.columns([1, 1])
            with risk_cols[0]:
                st.markdown("#### 🎯 Risk Distribution")
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
                if not risk_df.empty:
                    st.bar_chart(risk_df.set_index("Risk Level"), color="#1E3A8A")

            with risk_cols[1]:
                st.markdown("#### 🩺 Most Common Symptoms")
                if symptom_df.empty:
                    st.info("ℹ️ No symptoms detected in the processed patients.")
                else:
                    st.dataframe(symptom_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Patient summary table
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📋 Patient Summary Overview")
            summary_rows = []
            for patient_result in results["patients"]:
                summary_rows.append(
                    {
                        "Patient ID": patient_result["patient_id"],
                        "Risk Level": patient_result["risk"]["risk_level"],
                        "Diseases": len(patient_result["structured_data"]["diseases"]),
                        "Symptoms": len(patient_result["structured_data"]["symptoms"]),
                        "Medications": len(patient_result["structured_data"]["medications"]),
                        "Procedures": len(patient_result["structured_data"]["procedures"]),
                        "Summary": patient_result["summary"],
                    }
                )
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Evaluation metrics with modern styling
            evaluation_metrics = uploaded_eval_metrics or {}
            if uploaded_eval_metrics:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🎯 Model Performance Metrics")
                metrics_cols = st.columns(5)
                metrics_info = [
                    (uploaded_eval_metrics['precision'], "Precision", "#1E3A8A"),
                    (uploaded_eval_metrics['recall'], "Recall", "#22C55E"),
                    (uploaded_eval_metrics['f1_score'], "F1 Score", "#EA580C"),
                    (uploaded_eval_metrics['accuracy'], "Accuracy", "#DC2626"),
                    (uploaded_eval_metrics.get('exact_match_rate', 0.0), "Exact Match", "#7C3AED")
                ]

                for i, (value, label, color) in enumerate(metrics_info):
                    with metrics_cols[i]:
                        st.markdown(f"""
                        <div style="background: white; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                            <div style="font-size: 1.8rem; font-weight: 700; color: {color}; margin-bottom: 8px;">{value:.1f}%</div>
                            <div style="color: #4a5568; font-size: 0.9rem; font-weight: 500;">{label}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Enhanced metrics chart
                fig = px.bar(
                    x=['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Exact Match'],
                    y=[
                        uploaded_eval_metrics['precision'],
                        uploaded_eval_metrics['recall'],
                        uploaded_eval_metrics['f1_score'],
                        uploaded_eval_metrics['accuracy'],
                        uploaded_eval_metrics.get('exact_match_rate', 0.0),
                    ],
                    title="📊 Model Evaluation Metrics",
                    labels={'x': 'Metric', 'y': 'Score (%)'},
                    color=['Precision', 'Recall', 'F1 Score', 'Accuracy', 'Exact Match'],
                    color_discrete_sequence=px.colors.qualitative.Vivid
                )
                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.9)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    margin=dict(t=50, r=20, b=20, l=20),
                    yaxis=dict(range=[0, 100]),
                    font=dict(size=14),
                    title_font=dict(size=20, color='#1E3A8A')
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info(
                    "💡 Upload evaluation data with gold standard labels to see detailed performance metrics."
                )

            with st.expander("🔧 Complete JSON Report", expanded=False):
                st.json(
                    {
                        "aggregate_report": results["aggregate_report"],
                        "patients": [
                            {
                                "patient_id": patient_result["patient_id"],
                                "structured_data": patient_result["structured_data"],
                                "risk": patient_result["risk"],
                                "insights": patient_result["insights"],
                                "summary": patient_result["summary"],
                            }
                            for patient_result in results["patients"]
                        ],
                        "evaluation": evaluation_metrics,
                    }
                )

    except Exception as exc:
        st.error("The app hit an unexpected error while processing this input.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.text(traceback.format_exc())
