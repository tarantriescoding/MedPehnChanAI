import html
from typing import Dict, List, Tuple

import pandas as pd

from config import ENTITY_COLORS, INPUT_MODES


def resolve_input_text(typed_text: str, uploaded_text: str, image_text: str = "", input_mode: str = "Use typed text only") -> Tuple[str, str]:
    typed = (typed_text or "").strip()
    uploaded = (uploaded_text or "").strip()
    image = (image_text or "").strip()
    mode = input_mode if input_mode in INPUT_MODES else INPUT_MODES[0]

    if mode == "Use typed text only":
        return typed, "typed_text_only"
    if mode == "Use uploaded file only":
        return uploaded, "uploaded_file_only"
    if mode == "Use uploaded image only":
        return image, "uploaded_image_only"
    # Combine all inputs
    return " ".join(part for part in [typed, uploaded, image] if part).strip(), "combined_inputs"


def build_entity_table(entities: List[Dict[str, object]]) -> pd.DataFrame:
    columns = ["Entity", "Type", "Confidence", "Confidence Level", "Source"]
    if not entities:
        return pd.DataFrame(columns=columns)

    rows = []
    for entity in entities:
        rows.append(
            {
                "Entity": entity["text"],
                "Type": entity["label"],
                "Confidence": f"{float(entity['confidence']):.2f}%",
                "Confidence Level": entity.get("confidence_label", "Unknown"),
                "Source": entity.get("source", "model"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def highlight_entities_html(text: str, entities: List[Dict[str, object]]) -> str:
    if not text:
        return "<div style='padding:12px;border:1px solid #ddd;border-radius:12px;background:#fff;'></div>"

    ordered = sorted(entities, key=lambda item: (int(item["start"]), int(item["end"])))
    safe_text = text
    parts = []
    cursor = 0
    for entity in ordered:
        start = int(entity["start"])
        end = int(entity["end"])
        if start < cursor or start < 0 or end > len(safe_text):
            continue
        parts.append(html.escape(safe_text[cursor:start]))
        color = ENTITY_COLORS.get(entity["label"], "#eeeeee")
        title = f"{entity['label']} | {float(entity['confidence']):.2f}% | {entity.get('source', 'model')}"
        parts.append(
            "<span "
            f"style='background:{color}; padding:2px 6px; border-radius:6px; border:1px solid #d5d5d5;' "
            f"title='{html.escape(title)}'>{html.escape(safe_text[start:end])}</span>"
        )
        cursor = end
    parts.append(html.escape(safe_text[cursor:]))

    legend = (
        "<div style='margin-bottom:10px;'>"
        "<span style='background:#f6d2d2;padding:2px 7px;border-radius:6px;margin-right:6px;'>Disease</span>"
        "<span style='background:#f9e2c2;padding:2px 7px;border-radius:6px;margin-right:6px;'>Symptom</span>"
        "<span style='background:#d7efda;padding:2px 7px;border-radius:6px;margin-right:6px;'>Medication</span>"
        "<span style='background:#d8e7fb;padding:2px 7px;border-radius:6px;'>Procedure</span>"
        "</div>"
    )
    body = (
        "<div style='line-height:1.75;border:1px solid #ddd;background:#fff;padding:14px;border-radius:12px;'>"
        f"{''.join(parts)}</div>"
    )
    return legend + body


def structured_output(entities: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    payload = {"diseases": [], "symptoms": [], "medications": [], "procedures": []}
    key_map = {
        "Disease": "diseases",
        "Symptom": "symptoms",
        "Medication": "medications",
        "Procedure": "procedures",
    }
    for entity in entities:
        payload[key_map[entity["label"]]].append(
            {
                "name": entity["text"],
                "confidence": round(float(entity["confidence"]), 2),
                "confidence_label": entity.get("confidence_label"),
                "source": entity.get("source", "model"),
            }
        )
    return payload
