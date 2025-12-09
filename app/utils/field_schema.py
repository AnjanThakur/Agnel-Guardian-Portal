# app/utils/field_schema.py
import re
from typing import Dict, Any

from app.models.constants import FIELD_SCHEMAS


def _validate_field(field_name: str, value: str) -> bool:
    schema = FIELD_SCHEMAS.get(field_name)
    if not schema or not value:
        return False
    return bool(re.search(schema["pattern"], value.strip(), re.I))


def reassign_fields_by_schema(fields: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Cleans up OCR mix-ups by comparing each field's value
    with expected regex patterns and redistributing if needed.
    """
    out: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in fields.items()}

    def _move_value(src: str, dst: str) -> None:
        if src not in out or dst not in out:
            return
        val = (out[src].get("value") or "").strip()
        if not val:
            return
        prev = (out[dst].get("value") or "").strip()
        out[dst]["value"] = (prev + "\n" + val).strip() if prev else val
        out[src]["value"] = ""
        out[src]["confidence"] = 0.0
        out[dst]["confidence"] = max(float(out[dst].get("confidence") or 0.0), 0.9)

    # Name accidentally inside comments (last line)
    cmnt = (out.get("comments", {}).get("value") or "")
    pname = (out.get("parent_name", {}).get("value") or "")
    if cmnt:
        lines = [ln.strip() for ln in cmnt.splitlines() if ln.strip()]
        if lines and _validate_field("parent_name", lines[-1]) and not pname:
            out.setdefault("parent_name", {})
            out["parent_name"]["value"] = lines[-1]
            out["parent_name"]["confidence"] = 0.9
            lines.pop(-1)
            out.setdefault("comments", {})
            out["comments"]["value"] = "\n".join(lines).strip()

    # Bad-looking parent_name → move to comments
    pname = (out.get("parent_name", {}).get("value") or "")
    if pname and not _validate_field("parent_name", pname):
        _move_value("parent_name", "comments")

    # Phone number hidden inside other fields
    phone_pat = FIELD_SCHEMAS["contact_number"]["pattern"]
    for key in ["comments", "parent_name"]:
        val = out.get(key, {}).get("value", "")
        if val and re.search(phone_pat, val):
            _move_value(key, "contact_number")

    # Ward/Dept swap
    ward = (out.get("ward_name", {}).get("value") or "")
    dept = (out.get("department_year", {}).get("value") or "")
    if ward and _validate_field("department_year", ward) and not dept:
        _move_value("ward_name", "department_year")

    return out
