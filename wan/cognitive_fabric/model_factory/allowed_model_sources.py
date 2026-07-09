"""Model source policy (spec §20): only open, licensed, self-trained or
user-owned models. Proprietary closed-source weights are refused by policy."""
from __future__ import annotations

from typing import Any, Dict

ALLOWED_LICENSE_KINDS = {
    "apache-2.0", "mit", "bsd-3-clause", "openrail", "openrail++",
    "cc-by-4.0", "local/open", "self-trained", "user-owned",
}

REFUSED_NOTE = (
    "refused by local-model policy: only open, licensed, self-trained or "
    "user-owned model sources are allowed. Proprietary weights (e.g. "
    "Seedance) are never ingested or cloned.")


def check_source(entry: Dict[str, Any]) -> Dict[str, Any]:
    license_kind = str(entry.get("license", "")).lower()
    allowed = license_kind in ALLOWED_LICENSE_KINDS
    return {
        "id": entry.get("id"),
        "allowed": allowed,
        "license": license_kind,
        "note": "ok" if allowed else REFUSED_NOTE,
    }
