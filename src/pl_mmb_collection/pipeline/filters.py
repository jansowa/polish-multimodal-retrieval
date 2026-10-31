from __future__ import annotations

import re
from urllib.parse import urlparse

from ..config import FilterConfig
from ..models import DatasetRecord


def license_is_allowed(name: str | None, url: str | None, allowed: list[str]) -> bool:
    """Require agreement of an explicit license name and its canonical CC URL."""
    if not name or not url or name not in allowed:
        return False
    parsed = urlparse(url if not url.startswith("//") else "https:" + url)
    if parsed.scheme not in {"https", "http"} or parsed.hostname not in {"creativecommons.org", "www.creativecommons.org"}:
        return False
    path = parsed.path.rstrip("/")
    if name == "CC0":
        return path in {"/publicdomain/zero/1.0", "/publicdomain/zero/1.0/deed.en", "/publicdomain/zero/1.0/deed.pl"}
    match = re.fullmatch(r"CC (BY(?:-SA)?) (1\.0|2\.0|2\.5|3\.0|4\.0)", name)
    if not match:
        return False
    prefix = f"/licenses/{match[1].lower()}/{match[2]}"
    return path in {prefix, prefix + "/deed.en", prefix + "/deed.pl", prefix + "/legalcode"}


def rejection_reasons(record: DatasetRecord, config: FilterConfig) -> list[str]:
    reasons = []
    if record.width is None or record.height is None:
        reasons.append("missing_dimensions")
    else:
        if record.width < config.min_width:
            reasons.append("min_width")
        if record.height < config.min_height:
            reasons.append("min_height")
        if record.width * record.height < config.min_area:
            reasons.append("min_area")
    if record.image_url != record.image_original_url:
        if (record.image_width is None or record.image_height is None
                or record.image_width < config.min_width or record.image_height < config.min_height
                or record.image_width * record.image_height < config.min_area):
            reasons.append("small_thumbnail")
    if record.mime_type not in config.allowed_mime_types:
        reasons.append("mime_type")
    if config.reject_svg and record.mime_type == "image/svg+xml":
        reasons.append("svg")
    if config.require_license and (not record.license or not record.license_url):
        reasons.append("missing_license")
    if config.allowed_licenses and not license_is_allowed(record.license, record.license_url, config.allowed_licenses):
        reasons.append("license_not_allowed")
    if config.require_text:
        useful = [t for t in record.text_candidates if t.get("text") and len(t["text"].strip()) >= 3
                  and t.get("source") != "depicts"]
        # Common camera-generated basenames do not provide semantic text.
        useful = [t for t in useful if t["source"] != "image_title" or not re.fullmatch(
            r"(?:IMG|DSC|DSCF|PXL|DCIM)[ _-]?\d+[\w _-]*", t["text"], re.IGNORECASE)]
        if not useful:
            reasons.append("missing_text")
    for pattern in config.excluded_category_patterns:
        if any(re.search(pattern, category, re.IGNORECASE) for category in record.categories):
            reasons.append("excluded_category:" + pattern)
    return reasons
