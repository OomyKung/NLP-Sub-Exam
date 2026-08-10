"""Reusable, explainable NLP utilities for the support-ticket analyzer.

The project intentionally uses deterministic rules so that students can explain
each NLP step during a demonstration and the app remains lightweight to deploy.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

from pythainlp.corpus.common import thai_stopwords
from pythainlp.tokenize import word_tokenize


# The patterns are kept readable because they are part of the assignment's
# Regex & Cleansing demonstration.
_IPV4_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_IPV4_RE = re.compile(
    rf"(?<![\d.])(?:{_IPV4_OCTET}\.){{3}}{_IPV4_OCTET}(?![\d.])"
)
_PHONE_RE = re.compile(
    r"(?<!\w)(?:0\d{1,2}[-\s]\d{3,4}[-\s]\d{4}|0\d{9})(?!\w)"
)
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![\w.-])"
)
_URL_RE = re.compile(
    r"(?i)(?:\bhttps?://[^\s<>]+|\bwww\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?:/[^\s<>]*)?|(?<![@\w])(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"
    r"(?:/[^\s<>]*)?)"
)
_TIME_RE = re.compile(
    r"(?<!\w)(?:[01]?\d|2[0-3]):[0-5]\d(?:\s?(?:AM|PM))?(?!\w)",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?<!\w)(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})(?!\w)"
)

_VENDORS = (
    "MikroTik",
    "TP-Link",
    "Ubiquiti",
    "Fortinet",
    "Juniper",
    "Huawei",
    "Aruba",
    "Cisco",
    "Dell",
    "HPE",
)
_DEVICES = (
    "Core Switch",
    "Access Point",
    "Firewall",
    "Router",
    "Switch",
    "Server",
    "Gateway",
    "Modem",
    "\u0e40\u0e04\u0e23\u0e37\u0e48\u0e2d\u0e07\u0e40\u0e0b\u0e34\u0e23\u0e4c\u0e1f\u0e40\u0e27\u0e2d\u0e23\u0e4c",
    "AP",
)

_LOCATION_PATTERNS = (
    re.compile(r"(?i)(?:\bRoom\s+[A-Za-z]?\s*\d+|ห้อง\s+(?:Lab\s+)?\d+)") ,
    re.compile(r"(?i)(?:\bBuilding\s+[A-Za-z0-9-]+|อาคาร\s+[A-Za-z0-9-]+)"),
    re.compile(r"(?i)(?:\bFloor\s+\d+|ชั้น\s+\d+)") ,
)

_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "please",
    "that",
    "the",
    "this",
    "to",
    "was",
    "we",
    "with",
}

# Each topic is represented by explainable keywords and weights. More
# specific terms receive a higher weight than broad words such as "IP".
_TOPIC_RULES: dict[str, tuple[tuple[str, int], ...]] = {
    "Network Connectivity": (
        ("network", 4),
        ("internet", 2),
        ("connectivity", 3),
        ("connection", 2),
        ("เชื่อมต่อ", 3),
        ("อินเทอร์เน็ต", 3),
        ("เน็ต", 3),
        ("ใช้งานไม่ได้", 2),
        ("ไม่สามารถเชื่อมต่อ", 3),
        ("unreachable", 3),
        ("down", 2),
        ("ping", 2),
        ("packet loss", 3),
        ("หลุด", 2),
        ("ไม่เจอ", 2),
        ("ip", 1),
    ),
    "Wi-Fi": (
        ("wi-fi", 5),
        ("wifi", 5),
        ("wireless", 4),
        ("ไวไฟ", 5),
        ("สัญญาณไร้สาย", 5),
    ),
    "DNS": (
        ("dns", 6),
        ("domain", 5),
        ("ชื่อโดเมน", 5),
        ("resolve", 5),
        ("resolved", 5),
        ("nslookup", 6),
        ("เปิดเว็บ", 2),
    ),
    "DHCP": (
        ("dhcp", 6),
        ("แจก ip", 5),
        ("รับ ip", 5),
        ("assigned ip", 5),
        ("ip address", 2),
    ),
    "Authentication": (
        ("authentication", 6),
        ("login", 6),
        ("log in", 6),
        ("เข้าสู่ระบบ", 6),
        ("รหัสผ่าน", 5),
        ("password", 5),
        ("username", 4),
        ("account", 3),
        ("vpn", 3),
    ),
    "Hardware": (
        ("hardware", 5),
        ("router", 3),
        ("switch", 4),
        ("firewall", 3),
        ("server", 3),
        ("modem", 3),
        ("gateway", 3),
        ("อุปกรณ์", 3),
        ("เครื่อง", 1),
        ("ไฟสี", 4),
        ("เสีย", 2),
        ("เปิดไม่ติด", 4),
    ),
    "Application": (
        ("application", 5),
        ("software", 5),
        ("app", 4),
        ("website", 4),
        ("portal", 5),
        ("500 error", 6),
        ("error", 2),
        ("ระบบงาน", 4),
        ("โปรแกรม", 4),
        ("เว็บ", 2),
    ),
}

_PRIORITY_RULES: dict[str, tuple[str, ...]] = {
    "CRITICAL": (
        "critical",
        "ฉุกเฉิน",
        "ทั้งระบบ",
        "ล่มทั้งระบบ",
        "ทั้งอาคาร",
        "production down",
        "affects all users",
        "every user is affected",
        "กระทบทุกคน",
        "กระทบผู้ใช้ทั้งหมด",
        "ใช้งานไม่ได้ทั้งองค์กร",
    ),
    "HIGH": (
        "urgent",
        "ด่วน",
        "ใช้งานไม่ได้",
        "down",
        "unreachable",
        "ล่ม",
        "ไม่สามารถเชื่อมต่อ",
        "ไม่เจอ",
    ),
    "MEDIUM": (
        "ช้า",
        "slow",
        "หลุดบ่อย",
        "intermittent",
        "ไม่เสถียร",
        "unstable",
        "packet loss",
        "กระตุก",
        "ล่าช้า",
    ),
}


def _has_thai(text: str) -> bool:
    return bool(re.search(r"[\u0E00-\u0E7F]", text))


def _has_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def detect_language(text: str) -> str:
    """Classify text using the presence of Thai and Latin alphabet characters."""

    thai = _has_thai(text)
    latin = _has_latin(text)
    if thai and latin:
        return "Mixed Thai-English"
    if thai:
        return "Thai"
    if latin:
        return "English"
    return "Unknown"


def _protect_matches(text: str, patterns: Iterable[re.Pattern[str]]) -> tuple[str, list[str]]:
    protected: list[str] = []

    def replace(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    protected_text = text
    for pattern in patterns:
        protected_text = pattern.sub(replace, protected_text)
    return protected_text, protected


def _restore_matches(text: str, protected: list[str]) -> str:
    for index, value in enumerate(protected):
        text = text.replace(f"\x00{index}\x00", value)
    return text


def normalize_text(text: str) -> str:
    """Normalize Unicode, spaces, stretching, and repeated punctuation safely."""

    normalized = unicodedata.normalize("NFC", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized, protected = _protect_matches(
        normalized, (_EMAIL_RE, _URL_RE, _IPV4_RE, _PHONE_RE)
    )
    # Three or more identical non-whitespace characters are usually emphasis
    # (for example "ช้ามากกกก"), while protected technical strings are restored.
    normalized = re.sub(r"([^\W_])\1{2,}", r"\1", normalized, flags=re.UNICODE)
    normalized = re.sub(r"([!?！？])\1+", r"\1", normalized)
    return _restore_matches(normalized, protected)


def _trim_url(value: str) -> str:
    return value.rstrip(".,!?;:)]}。！？ฯ")


def _match_value(pattern: re.Pattern[str], text: str) -> list[dict[str, Any]]:
    results = []
    for match in pattern.finditer(text):
        value = match.group(0)
        if pattern is _URL_RE:
            trimmed = _trim_url(value)
            end = match.start() + len(trimmed)
            value = trimmed
        else:
            end = match.end()
        results.append({"text": value, "start": match.start(), "end": end})
    return results


def extract_regex_entities(text: str) -> dict[str, list[dict[str, Any]]]:
    """Extract structured values using validated regular expressions."""

    return {
        "IP_ADDRESS": _match_value(_IPV4_RE, text),
        "PHONE": _match_value(_PHONE_RE, text),
        "EMAIL": _match_value(_EMAIL_RE, text),
        "URL": _match_value(_URL_RE, text),
        "TIME": _match_value(_TIME_RE, text),
        "DATE": _match_value(_DATE_RE, text),
    }


def cleanse_text(text: str) -> str:
    """Mask personal contact data but preserve IP addresses for IT diagnosis."""

    cleansed = _EMAIL_RE.sub("[EMAIL]", text)
    cleansed = _URL_RE.sub(lambda match: "[URL]", cleansed)
    cleansed = _PHONE_RE.sub("[PHONE]", cleansed)
    return cleansed


def _regex_tokenize(text: str) -> list[str]:
    return re.findall(
        r"\[[A-Z_]+\]|(?:[A-Za-z]+(?:[-'][A-Za-z]+)?)|"
        r"(?:\d+(?:\.\d+)+)|\d+|[\u0E00-\u0E7F]+|[^\w\s]",
        text,
        flags=re.UNICODE,
    )


def tokenize_text(text: str, language: str | None = None) -> list[str]:
    """Tokenize Thai/mixed text with PyThaiNLP and English with a small regex."""

    language = language or detect_language(text)
    if language in {"Thai", "Mixed Thai-English"}:
        return word_tokenize(text, engine="newmm", keep_whitespace=False)
    return _regex_tokenize(text)


def remove_stopwords(tokens: Iterable[str]) -> list[str]:
    """Remove Thai and English stopwords while retaining technical tokens."""

    thai_words = set(thai_stopwords())
    filtered: list[str] = []
    for token in tokens:
        stripped = token.strip()
        if not stripped or not re.search(r"[A-Za-z0-9\u0E00-\u0E7F]", stripped):
            continue
        if stripped in thai_words or stripped.casefold() in _ENGLISH_STOPWORDS:
            continue
        filtered.append(stripped)
    return filtered


def _contains_keyword(text: str, keyword: str) -> bool:
    if re.fullmatch(r"[A-Za-z0-9 -]+", keyword):
        return re.search(rf"(?<![A-Za-z0-9]){re.escape(keyword)}(?![A-Za-z0-9])", text, re.I) is not None
    return keyword.casefold() in text.casefold()


def identify_topic(text: str) -> dict[str, Any]:
    """Identify a topic and expose the keyword evidence and weighted scores."""

    scores = {topic: 0 for topic in (*_TOPIC_RULES.keys(), "Other")}
    matched: dict[str, list[str]] = {topic: [] for topic in _TOPIC_RULES}
    for topic, rules in _TOPIC_RULES.items():
        for keyword, weight in rules:
            if _contains_keyword(text, keyword):
                scores[topic] += weight
                matched[topic].append(keyword)

    topic_order = list(_TOPIC_RULES.keys()) + ["Other"]
    topic = max(topic_order, key=lambda item: scores[item])
    if scores[topic] == 0:
        topic = "Other"
    return {
        "topic": topic,
        "topic_scores": scores,
        "topic_keywords": matched.get(topic, []),
        "matched_topic_keywords": matched,
    }


def detect_priority(text: str) -> dict[str, Any]:
    """Assign urgency by highest matching severity, with keyword evidence."""

    matched: dict[str, list[str]] = {level: [] for level in _PRIORITY_RULES}
    for level, keywords in _PRIORITY_RULES.items():
        for keyword in keywords:
            if _contains_keyword(text, keyword):
                matched[level].append(keyword)

    for level in ("CRITICAL", "HIGH", "MEDIUM"):
        if matched[level]:
            return {"priority": level, "priority_keywords": matched[level]}
    return {"priority": "LOW", "priority_keywords": []}


def _add_entity(
    entities: list[dict[str, Any]],
    label: str,
    value: str,
    start: int,
    end: int,
) -> None:
    key = (label, value.casefold(), start, end)
    if not any(
        (entity["label"], entity["text"].casefold(), entity["start"], entity["end"]) == key
        for entity in entities
    ):
        entities.append({"text": value, "label": label, "start": start, "end": end})


def extract_entities(text: str) -> list[dict[str, Any]]:
    """Extract networking and support-domain entities without a large model."""

    entities: list[dict[str, Any]] = []
    regex_entities = extract_regex_entities(text)
    for label, matches in regex_entities.items():
        for match in matches:
            _add_entity(entities, label, match["text"], match["start"], match["end"])

    for vendor in _VENDORS:
        for match in re.finditer(rf"(?<![\w-]){re.escape(vendor)}(?![\w-])", text, re.I):
            _add_entity(entities, "VENDOR", match.group(0), match.start(), match.end())

    for device in _DEVICES:
        for match in re.finditer(rf"(?<![\w-]){re.escape(device)}(?![\w-])", text, re.I):
            _add_entity(entities, "DEVICE", match.group(0), match.start(), match.end())

    for pattern in _LOCATION_PATTERNS:
        for match in pattern.finditer(text):
            _add_entity(entities, "LOCATION", match.group(0), match.start(), match.end())

    entities.sort(key=lambda entity: (entity["start"], entity["end"], entity["label"]))
    return entities


def analyze_text(text: str) -> dict[str, Any]:
    """Run the complete analysis pipeline and return a serializable dictionary."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("กรุณาป้อนข้อความแจ้งปัญหาอย่างน้อยหนึ่งข้อความ")

    original_text = text
    normalized_text = normalize_text(original_text)
    cleaned_text = cleanse_text(normalized_text)
    language = detect_language(normalized_text)
    raw_tokens = tokenize_text(cleaned_text, language)
    filtered_tokens = remove_stopwords(raw_tokens)
    topic_result = identify_topic(normalized_text)
    priority_result = detect_priority(normalized_text)

    return {
        "original_text": original_text,
        "language": language,
        "cleaned_text": cleaned_text,
        "normalized_text": normalized_text,
        "raw_tokens": raw_tokens,
        "tokens": filtered_tokens,
        "filtered_tokens": filtered_tokens,
        "topic": topic_result["topic"],
        "topic_scores": topic_result["topic_scores"],
        "topic_keywords": topic_result["topic_keywords"],
        "matched_topic_keywords": topic_result["matched_topic_keywords"],
        "priority": priority_result["priority"],
        "priority_keywords": priority_result["priority_keywords"],
        "entities": extract_entities(original_text),
    }


__all__ = [
    "analyze_text",
    "cleanse_text",
    "detect_language",
    "detect_priority",
    "extract_entities",
    "extract_regex_entities",
    "identify_topic",
    "normalize_text",
    "remove_stopwords",
    "tokenize_text",
]
