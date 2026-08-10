import pytest

from nlp_utils import (
    analyze_text,
    cleanse_text,
    detect_language,
    detect_priority,
    extract_entities,
    extract_regex_entities,
    identify_topic,
    normalize_text,
)


def labels(result):
    return {(item["text"], item["label"]) for item in result["entities"]}


def test_language_detection():
    assert detect_language("อินเทอร์เน็ตใช้งานไม่ได้") == "Thai"
    assert detect_language("Internet is down") == "English"
    assert detect_language("Cisco Router ใช้งานไม่ได้") == "Mixed Thai-English"


def test_regex_entities_and_contact_masking():
    text = (
        "IP 192.168.1.1, invalid 999.1.1.1, phone 081-234-5678, "
        "email admin@example.com, URL https://example.com, time 09:30, date 10/08/2026"
    )
    extracted = extract_regex_entities(text)
    assert [item["text"] for item in extracted["IP_ADDRESS"]] == ["192.168.1.1"]
    assert extracted["PHONE"][0]["text"] == "081-234-5678"
    assert extracted["EMAIL"][0]["text"] == "admin@example.com"
    assert extracted["URL"][0]["text"] == "https://example.com"
    assert extracted["TIME"][0]["text"] == "09:30"
    assert extracted["DATE"][0]["text"] == "10/08/2026"
    cleaned = cleanse_text(text)
    assert "[PHONE]" in cleaned
    assert "[EMAIL]" in cleaned
    assert "[URL]" in cleaned
    assert "192.168.1.1" in cleaned
    assert "999.1.1.1" in cleaned


def test_domain_entities():
    result = analyze_text(
        "Cisco Router ที่ห้อง Lab 402 ใช้ IP 192.168.1.1 เวลา 09:30 "
        "ติดต่อ 081-234-5678 หรือ admin@example.com"
    )
    found = labels(result)
    assert ("Cisco", "VENDOR") in found
    assert ("Router", "DEVICE") in found
    assert ("ห้อง Lab 402", "LOCATION") in found
    assert ("192.168.1.1", "IP_ADDRESS") in found
    assert ("09:30", "TIME") in found
    assert ("081-234-5678", "PHONE") in found
    assert ("admin@example.com", "EMAIL") in found


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("เน็ตล่ม ping ไม่เจอ", "Network Connectivity"),
        ("Wi-Fi ช้ามาก", "Wi-Fi"),
        ("เปิด domain ไม่ได้ มีปัญหา DNS", "DNS"),
        ("ไม่ได้รับ IP จาก DHCP", "DHCP"),
        ("Login เข้า account ไม่ได้", "Authentication"),
        ("Switch มีไฟสีแดง", "Hardware"),
        ("เว็บ portal ขึ้น 500 error", "Application"),
    ],
)
def test_topic_rules(text, expected):
    assert identify_topic(text)["topic"] == expected


def test_priority_rules():
    assert detect_priority("ทั้งระบบล่ม กระทบทุกคน") ["priority"] == "CRITICAL"
    assert detect_priority("Network down ด่วน") ["priority"] == "HIGH"
    assert detect_priority("Wi-Fi ช้าและหลุดบ่อย") ["priority"] == "MEDIUM"
    assert detect_priority("ขอข้อมูลการตั้งค่า") ["priority"] == "LOW"


def test_normalization_reduces_stretching_without_corrupting_ip():
    assert normalize_text("ช้ามากกกกกก!!!") == "ช้ามาก!"
    assert normalize_text("IP 192.168.1.1 Wi-Fi example.com") == "IP 192.168.1.1 Wi-Fi example.com"


def test_blank_input_handling():
    with pytest.raises(ValueError, match="กรุณาป้อน"):
        analyze_text("   ")


def test_analysis_contract_and_stopword_removal():
    result = analyze_text("Cisco Router ใช้งานไม่ได้ ด่วน")
    for key in (
        "original_text",
        "language",
        "cleaned_text",
        "normalized_text",
        "raw_tokens",
        "tokens",
        "topic",
        "topic_scores",
        "topic_keywords",
        "priority",
        "priority_keywords",
        "entities",
    ):
        assert key in result
    assert "Cisco" in result["tokens"]
    assert "Router" in result["tokens"]
