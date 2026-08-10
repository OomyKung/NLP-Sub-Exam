"""Streamlit interface for the Network & IT Support Ticket Analyzer."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from nlp_utils import analyze_text


APP_ROOT = Path(__file__).resolve().parent
DATASET_PATH = APP_ROOT / "data" / "test_data.csv"

EXAMPLES = {
    "พิมพ์ข้อความเอง": "",
    "1. Network": (
        "เน็ตห้อง Lab 402 ใช้งานไม่ได้ตั้งแต่ 09:30 น. "
        "Cisco Router IP 192.168.1.1 ping ไม่เจอ ด่วนมากครับ ติดต่อ 081-234-5678"
    ),
    "2. Wi-Fi": "Wi-Fi ห้องประชุมชั้น 2 ช้ามากกกกกก และหลุดบ่อย",
    "3. DNS": "เปิด portal.example.com ไม่ได้ แต่ ping 8.8.8.8 ได้ น่าจะเป็นปัญหา DNS",
    "4. DHCP": "เครื่อง PC ไม่ได้รับ IP address จาก DHCP ที่ Building A",
    "5. English": "Production network is down for all users; Juniper Firewall is unreachable.",
    "6. Authentication": "Login เข้า VPN account ไม่ได้ รบกวนตรวจสอบ password ด่วน",
    "7. Application": "เว็บระบบลงทะเบียนขึ้น 500 error ที่ Room 210",
}


def _entity_table(entities: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Entity": entity["text"], "Label": entity["label"]} for entity in entities],
        columns=["Entity", "Label"],
    )


def _analyze_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in frame.iterrows():
        item = row.to_dict()
        raw_text = item.get("text", "")
        text = "" if pd.isna(raw_text) else str(raw_text)
        try:
            result = analyze_text(text)
            item["predicted_topic"] = result["topic"]
            item["predicted_priority"] = result["priority"]
            item["language"] = result["language"]
            item["extracted_entities"] = "; ".join(
                f"{entity['text']} ({entity['label']})" for entity in result["entities"]
            )
        except ValueError as exc:
            item["predicted_topic"] = "ERROR"
            item["predicted_priority"] = "ERROR"
            item["language"] = "Unknown"
            item["extracted_entities"] = str(exc)
        rows.append(item)
    return pd.DataFrame(rows)


def _accuracy(frame: pd.DataFrame, expected_column: str, predicted_column: str) -> float | None:
    if expected_column not in frame.columns or predicted_column not in frame.columns:
        return None
    valid = frame[expected_column].notna() & frame[expected_column].astype(str).str.strip().ne("")
    if not valid.any():
        return None
    expected = frame.loc[valid, expected_column].astype(str).str.strip().str.casefold()
    predicted = frame.loc[valid, predicted_column].astype(str).str.strip().str.casefold()
    return float((expected == predicted).mean())


def _load_csv(uploaded_file: BytesIO | None) -> pd.DataFrame | None:
    try:
        if uploaded_file is None:
            return pd.read_csv(DATASET_PATH)
        return pd.read_csv(uploaded_file)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        st.error(f"อ่านไฟล์ CSV ไม่สำเร็จ: {exc}")
        return None


def render_analysis_tab() -> None:
    st.subheader("วิเคราะห์ข้อความแจ้งปัญหา")
    selected = st.selectbox("เลือกตัวอย่างข้อความ", list(EXAMPLES))
    default_text = EXAMPLES[selected]
    text = st.text_area(
        "ข้อความภาษาไทย ภาษาอังกฤษ หรือข้อความผสม",
        value=default_text,
        height=150,
        placeholder="เช่น Wi-Fi ห้อง 401 ใช้งานไม่ได้ ติดต่อ admin@example.com",
    )

    if not st.button("วิเคราะห์ข้อความ", type="primary"):
        st.info("เลือกตัวอย่างหรือพิมพ์ข้อความ แล้วกดปุ่มวิเคราะห์")
        return

    try:
        result = analyze_text(text)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.divider()
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("ภาษา", result["language"])
    metric_two.metric("หัวข้อ", result["topic"])
    metric_three.metric("ความเร่งด่วน", result["priority"])

    st.markdown("#### Named Entities (Rule-based NER)")
    entities = _entity_table(result["entities"])
    if entities.empty:
        st.caption("ไม่พบ Entity ตามกฎที่กำหนด")
    else:
        st.dataframe(entities, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### ข้อความหลัง Cleansing")
        st.code(result["cleaned_text"], language=None)
        st.markdown("#### Filtered Tokens")
        st.write(" · ".join(result["filtered_tokens"]) or "ไม่พบ token")
    with right:
        st.markdown("#### ข้อความหลัง Normalization")
        st.code(result["normalized_text"], language=None)
        st.markdown("#### Topic Keyword Evidence")
        st.write(", ".join(result["topic_keywords"]) or "ไม่พบ keyword")
        st.markdown("#### Priority Keyword Evidence")
        st.write(", ".join(result["priority_keywords"]) or "ไม่พบ keyword (จึงเป็น LOW)")

    st.markdown("#### Topic Scores")
    st.bar_chart(pd.Series(result["topic_scores"], name="score"))


def render_dataset_tab() -> None:
    st.subheader("ทดสอบด้วย Dataset")
    source = st.radio("แหล่งข้อมูล", ["Bundled data/test_data.csv", "Upload CSV"], horizontal=True)
    uploaded = st.file_uploader("อัปโหลด CSV ที่มีคอลัมน์ text", type="csv") if source == "Upload CSV" else None

    if uploaded is not None:
        frame = _load_csv(uploaded)
    elif source == "Bundled data/test_data.csv":
        frame = _load_csv(None)
    else:
        st.info("เลือกไฟล์ CSV เพื่อเริ่มทดสอบ")
        return

    if frame is None:
        return
    if "text" not in frame.columns:
        st.error("CSV ต้องมีคอลัมน์ text")
        return

    result_frame = _analyze_dataset(frame)
    topic_accuracy = _accuracy(result_frame, "expected_topic", "predicted_topic")
    priority_accuracy = _accuracy(result_frame, "expected_priority", "predicted_priority")
    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("จำนวนข้อความ", len(result_frame))
    metric_two.metric("Topic Accuracy", "N/A" if topic_accuracy is None else f"{topic_accuracy:.1%}")
    metric_three.metric(
        "Priority Accuracy", "N/A" if priority_accuracy is None else f"{priority_accuracy:.1%}"
    )

    display_columns = [
        column
        for column in (
            "id",
            "text",
            "expected_topic",
            "predicted_topic",
            "expected_priority",
            "predicted_priority",
            "language",
            "extracted_entities",
        )
        if column in result_frame.columns
    ]
    st.dataframe(result_frame[display_columns], use_container_width=True, hide_index=True)
    csv_bytes = result_frame.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "ดาวน์โหลดผลการวิเคราะห์ CSV",
        data=csv_bytes,
        file_name="analysis_results.csv",
        mime="text/csv",
    )


def render_explanation_tab() -> None:
    st.subheader("NLP ที่ใช้ในโครงงาน")
    st.markdown(
        """
        โครงงานนี้เป็นระบบ **Rule-based NLP** ที่ออกแบบให้ตรวจสอบเหตุผลได้ง่าย
        และไม่ต้องใช้ API, GPU หรือโมเดลขนาดใหญ่

        1. **Regex & Cleansing** — ตรวจหา IPv4 ที่แต่ละ octet อยู่ในช่วง 0–255,
           เบอร์โทร, email, URL, เวลา และวันที่ แล้ว mask เฉพาะ PHONE, EMAIL และ URL
           โดยเก็บ IP ไว้เพราะเป็นข้อมูลสำคัญสำหรับ Network Support
        2. **Normalization** — ทำ Unicode NFC, รวมช่องว่าง, ลดการลากตัวอักษรและเครื่องหมายซ้ำ
           โดยปกป้อง IP, URL และ email ไม่ให้ถูกแก้ไข
        3. **Tokenization** — ข้อความ Thai/mixed ใช้
           `pythainlp.tokenize.word_tokenize(..., engine="newmm")` ส่วน English ใช้ regex tokenizer
        4. **Stopword Removal** — ใช้ Thai stopwords จาก PyThaiNLP และชุด English stopwords ขนาดเล็ก
        5. **Topic Identification** — นับ keyword แบบมีน้ำหนักเพื่อเลือก Network Connectivity,
           Wi-Fi, DNS, DHCP, Authentication, Hardware, Application หรือ Other พร้อมแสดง evidence
        6. **Rule-based NER** — ดึง IP_ADDRESS, PHONE, EMAIL, URL, TIME, DATE, VENDOR,
           DEVICE และ LOCATION ด้วย regex และ domain dictionaries
        7. **Priority Detection** — เลือกระดับ CRITICAL, HIGH, MEDIUM หรือ LOW จากคำเร่งด่วนที่พบ
        """
    )


st.set_page_config(page_title="Network & IT Support Ticket Analyzer", page_icon="🛠️", layout="wide")
st.title("Network & IT Support Ticket Analyzer")
st.caption("เว็บแอป NLP สำหรับวิเคราะห์ข้อความแจ้งปัญหา IT/Network ภาษาไทย ภาษาอังกฤษ และข้อความผสม")

tab_analyze, tab_dataset, tab_explain = st.tabs(
    ["🔎 วิเคราะห์ข้อความ", "📁 ทดสอบ Dataset", "🧠 NLP ที่ใช้"]
)
with tab_analyze:
    render_analysis_tab()
with tab_dataset:
    render_dataset_tab()
with tab_explain:
    render_explanation_tab()
