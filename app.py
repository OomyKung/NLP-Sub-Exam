"""Streamlit interface for the Network & IT Support Ticket Analyzer."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from nlp_utils import analyze_text


APP_ROOT = Path(__file__).resolve().parent
DATASET_PATH = APP_ROOT / "data" / "test_data.csv"

EXAMPLES = {
    "พิมพ์ข้อความเอง": "",
    "1. Network Down": (
        "เน็ตห้อง Lab 402 ใช้งานไม่ได้ตั้งแต่ 09:30 น. "
        "Cisco Router IP 192.168.1.1 ping ไม่เจอ ด่วนมากครับ ติดต่อ 081-234-5678"
    ),
    "2. Wi-Fi Issue": "Wi-Fi ห้องประชุมชั้น 2 ช้ามากกกกกก และหลุดบ่อย",
    "3. DNS Issue": "เปิด portal.example.com ไม่ได้ และค้นหา domain ช้ามาก น่าจะเป็นปัญหา DNS",
    "4. DHCP Issue": "เครื่อง PC ไม่ได้รับ IP address จาก DHCP ที่ Building A",
    "5. Authentication Issue": "Login เข้า VPN account ไม่ได้ รบกวนตรวจสอบ password ด่วน",
    "6. English Example": "Production network is down for all users; Juniper Firewall is unreachable.",
    "7. Application Issue": "เว็บระบบลงทะเบียนขึ้น 500 error ที่ Room 210",
}

PRIORITY_STYLE = {
    "CRITICAL": ("🔴", "critical"),
    "HIGH": ("🟠", "high"),
    "MEDIUM": ("🟡", "medium"),
    "LOW": ("🟢", "low"),
}


st.set_page_config(
    page_title="Network & IT Support Ticket Analyzer",
    page_icon="🛠️",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            background: linear-gradient(135deg, #10243f 0%, #173d61 100%);
            border: 1px solid #28577e;
            border-radius: 18px;
            padding: 1.6rem 1.8rem 1.45rem;
            color: #f4f8fc;
            margin-bottom: 1.25rem;
        }
        .hero-kicker {
            color: #8bd3ff;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.45rem;
        }
        .hero h1 {
            color: #ffffff;
            font-size: clamp(1.8rem, 4vw, 2.65rem);
            line-height: 1.12;
            margin: 0 0 0.55rem;
        }
        .hero p {
            color: #d6e6f3;
            font-size: 1rem;
            margin: 0;
        }
        .chip-row { margin-top: 1.05rem; }
        .chip {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 999px;
            color: #e6f4ff;
            display: inline-block;
            font-size: 0.78rem;
            margin: 0.18rem 0.28rem 0 0;
            padding: 0.3rem 0.65rem;
        }
        .summary-card {
            background: #ffffff;
            border: 1px solid #dce6ef;
            border-radius: 14px;
            box-shadow: 0 4px 15px rgba(20, 49, 78, 0.06);
            min-height: 106px;
            padding: 0.95rem 1rem;
        }
        .summary-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .summary-value {
            color: #10243f;
            font-size: 1.22rem;
            font-weight: 750;
            line-height: 1.3;
            margin-top: 0.55rem;
            word-break: break-word;
        }
        .priority-value.critical { color: #b42318; }
        .priority-value.high { color: #b54708; }
        .priority-value.medium { color: #9a6700; }
        .priority-value.low { color: #087443; }
        .token-pill, .evidence-pill {
            background: #eef6fb;
            border: 1px solid #cfe3f0;
            border-radius: 999px;
            color: #174a6b;
            display: inline-block;
            font-size: 0.84rem;
            margin: 0.18rem 0.22rem 0.18rem 0;
            padding: 0.25rem 0.58rem;
        }
        .evidence-pill { background: #f1f5ff; border-color: #d7def5; color: #3d4d83; }
        .section-note { color: #64748b; font-size: 0.9rem; }
        .pipeline {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin: 0.7rem 0 1.2rem;
        }
        .pipeline-step {
            background: #f5f8fb;
            border: 1px solid #dce6ef;
            border-radius: 10px;
            color: #24445d;
            font-size: 0.86rem;
            padding: 0.48rem 0.65rem;
        }
        .pipeline-arrow { color: #8ba1b4; font-weight: 700; }
        .footer {
            border-top: 1px solid #e2e8f0;
            color: #8292a3;
            font-size: 0.78rem;
            margin-top: 2rem;
            padding-top: 1rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Natural Language Processing Project</div>
        <h1>🛠️ Network &amp; IT Support<br>Ticket Analyzer</h1>
        <p>วิเคราะห์ข้อความแจ้งปัญหา IT/Network ภาษาไทย ภาษาอังกฤษ และข้อความผสม<br>
        พร้อมแสดงเหตุผลของ Topic, Priority และ Named Entities อย่างเข้าใจง่าย</p>
        <div class="chip-row">
            <span class="chip">Regex</span>
            <span class="chip">Tokenization</span>
            <span class="chip">Normalization</span>
            <span class="chip">Topic Classification</span>
            <span class="chip">Rule-based NER</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def _entity_table(entities: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"Entity": entity["text"], "Type": entity["label"]} for entity in entities],
        columns=["Entity", "Type"],
    )


def _summary_card(label: str, value: str, extra_class: str = "") -> str:
    return (
        '<div class="summary-card">'
        f'<div class="summary-label">{escape(label)}</div>'
        f'<div class="summary-value {extra_class}">{escape(value)}</div>'
        "</div>"
    )


def _priority_text(priority: str) -> str:
    icon, _ = PRIORITY_STYLE.get(priority, ("•", "low"))
    return f"{icon} {priority}"


def _pills(values: list[str], css_class: str = "token-pill") -> str:
    if not values:
        return '<span class="section-note">ไม่พบข้อมูล</span>'
    return " ".join(
        f'<span class="{css_class}">{escape(str(value))}</span>' for value in values
    )


def _analyze_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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


def _load_csv(uploaded_file: Any | None) -> pd.DataFrame | None:
    try:
        if uploaded_file is None:
            return pd.read_csv(DATASET_PATH)
        return pd.read_csv(uploaded_file)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
    ) as exc:
        st.error(f"อ่านไฟล์ CSV ไม่สำเร็จ: {exc}")
        return None


def render_analysis_tab() -> None:
    st.header("วิเคราะห์ข้อความแจ้งปัญหา")
    st.markdown(
        '<p class="section-note">รองรับข้อความภาษาไทย ภาษาอังกฤษ และข้อความแบบผสมจากงาน IT Support</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        selected = st.selectbox("ตัวอย่างข้อความ", list(EXAMPLES), label_visibility="visible")
        text = st.text_area(
            "ข้อความแจ้งปัญหา",
            value=EXAMPLES[selected],
            height=150,
            placeholder="เช่น Wi-Fi ห้อง 401 ใช้งานไม่ได้ ติดต่อ admin@example.com",
            help="เลือกตัวอย่างด้านบนหรือพิมพ์ ticket ของคุณเอง",
        )
        analyze_clicked = st.button(
            "🔎 วิเคราะห์ข้อความ",
            type="primary",
            use_container_width=True,
        )

    if not analyze_clicked:
        st.info("เลือกตัวอย่างหรือพิมพ์ข้อความ แล้วกดปุ่มวิเคราะห์เพื่อดูผลลัพธ์")
        return

    try:
        result = analyze_text(text)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.divider()
    st.header("ผลการวิเคราะห์")
    priority_icon, priority_class = PRIORITY_STYLE.get(result["priority"], ("•", "low"))
    summary_columns = st.columns(4)
    with summary_columns[0]:
        st.markdown(_summary_card("Topic", result["topic"]), unsafe_allow_html=True)
    with summary_columns[1]:
        st.markdown(
            _summary_card("Priority", f"{priority_icon} {result['priority']}", f"priority-value {priority_class}"),
            unsafe_allow_html=True,
        )
    with summary_columns[2]:
        st.markdown(_summary_card("Language", result["language"]), unsafe_allow_html=True)
    with summary_columns[3]:
        st.markdown(_summary_card("Entities Found", str(len(result["entities"]))), unsafe_allow_html=True)

    st.subheader("ข้อมูลที่ตรวจพบ")
    entities = _entity_table(result["entities"])
    if entities.empty:
        st.caption("ไม่พบ Entity ตามกฎที่กำหนด")
    else:
        st.dataframe(entities, use_container_width=True, hide_index=True)

    st.subheader("เปรียบเทียบข้อความ")
    original_column, cleaned_column = st.columns(2)
    with original_column:
        st.markdown("**Original Text**")
        st.code(result["original_text"], language=None)
    with cleaned_column:
        st.markdown("**Cleaned Text**")
        st.code(result["cleaned_text"], language=None)

    with st.expander("ดูรายละเอียดการประมวลผล NLP", expanded=True):
        normalized_column, token_column = st.columns(2)
        with normalized_column:
            st.markdown("#### Normalization")
            st.code(result["normalized_text"], language=None)
            st.caption("Unicode NFC, รวมช่องว่าง และลดตัวอักษร/เครื่องหมายซ้ำ")
        with token_column:
            st.markdown("#### Filtered Tokens")
            st.markdown(_pills(result["filtered_tokens"]), unsafe_allow_html=True)
            st.caption(f"เหลือ {len(result['filtered_tokens'])} tokens หลังตัด stopwords")

        st.markdown("#### Topic Evidence")
        st.markdown(
            f"**Predicted Topic:** `{escape(result['topic'])}`  \n**Matched keywords:** "
            + _pills(result["topic_keywords"], "evidence-pill"),
            unsafe_allow_html=True,
        )
        st.bar_chart(pd.Series(result["topic_scores"], name="score"), height=260)

        priority_keywords = result["priority_keywords"]
        st.markdown("#### Priority Evidence")
        st.markdown(
            f"**Priority:** `{escape(_priority_text(result['priority']))}`  \n**Matched keywords:** "
            + _pills(priority_keywords, "evidence-pill"),
            unsafe_allow_html=True,
        )

        with st.expander("ดู Raw Tokens เพิ่มเติม"):
            st.write(result["raw_tokens"])


def render_dataset_tab() -> None:
    st.header("📁 Dataset Testing")
    st.markdown(
        '<p class="section-note">รัน NLP pipeline กับ dataset ที่ bundled หรือ CSV ของคุณเอง แล้วตรวจสอบ accuracy</p>',
        unsafe_allow_html=True,
    )
    source = st.radio(
        "แหล่งข้อมูล",
        ["ใช้ bundled test dataset", "Upload CSV"],
        horizontal=True,
    )
    uploaded = (
        st.file_uploader("อัปโหลด CSV ที่มีคอลัมน์ text", type="csv")
        if source == "Upload CSV"
        else None
    )

    if source == "Upload CSV" and uploaded is None:
        st.info("เลือกไฟล์ CSV เพื่อเริ่มทดสอบ")
        return
    frame = _load_csv(uploaded if source == "Upload CSV" else None)
    if frame is None:
        return
    if "text" not in frame.columns:
        st.error("CSV ต้องมีคอลัมน์ text")
        return

    result_frame = _analyze_dataset(frame)
    topic_accuracy = _accuracy(result_frame, "expected_topic", "predicted_topic")
    priority_accuracy = _accuracy(result_frame, "expected_priority", "predicted_priority")
    metrics = st.columns(3)
    metrics[0].metric("Messages", len(result_frame))
    metrics[1].metric("Topic Accuracy", "N/A" if topic_accuracy is None else f"{topic_accuracy:.1%}")
    metrics[2].metric(
        "Priority Accuracy",
        "N/A" if priority_accuracy is None else f"{priority_accuracy:.1%}",
    )

    has_expected_labels = "expected_topic" in result_frame.columns or "expected_priority" in result_frame.columns
    show_errors = st.checkbox(
        "แสดงเฉพาะรายการที่ทำนายผิด",
        disabled=not has_expected_labels,
        help="ใช้ได้เมื่อ CSV มี expected_topic หรือ expected_priority",
    )
    visible_frame = result_frame
    if show_errors:
        mismatch = pd.Series(False, index=result_frame.index)
        if "expected_topic" in result_frame.columns:
            mismatch |= result_frame["expected_topic"].astype(str).str.casefold() != result_frame[
                "predicted_topic"
            ].astype(str).str.casefold()
        if "expected_priority" in result_frame.columns:
            mismatch |= result_frame["expected_priority"].astype(str).str.casefold() != result_frame[
                "predicted_priority"
            ].astype(str).str.casefold()
        visible_frame = result_frame[mismatch]
        st.caption(f"พบรายการที่ทำนายผิด {len(visible_frame)} รายการ")

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
        if column in visible_frame.columns
    ]
    st.dataframe(visible_frame[display_columns], use_container_width=True, hide_index=True)
    csv_bytes = result_frame.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "ดาวน์โหลดผลการวิเคราะห์ CSV",
        data=csv_bytes,
        file_name="analysis_results.csv",
        mime="text/csv",
    )


def render_explanation_tab() -> None:
    st.header("🧠 NLP ที่ใช้ในโครงงาน")
    st.markdown(
        '<p class="section-note">ทำความเข้าใจ pipeline ได้ภายในหนึ่งนาที และเปิดดูรายละเอียดเพิ่มเติมได้ในแต่ละหัวข้อ</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="pipeline">
            <span class="pipeline-step">Input</span><span class="pipeline-arrow">→</span>
            <span class="pipeline-step">Cleansing</span><span class="pipeline-arrow">→</span>
            <span class="pipeline-step">Normalization</span><span class="pipeline-arrow">→</span>
            <span class="pipeline-step">Tokenization</span><span class="pipeline-arrow">→</span>
            <span class="pipeline-step">Topic</span><span class="pipeline-arrow">→</span>
            <span class="pipeline-step">NER</span><span class="pipeline-arrow">→</span>
            <span class="pipeline-step">Priority</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    explanation_columns = st.columns(2)
    explanations = [
        (
            "Regex & Cleansing",
            "ตรวจ IPv4 ที่มี octet อยู่ในช่วง 0–255, phone, email, URL, time และ date "
            "จากนั้น mask PHONE, EMAIL และ URL โดยเก็บ IP ไว้เพื่อการวิเคราะห์ Network",
        ),
        (
            "Tokenization",
            'Thai/mixed ใช้ PyThaiNLP `word_tokenize(text, engine="newmm", keep_whitespace=False)` ส่วน English ใช้ regex tokenizer',
        ),
        (
            "Normalization",
            "ใช้ Unicode NFC, รวม whitespace และลดการลากตัวอักษร/เครื่องหมายซ้ำ โดยปกป้อง technical strings",
        ),
        (
            "Topic Identification",
            "ใช้ domain keyword scoring เพื่อจำแนก Network Connectivity, Wi-Fi, DNS, DHCP, Authentication, Hardware, Application และ Other",
        ),
        (
            "Rule-based NER",
            "ดึง Vendor, Device, IP, Phone, Email, URL, Location, Time และ Date จาก regex และ domain dictionaries",
        ),
        (
            "Priority Detection",
            "ตรวจ evidence ตามระดับ CRITICAL, HIGH, MEDIUM และ LOW ด้วยกฎที่ deterministic และอธิบายได้",
        ),
    ]
    for index, (title, description) in enumerate(explanations):
        with explanation_columns[index % 2].container(border=True):
            st.markdown(f"#### {title}")
            st.write(description)


tab_analyze, tab_dataset, tab_explain = st.tabs(
    ["🔎 วิเคราะห์ข้อความ", "📁 ทดสอบ Dataset", "🧠 NLP ที่ใช้"]
)
with tab_analyze:
    render_analysis_tab()
with tab_dataset:
    render_dataset_tab()
with tab_explain:
    render_explanation_tab()

st.markdown(
    """
    <div class="footer">
        Network &amp; IT Support Ticket Analyzer · Natural Language Processing Project
    </div>
    """,
    unsafe_allow_html=True,
)
