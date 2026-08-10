# Network & IT Support Ticket Analyzer

เว็บแอปสำหรับวิเคราะห์ข้อความแจ้งปัญหา IT/Network ภาษาไทย ภาษาอังกฤษ และข้อความผสม
พัฒนาด้วย Python, Streamlit และ PyThaiNLP เพื่อใช้เป็นโครงงานวิชา NLP ระดับมหาวิทยาลัย

## 1. แนวคิดและปัญหาที่แก้ไข

ทีม Help Desk ได้รับข้อความที่มีทั้งคำบรรยายปัญหา ข้อมูลเครือข่าย และข้อมูลติดต่ออยู่ในประโยคเดียวกัน
เช่น ผู้ใช้แจ้งว่า Wi-Fi ใช้งานไม่ได้ พร้อมระบุห้อง, IP address และเวลาที่เริ่มเกิดปัญหา
แอปนี้ช่วยจัดหมวดหมู่ ticket, ประเมินความเร่งด่วน และดึงข้อมูลสำคัญให้อ่านได้เร็วขึ้น

ระบบนี้เป็น **Educational Rule-based NLP system** ไม่ใช่ระบบ Production-ready
การนำไปใช้จริงควรมี dataset ที่กว้างขึ้น การตรวจสอบโดยผู้เชี่ยวชาญ และ/หรือโมเดล Machine Learning

## 2. ความสามารถหลัก

- ตรวจภาษา Thai, English, Mixed Thai-English และ Unknown
- ทำ Unicode normalization, รวมช่องว่าง และลดการลากตัวอักษร/เครื่องหมายซ้ำ
- ใช้ Regex หา IPv4, เบอร์โทร, email, URL, เวลา และวันที่
- Mask ข้อมูลติดต่อเป็น `[PHONE]`, `[EMAIL]`, `[URL]` แต่เก็บ IP address ไว้เพราะสำคัญต่อการวิเคราะห์ Network
- Tokenization ด้วย `pythainlp.tokenize.word_tokenize` และ `engine="newmm"`
- ตัด Thai stopwords จาก PyThaiNLP และ English stopwords ชุดเล็กที่อ่านทำความเข้าใจได้
- จำแนก Topic: Network Connectivity, Wi-Fi, DNS, DHCP, Authentication, Hardware, Application และ Other
- Rule-based NER สำหรับ IP_ADDRESS, PHONE, EMAIL, URL, TIME, DATE, VENDOR, DEVICE และ LOCATION
- ตรวจ Priority: CRITICAL, HIGH, MEDIUM และ LOW พร้อมแสดง keyword evidence
- ทดสอบ dataset ในหน้าเว็บและดาวน์โหลด `analysis_results.csv`

## 3. เทคนิค NLP ที่ใช้

### Regex & Cleansing

ไฟล์ `nlp_utils.py` มี regular expressions สำหรับข้อมูลที่มีรูปแบบชัดเจน โดย IPv4 ใช้ octet ที่ตรวจช่วง
0–255 เช่น `192.168.1.1` ส่วน phone, email และ URL ถูก mask ใน `cleaned_text`
โดยไม่แก้ `original_text` และไม่ลบ IP address

### Tokenization & Normalization

ข้อความ Thai และ mixed ใช้ PyThaiNLP ตามคำสั่งต่อไปนี้:

```python
from pythainlp.tokenize import word_tokenize
tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
```

English ใช้ regex tokenizer ขนาดเล็ก ส่วน normalization ใช้ Unicode NFC, whitespace collapsing,
การลดตัวอักษรซ้ำ และการลด `!!!` เป็น `!` โดยปกป้อง IP, email และ URL ไม่ให้เสียรูป

### Topic Identification

ใช้ dictionary ของ keyword ที่มี weight เช่น `DNS`, `domain` และ `nslookup` มีน้ำหนักสูงในหัวข้อ DNS
ผลลัพธ์จะแสดง `topic_scores` และ `topic_keywords` เพื่อให้ตรวจสอบเหตุผลได้ ไม่ใช่การทำนายแบบกล่องดำ

### NER

ใช้ regex สำหรับ structured entities และใช้ domain dictionary สำหรับ vendor เช่น Cisco, Juniper,
Fortinet และ device เช่น Router, Switch, Firewall รวมทั้ง pattern ของ `ห้อง Lab 402`, `Room 210`,
`Building A` และ `Floor 4` จุดเด่นของโครงงานคือการออกแบบ NER ให้เหมาะกับงาน Network Support
โดยไม่ต้องดาวน์โหลด Transformer model

### Priority Detection

กฎจะตรวจคำระดับ CRITICAL ก่อน HIGH และ MEDIUM เช่น `ทั้งระบบ`, `affects all users`, `urgent`,
`ใช้งานไม่ได้`, `slow` และ `packet loss` หากไม่พบคำใดจะเป็น LOW พร้อมส่ง matched keywords กลับมา

โครงงานเลือกใช้ NER แทน POS tagging เพราะ NER ตรงกับโจทย์การดึงข้อมูล IT Support มากกว่า และทำให้
deployment บน Streamlit Community Cloud เสถียรโดยไม่ต้องใช้โมเดลหรือ dependency ขนาดใหญ่

## 4. โครงสร้างโครงการ

```text
.
├── app.py
├── nlp_utils.py
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── .gitignore
├── colab_demo.ipynb
├── data/
│   └── test_data.csv
└── tests/
    └── test_nlp_utils.py
```

## 5. ติดตั้งและรันในเครื่อง

ต้องใช้ Python 3.10 ขึ้นไป แนะนำให้สร้าง virtual environment:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

เปิด URL ที่ Streamlit แสดงใน terminal แล้วเลือก tab **วิเคราะห์ข้อความ**

สำหรับการรัน test ให้ติดตั้ง development dependencies:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## 6. การใช้บน Google Colab

เปิดไฟล์ `colab_demo.ipynb` ใน Google Colab แล้ว Run all ได้เลย Notebook จะติดตั้ง dependency
และดาวน์โหลดไฟล์จาก repository เมื่อไม่มีไฟล์ใน working directory จึงไม่ผูกกับ path ในเครื่อง Windows
ถ้า repository ถูก fork ให้แก้ตัวแปร `REPO_RAW_BASE` ใน notebook ให้เป็น URL ของ fork นั้น

Notebook จะแสดงผล sample message, วิเคราะห์ทุกแถวใน dataset, คำนวณ accuracy, แสดง prediction errors
และ export เป็น `analysis_results.csv`

## 7. Dataset

`data/test_data.csv` มี 50 ข้อความที่ผสม Thai, English และ mixed language ครอบคลุม 7 topic หลักและ Other
มีคอลัมน์ `id`, `text`, `expected_topic` และ `expected_priority` พร้อมตัวอย่าง IP, vendor, device,
location, time, phone, email และ URL สำหรับทดสอบ Regex และ NER

ใน Streamlit tab **ทดสอบ Dataset** สามารถใช้ dataset ที่ bundled หรือ upload CSV ใหม่ได้ โดยไฟล์ใหม่ต้องมี
คอลัมน์ `text`; `expected_topic` และ `expected_priority` เป็นคอลัมน์เสริมสำหรับคำนวณ accuracy

## 8. ตัวอย่าง Input / Output

Input:

```text
เน็ตห้อง Lab 402 ใช้งานไม่ได้ตั้งแต่ 09:30 น. Cisco Router IP 192.168.1.1 ping ไม่เจอ ด่วนมากครับ
ติดต่อ 081-234-5678
```

ผลลัพธ์โดยประมาณ:

```text
Language: Mixed Thai-English
Topic: Network Connectivity
Priority: HIGH
Entities: Cisco/VENDOR, Router/DEVICE, ห้อง Lab 402/LOCATION,
          192.168.1.1/IP_ADDRESS, 09:30/TIME, 081-234-5678/PHONE
Cleaned text: ... ติดต่อ [PHONE]
```

## 9. GitHub

ตรวจสอบไฟล์และ commit เฉพาะไฟล์ของโครงงาน:

```bash
git status
git add app.py nlp_utils.py requirements.txt requirements-dev.txt README.md .gitignore data tests colab_demo.ipynb
git commit -m "Complete NLP Streamlit app and fix deployment dependencies"
git push
```

ห้าม commit password, API key, token หรือไฟล์ secrets

## 10. Streamlit Community Cloud

1. เปิด Streamlit Community Cloud และเลือก **Create app**
2. เลือก GitHub repository และ branch ที่ push โค้ดแล้ว
3. ตั้งค่า Main file path เป็น `app.py`
4. กด Deploy

`requirements.txt` อยู่ที่ repository root และระบุ `pythainlp` แล้ว จึงแก้ปัญหา dependency discovery
ที่เคยทำให้เกิด `ModuleNotFoundError` ได้ โครงการนี้ไม่ต้องใช้ secret, API key, database หรือ model download

## 11. ตัวอย่าง AI Prompt ที่ใช้ระหว่างพัฒนา

> ช่วยพัฒนา Web Application ด้วย Python และ Streamlit สำหรับวิเคราะห์ข้อความแจ้งปัญหา Network/IT Support
> ภาษาไทย ภาษาอังกฤษ และข้อความผสม โดยใช้ Regex & Cleansing, PyThaiNLP Tokenization, Normalization,
> Topic Identification และ Rule-based NER พร้อม Dataset, Unit Tests และหน้าเว็บที่อธิบายหลักการ NLP
> อย่างชัดเจน โดยไม่ใช้ paid API, GPU หรือโมเดลขนาดใหญ่

## 12. ข้อจำกัด

- Topic และ Priority ขึ้นกับ keyword ที่กำหนด จึงอาจพลาดคำพ้องหรือข้อความที่กำกวม
- Location และ vendor/device รองรับ pattern ตามตัวอย่างของโครงงาน ไม่ใช่ NER ทั่วไปทุกบริบท
- ความแม่นยำของผลทดสอบจาก dataset ขนาดเล็กไม่ควรถูกตีความเป็นประสิทธิภาพ Production
- การใช้งานจริงควรเพิ่มข้อมูลหลายรูปแบบ, human review, monitoring และอาจพัฒนาเป็น ML model ในอนาคต
