import streamlit as st
import pandas as pd
import re
from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import normalize

class TicketAnalyzer:
    def __init__(self):
        self.stopwords = list(thai_stopwords())
        
    def analyze(self, text):
        # 1. Clean
        cleaned = re.sub(r'\d{2,3}-\d{3,4}-\d{4}', '[PHONE]', text)
        cleaned = re.sub(r'0\d{9}', '[PHONE]', cleaned)
        cleaned = normalize(cleaned)
        
        # 2. Tokenize
        tokens = [t for t in word_tokenize(cleaned) if t not in self.stopwords and not t.isspace()]
        
        # 3. Entities
        entities = {}
        ip = re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
        if ip: entities['IP_ADDRESS'] = ip.group()
        loc = re.search(r'(ห้อง|แผนก|ชั้น)\s?(\w+|\d+)', text)
        if loc: entities['LOCATION'] = loc.group()
        
        # 4. Topic
        t_low = text.lower()
        if any(w in t_low for w in ['เน็ตล่ม', 'down', 'ping']): topic = "Network Connectivity"
        elif 'wifi' in t_low: topic = "Wi-Fi"
        else: topic = "General IT"
        
        return {"cleaned": cleaned, "tokens": tokens, "entities": entities, "topic": topic}

st.set_page_config(page_title="IT Support Analyzer")
st.title("🔍 IT Support Ticket Analyzer")

analyzer = TicketAnalyzer()
input_text = st.text_area("กรอกข้อความแจ้งปัญหา:", "เน็ตห้อง 402 ล่มครับ IP 192.168.1.50")

if st.button("วิเคราะห์"):
    res = analyzer.analyze(input_text)
    st.success(f"Topic: {res['topic']}")
    st.write("**Entities:**", res['entities'])
    st.write("**Tokens:**", res['tokens'])
