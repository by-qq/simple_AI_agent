import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8002"

st.set_page_config(page_title="Enterprise KB Assistant", layout="wide")

# ---- deep blue modern theme ----
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px circle at 10% 0%, #0b2a5b 0%, #06162f 45%, #040b19 100%);
        color: #e6eefc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .block-container { padding-top: 2rem; }
    h1, h2, h3, h4 { color: #e6eefc; }
    .card {
        background: rgba(8, 22, 48, 0.85);
        border: 1px solid rgba(120,160,220,0.25);
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }
    .small { color: #b9c8e6; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>🏢 Enterprise Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown('<div class="small">Upload policies / ask questions / see citations.</div>', unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns([0.38, 0.62], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📎 文档上传入库")
    uploaded = st.file_uploader("选择 Word/PDF/MD/TXT", type=["pdf","docx","doc","md","txt"])
    visibility = st.selectbox("可见性", ["public", "hr", "finance", "it"], index=0)
    doc_id = st.text_input("doc_id（可选）", "")
    if st.button("上传并入库", use_container_width=True, type="primary") and uploaded:
        files = {"file": (uploaded.name, uploaded.getvalue())}
        data = {"visibility": visibility}
        if doc_id.strip():
            data["doc_id"] = doc_id.strip()
        r = requests.post(f"{API_BASE}/ingest", files=files, data=data, timeout=120)
        if r.ok:
            st.success(f"入库成功：chunks={r.json().get('chunks')}")
        else:
            st.error(r.text)
    st.write("")
    if st.button("全量重建（清库+重建）", use_container_width=True):
        r = requests.post(f"{API_BASE}/reindex", data={"visibility_default": "public"}, timeout=300)
        if r.ok:
            st.success(f"重建完成：chunks={r.json().get('chunks')}")
        else:
            st.error(r.text)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 企业知识问答")
    q = st.text_area("输入你的问题", height=120, placeholder="例如：年假需要提前多久申请？")
    role = st.selectbox("你的角色", ["public", "hr", "finance", "it_admin", "manager"], index=0)
    if st.button("发送", use_container_width=True) and q.strip():
        payload = {"text": q.strip(), "user_role": role, "requester": "streamlit"}
        r = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
        if r.ok:
            st.markdown("**回答：**")
            st.write(r.json()["answer"])
        else:
            st.error(r.text)
    st.markdown('</div>', unsafe_allow_html=True)

