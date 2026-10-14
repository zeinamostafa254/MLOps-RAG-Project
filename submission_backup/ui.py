import requests
import streamlit as st

API = "http://localhost:8000"

st.title("Egyptian Legal RAG")
question = st.text_area("Question (Arabic or English)", height=100)

if st.button("Ask") and question.strip():
    with st.spinner("Searching..."):
        r = requests.post(f"{API}/ask", json={"question": question}, timeout=180)
    if r.status_code == 200:
        data = r.json()
        st.markdown(data["answer"])
        st.subheader("Sources")
        for s in data["sources"]:
            st.write("•", s)
    else:
        st.error(f"{r.status_code}: {r.text}")