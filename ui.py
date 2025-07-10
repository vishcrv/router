import streamlit as st
import requests

st.set_page_config(page_title="LLM Router UI", layout="centered")

st.title("🔀 LLM Router")

prompt = st.text_area("Enter your prompt:", height=200)

if st.button("Route and Get Response"):
    with st.spinner("Routing and querying..."):
        response = requests.post("http://localhost:8000/query", json={"prompt": prompt})
        if response.status_code == 200:
            data = response.json()
            st.success(f"🔧 Routed to: {data['selected_model']}")
            st.markdown("### 📤 Response")
            st.write(data["response"])
        else:
            st.error("Error: " + response.text)

