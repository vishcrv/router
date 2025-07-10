import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="LLM Router", layout="centered")
st.title("🔀 LLM Router – Prompt Router and Responder")

prompt = st.text_area("Enter your prompt:", height=200)

if st.button("Route and Get Response"):
    with st.spinner("Routing and querying..."):
        try:
            response = requests.post("http://localhost:8000/query", json={"prompt": prompt})
            if response.status_code == 200:
                data = response.json()
                
                # Show routed model and meta
                st.success(f"🔧 Routed to: **{data['selected_model']}**")
                st.markdown(f"**Confidence**: `{data['selection_confidence']:.2f}`")
                st.markdown(f"**Reasoning**: {data['selection_reasoning']}")
                
                # Display final response
                st.markdown("### 📤 Response")
                st.write(data["response"])

                # Bar chart of all scores
                st.markdown("### 📊 Model Scores")
                scores = data.get("all_model_scores", {})
                if scores:
                    df = pd.DataFrame(list(scores.items()), columns=["Model", "Score"]).sort_values(by="Score", ascending=False)
                    st.bar_chart(df.set_index("Model"))
                else:
                    st.info("No model scores available.")
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the backend. Is FastAPI running on port 8000?")

