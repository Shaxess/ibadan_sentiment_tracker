
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="Ibadan Sentiment Tracker", layout="wide", page_icon="🇳🇬")

st.title("🇳🇬 Ibadan Kidnap Rescue - Sentiment Tracker")
st.caption("By Shakespeare Nwodo | Fine-tuned DistilBERT Model: felixshakespeare/ibadan_sentiment_model")

# Load your custom model from HuggingFace
@st.cache_resource
def load_custom_model():
    model_id = "felixshakespeare/ibadan_sentiment_model"
    try:
        # Try loading your fine-tuned model
        pipe = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)
        return pipe, f"Loaded custom model: {model_id}"
    except Exception as e:
        # Fallback to base model if custom fails
        try:
            pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            return pipe, f"Custom model failed ({str(e)[:100]}), using fallback"
        except Exception as e2:
            return None, f"Error: {e2}"

pipe, status_msg = load_custom_model()

st.sidebar.header("📊 Model Status")
st.sidebar.info(status_msg)
st.sidebar.divider()
st.sidebar.header("About Project")
st.sidebar.write("This dashboard analyzes public sentiment on Ibadan rescue: Relief, Neutral, Anger, Misinformation.")
st.sidebar.write("Model: DistilBERT fine-tuned on Ibadan dataset")

st.header("🔍 Live Prediction - Your Fine-tuned Model")
text = st.text_area("Paste a tweet / Facebook comment about Ibadan rescue:", height=120, 
                    placeholder="e.g. Thank God the children were rescued safely by the police...")

if st.button("Analyze Sentiment", type="primary"):
    if not text.strip():
        st.warning("Please type something")
    else:
        if pipe is None:
            st.error("Model not loaded. Check HuggingFace model.")
        else:
            result = pipe(text)[0]
            label = result['label']
            score = result['score']
            
            # Map your labels if you trained with custom labels
            # Adjust based on your training labels
            lower_label = label.lower()
            if "relief" in lower_label or "positive" in lower_label or "label_2" in lower_label or "pos" in lower_label:
                st.success(f"**{label}** - Relief / Positive ({score:.2%})")
                st.balloons()
                st.write("Public reaction: Gratitude, happiness about rescue")
            elif "anger" in lower_label or "negative" in lower_label or "label_0" in lower_label:
                st.error(f"**{label}** - Anger / Negative ({score:.2%})")
                st.write("Public reaction: Criticism of security failures")
            elif "misinfo" in lower_label or "skeptic" in lower_label:
                st.warning(f"**{label}** - Misinformation / Skepticism ({score:.2%})")
            else:
                st.info(f"**{label}** - Neutral ({score:.2%})")
            
            st.metric("Confidence", f"{score:.2%}")

st.divider()
st.header("📈 Sentiment Dashboard")

# Try to load real data
df_real = None
for fname in ["ibadan_final_training2.xls", "ibadan_final_training2.xlsx", "ibadan_final_training2.xls.xlsx"]:
    try:
        df_real = pd.read_excel(fname)
        break
    except:
        continue

if df_real is not None:
    st.subheader("Your Training Data")
    st.dataframe(df_real.head(10), use_container_width=True)
    
    # If has sentiment column, plot
    sentiment_col = None
    for col in df_real.columns:
        if "sentiment" in col.lower() or "label" in col.lower():
            sentiment_col = col
            break
    
    if sentiment_col:
        counts = df_real[sentiment_col].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(counts, values="Count", names="Sentiment", title="Real Data Distribution")
        st.plotly_chart(fig, use_container_width=True)
else:
    # Sample
    st.info("Upload ibadan_final_training2.xls to GitHub repo to see real data here (small file, allowed)")
    sample = pd.DataFrame({"Sentiment": ["Relief","Neutral","Anger","Misinformation"], "Count": [45,25,20,10]})
    fig = px.bar(sample, x="Sentiment", y="Count", color="Sentiment", title="Sample Distribution (replace with real data)")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built by Shakespeare Nwodo | Model: felixshakespeare/ibadan_sentiment_model on HuggingFace | Deployed on Streamlit Cloud")
