
import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Ibadan Sentiment Tracker", layout="wide", page_icon="🇳🇬")

st.title("🇳🇬 Ibadan Kidnap Rescue - Sentiment Tracker")
st.caption("By Shakespeare Nwodo | Model: felixshakespeare/ibadan_sentiment_model")

@st.cache_resource
def load_custom_model():
    try:
        from transformers import pipeline
        model_id = "felixshakespeare/ibadan_sentiment_model"
        try:
            pipe = pipeline("sentiment-analysis", model=model_id, tokenizer="distilbert-base-uncased")
            return pipe, f"✅ Loaded YOUR custom model: {model_id}"
        except Exception as e1:
            try:
                pipe = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)
                return pipe, f"✅ Loaded custom model (full): {model_id}"
            except Exception as e2:
                pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                return pipe, f"⚠️ Fallback. Custom error: {str(e2)[:100]}"
    except Exception as e:
        return None, f"❌ Error: {str(e)[:200]}"

pipe, status_msg = load_custom_model()

st.sidebar.header("📊 Model Status")
st.sidebar.info(status_msg)
st.sidebar.divider()
st.sidebar.write("Dashboard: Relief, Neutral, Anger, Misinformation")

# DEBUG: Show what files Streamlit sees
st.sidebar.subheader("🔍 Debug: Files on GitHub")
try:
    files = os.listdir(".")
    st.sidebar.write(files)
except Exception as e:
    st.sidebar.write(f"List error: {e}")

st.header("🔍 Live Prediction")
text = st.text_area("Paste tweet / comment:", height=120, placeholder="Thank God the children were rescued safely...")

if st.button("Analyze Sentiment", type="primary"):
    if not text.strip():
        st.warning("Please type something")
    else:
        if pipe is None:
            st.error(f"Model not loaded: {status_msg}")
        else:
            try:
                result = pipe(text)[0]
                label = result['label']
                score = result['score']
                st.write(f"**Raw:** {label} ({score:.2%})")
                ll = label.lower()
                if "pos" in ll or "relief" in ll or "label_2" in ll:
                    st.success(f"**{label}** - Relief / Positive ({score:.2%})")
                    st.balloons()
                elif "neg" in ll or "anger" in ll or "label_0" in ll:
                    st.error(f"**{label}** - Anger / Negative ({score:.2%})")
                else:
                    st.info(f"**{label}** - Neutral ({score:.2%})")
                st.metric("Confidence", f"{score:.2%}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

st.divider()
st.header("📈 Dashboard - Real Data")

df_real = None
excel_path = None
# Try multiple possible paths
candidates = ["ibadan_final_training2.xls", "ibadan_final_training2.xlsx", "./ibadan_final_training2.xls"]
for cand in candidates:
    if os.path.exists(cand):
        excel_path = cand
        break

if excel_path:
    st.success(f"Found Excel: {excel_path} ({os.path.getsize(excel_path)/1024:.1f} KB)")
    try:
        # Try with xlrd for .xls
        if excel_path.endswith(".xls"):
            try:
                df_real = pd.read_excel(excel_path, engine="xlrd")
            except:
                df_real = pd.read_excel(excel_path)
        else:
            df_real = pd.read_excel(excel_path, engine="openpyxl")
        st.dataframe(df_real.head(20), use_container_width=True)
        # Find sentiment column
        sent_col = None
        for col in df_real.columns:
            if any(k in col.lower() for k in ["sentiment","label","emotion"]):
                sent_col = col
                break
        if sent_col:
            counts = df_real[sent_col].value_counts().reset_index()
            counts.columns = ["Sentiment","Count"]
            fig1 = px.pie(counts, values="Count", names="Sentiment", title=f"Real Distribution from {sent_col}")
            st.plotly_chart(fig1, use_container_width=True)
            fig2 = px.bar(counts, x="Sentiment", y="Count", color="Sentiment", title="Count by Sentiment")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning(f"No sentiment column found. Columns: {list(df_real.columns)}")
            st.write(df_real.columns.tolist())
    except Exception as e:
        st.error(f"Error reading Excel {excel_path}: {e}")
        st.info("Trying to install xlrd? Check requirements.txt has xlrd")
else:
    st.warning("Excel not found at root. Checked: " + ", ".join(candidates))
    st.info("Your repo has it, but Streamlit path is different. Rebooting after deleting big folder usually fixes.")

st.caption("Built by Shakespeare Nwodo | felixshakespeare/ibadan_sentiment_model")
