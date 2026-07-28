
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
                return pipe, f"✅ Loaded custom model: {model_id}"
            except Exception as e2:
                pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                return pipe, f"⚠️ Using fallback. Custom error: {str(e2)[:120]}"
    except Exception as e:
        return None, f"❌ Error: {str(e)[:200]}"

pipe, status_msg = load_custom_model()
st.sidebar.info(status_msg)

st.header("🔍 Live Prediction")
text = st.text_area("Paste tweet / comment:", height=100)
if st.button("Analyze Sentiment", type="primary"):
    if text.strip() and pipe:
        result = pipe(text)[0]
        st.write(f"**{result['label']}** ({result['score']:.2%})")
        st.metric("Confidence", f"{result['score']:.2%}")

st.divider()
st.header("📈 Dashboard - Real Data")

def robust_read_excel(path):
    errors = []
    # Try 1: xlrd for old .xls
    try:
        return pd.read_excel(path, engine="xlrd"), "xlrd"
    except Exception as e:
        errors.append(f"xlrd: {e}")
    # Try 2: openpyxl for xlsx
    try:
        return pd.read_excel(path, engine="openpyxl"), "openpyxl"
    except Exception as e:
        errors.append(f"openpyxl: {e}")
    # Try 3: no engine
    try:
        return pd.read_excel(path), "auto"
    except Exception as e:
        errors.append(f"auto: {e}")
    # Try 4: read as csv (maybe csv renamed to xls)
    try:
        return pd.read_csv(path), "csv"
    except Exception as e:
        errors.append(f"csv: {e}")
    # Try 5: read as csv with different encodings
    try:
        return pd.read_csv(path, encoding='latin1'), "csv-latin1"
    except Exception as e:
        errors.append(f"csv-latin1: {e}")
    return None, " | ".join(errors)

excel_path = "ibadan_final_training2.xls"
if os.path.exists(excel_path):
    st.success(f"Found: {excel_path} ({os.path.getsize(excel_path)/1024:.1f} KB)")
    df, info = robust_read_excel(excel_path)
    if df is not None:
        st.success(f"✅ Read successfully with engine: {info}")
        st.dataframe(df.head(30), use_container_width=True)
        st.write(f"Shape: {df.shape} | Columns: {list(df.columns)}")
        # find sentiment col
        sent_col = None
        for c in df.columns:
            if any(k in c.lower() for k in ["sentiment","label","emotion","class"]):
                sent_col = c
                break
        if sent_col:
            counts = df[sent_col].value_counts().reset_index()
            counts.columns = ["Sentiment","Count"]
            fig = px.pie(counts, values="Count", names="Sentiment", title="Real Distribution")
            st.plotly_chart(fig, use_container_width=True)
            fig2 = px.bar(counts, x="Sentiment", y="Count", color="Sentiment")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No sentiment column found, showing first column distribution")
            if len(df.columns) > 0:
                counts = df.iloc[:,0].value_counts().reset_index()
                counts.columns = ["Value","Count"]
                st.bar_chart(counts.set_index("Value"))
    else:
        st.error(f"Failed to read: {info}")
        st.info("Tip: Open the file on your PC in Excel and Save As -> CSV, then upload CSV. Or re-upload original xlsx.")
        # Show raw bytes
        try:
            with open(excel_path, 'rb') as f:
                head = f.read(200)
            st.code(f"First 200 bytes: {head[:200]}")
        except:
            pass
else:
    st.warning(f"File not found: {excel_path}")
    st.write("Files in folder:", os.listdir("."))

st.caption("Built by Shakespeare Nwodo | felixshakespeare/ibadan_sentiment_model")
