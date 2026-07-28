
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Ibadan Sentiment Tracker", layout="wide", page_icon="🇳🇬")

st.title("🇳🇬 Ibadan Kidnap Rescue - Sentiment Tracker")
st.caption("By Shakespeare Nwodo | Model: felixshakespeare/ibadan_sentiment_model")

# Load model WITHOUT direct torch import at top
@st.cache_resource
def load_custom_model():
    try:
        from transformers import pipeline
        model_id = "felixshakespeare/ibadan_sentiment_model"
        # Try custom model with fallback tokenizer
        try:
            pipe = pipeline("sentiment-analysis", model=model_id, tokenizer="distilbert-base-uncased")
            return pipe, f"✅ Loaded YOUR custom model: {model_id}"
        except Exception as e1:
            # Try with same model as tokenizer (if you uploaded tokenizer)
            try:
                pipe = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)
                return pipe, f"✅ Loaded custom model (full): {model_id}"
            except Exception as e2:
                # Fallback to base model
                pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
                return pipe, f"⚠️ Custom model failed, using fallback. Error: {str(e2)[:150]}"
    except Exception as e:
        return None, f"❌ Error loading: {str(e)[:200]}"

pipe, status_msg = load_custom_model()

st.sidebar.header("📊 Model Status")
st.sidebar.info(status_msg)
st.sidebar.divider()
st.sidebar.write("Dashboard: Relief, Neutral, Anger, Misinformation")

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
                
                st.write(f"**Raw model output:** {label} ({score:.2%})")
                
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
st.header("📈 Dashboard")

df_real = None
for fname in ["ibadan_final_training2.xls", "ibadan_final_training2.xlsx"]:
    try:
        df_real = pd.read_excel(fname)
        break
    except:
        continue

if df_real is not None:
    st.success(f"Loaded real data: {len(df_real)} rows")
    st.dataframe(df_real.head(10), use_container_width=True)
    # Try find sentiment column
    for col in df_real.columns:
        if "sentiment" in col.lower() or "label" in col.lower():
            counts = df_real[col].value_counts().reset_index()
            counts.columns = ["Sentiment","Count"]
            fig = px.pie(counts, values="Count", names="Sentiment", title="Real Distribution")
            st.plotly_chart(fig, use_container_width=True)
            break
else:
    st.info("Upload ibadan_final_training2.xls to GitHub to see real data chart here")
    sample = pd.DataFrame({"Sentiment":["Relief","Neutral","Anger","Misinformation"],"Count":[45,25,20,10]})
    fig = px.bar(sample, x="Sentiment", y="Count", color="Sentiment", title="Sample (upload Excel to replace)")
    st.plotly_chart(fig, use_container_width=True)

st.caption("Built by Shakespeare Nwodo | felixshakespeare/ibadan_sentiment_model | Streamlit Cloud")
