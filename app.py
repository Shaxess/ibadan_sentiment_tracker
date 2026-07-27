import streamlit as st

st.set_page_config(page_title="Ibadan Tracker")

st.title("🇳🇬 Ibadan Sentiment Tracker")
st.success("Streamlit is working!")

st.header("Test Sentiment")
text = st.text_area("Type a tweet:")

if st.button("Analyze"):
    if "rescue" in text.lower():
        st.balloons()
        st.write("Prediction: **Relief / Positive**")
    else:
        st.write("Prediction: **Neutral**")

st.info("By Felix Shakespeare - Torch will be added after VC++ install")