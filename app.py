import streamlit as st
from main_model import evaluar_frase

st.set_page_config(page_title="Detector de Ciberbullying")

st.title("Detector de Ciberbullying")
st.write("Evaluá frases y detectá si contienen **ciberbullying** usando embeddings y Machine Learning.")

user_text = st.text_area("Escribí una frase para analizar:", height=150)

if st.button("Evaluar"):
    if not user_text.strip():
        st.warning("Por favor escribí una frase.")
    else:
        pred, proba = evaluar_frase(user_text)

        if pred == 1:
            st.error(f"**CIBERBULLYING**\nProbabilidad: {proba:.3f}")
        else:
            st.success(f"**No es ciberbullying**\nProbabilidad: {proba:.3f}")
