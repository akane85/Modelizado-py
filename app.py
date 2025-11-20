import streamlit as st
import numpy as np

from main_model import limpiar_texto, get_embeddings, clf, mlp, best_thr

# --- CONFIG DE LA APP ---
st.set_page_config(page_title="Detector de Cyberbullying")

st.title("Detector de Cyberbullying")
st.write("Ingresá una frase en inglés para evaluar si contiene ciberbullying.")

# Campo de texto
frase = st.text_area("Texto a analizar", height=120)

modelo = st.selectbox(
    "Modelo a usar:",
    ["Logistic Regression", "MLP (Neural Network)"]
)

if st.button("Evaluar frase"):
    if not frase.strip():
        st.warning("Por favor escribí una frase.")
    else:
        # limpiar texto
        texto_limpio = limpiar_texto(frase)

        # convertir a embeddings
        emb = get_embeddings([texto_limpio])

        # seleccionar modelo
        if modelo == "Logistic Regression":
            proba = clf.predict_proba(emb)[:, 1][0]
            pred = int(proba >= best_thr)
        else:
            proba = float(mlp.predict(emb).ravel()[0])
            pred = int(proba >= 0.5)

        st.write("---")
        st.subheader("Resultado")

        if pred == 1:
            st.error(f"**CIBERBULLYING**\n\n**Probabilidad:** `{proba:.3f}`")
        else:
            st.success(f"**No es ciberbullying**\n\n**Probabilidad:** `{proba:.3f}`")


