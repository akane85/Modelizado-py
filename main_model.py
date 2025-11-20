# type: ignore
#el archivo original está en otra carpeta este contiene lo justo y necesario para deployarlo en Streamlit


# main_model.py (versión compatible con Streamlit)
import os
import pandas as pd
import numpy as np
import re
import emoji
from typing import List

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from openai import OpenAI
from sentence_transformers import SentenceTransformer

print(">>> Cargando main_model.py...")

# -------------------------------------------------------
# 1) LIMPIEZA DE TEXTO
# -------------------------------------------------------

def limpiar_texto(texto: str) -> str:
    t = texto.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"(@\w+)|(#\w+)", " ", t)
    t = emoji.replace_emoji(t, replace="")
    t = re.sub(r"[^a-záéíóúñü'!? ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# -------------------------------------------------------
# 2) EMBEDDINGS (OpenAI si existe API key, sino local)
# -------------------------------------------------------

# Intentamos inicializar OpenAI
try:
    client = OpenAI()
    use_openai = True
    print(">>> Usando embeddings de OpenAI")
except:
    client = None
    use_openai = False
    print(">>> API de OpenAI no disponible, usando modelo local")

# Modelo local
_local_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

def embed_openai(texts: List[str]):
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([d.embedding for d in r.data], dtype=np.float32)

def embed_local(texts: List[str]):
    return _local_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def get_embeddings(texts: List[str]):
    if use_openai:
        return embed_openai(texts)
    return embed_local(texts)

# -------------------------------------------------------
# 3) CARGAR CSV Y ENTRENAR MODELO
# -------------------------------------------------------

# Ruta robusta para Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "CyberBullying.csv")

print(">>> Leyendo CSV desde:", CSV_PATH)

df = pd.read_csv(CSV_PATH)
df["clean"] = df["Text"].astype(str).apply(limpiar_texto)

X = df["clean"].tolist()
y = df["CB_Label"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Embeddings
Xtr_emb = get_embeddings(X_train)
Xte_emb = get_embeddings(X_test)

# Modelo
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xtr_emb, y_train)

roc = roc_auc_score(y_test, clf.predict_proba(Xte_emb)[:, 1])
print(">>> Modelo entrenado. ROC-AUC:", roc)

# -------------------------------------------------------
# 4) FUNCIÓN PÚBLICA PARA APP.PY
# -------------------------------------------------------

def evaluar_frase(frase: str):
    t = limpiar_texto(frase)
    emb = get_embeddings([t])
    proba = clf.predict_proba(emb)[0][1]
    pred = int(proba >= 0.5)
    return pred, proba

