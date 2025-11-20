# type: ignore
#el archivo original está en otra carpeta este contiene lo justo y necesario para deployarlo en Streamlit
# main_model.py
import os
import re
import numpy as np
import pandas as pd
#import emoji
from typing import List

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer


# 1) LIMPIEZA DE TEXTO

def limpiar_texto(texto: str) -> str:
    t = texto.lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"(@\w+)|(#\w+)", " ", t)
    t = emoji.replace_emoji(t, replace="")
    t = re.sub(r"[^a-záéíóúñü'!? ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# 2) EMBEDDINGS

try:
    from openai import OpenAI
    client = OpenAI()
    use_openai = True
except:
    client = None
    use_openai = False


def embed_openai(texts: List[str]):
    r = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([d.embedding for d in r.data], dtype=np.float32)


_local_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")


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


# 3) ENTRENAR O CARGAR MODELO

CSV_PATH = "CyberBullying.csv"

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

# Modelo simple
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(Xtr_emb, y_train)

# Evaluación
roc = roc_auc_score(y_test, clf.predict_proba(Xte_emb)[:, 1])
print("Modelo entrenado. ROC-AUC:", roc)

# 4) FUNCIÓN PARA PREDICCIONES

def evaluar_frase(frase: str):
    t = limpiar_texto(frase)
    emb = get_embeddings([t])
    proba = clf.predict_proba(emb)[0][1]
    pred = int(proba >= 0.5)
    return pred, proba







