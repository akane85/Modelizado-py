# type: ignore
import os
import time
import hashlib
import pathlib
import re
import pandas as pd
import numpy as np
import seaborn as sns
import emoji

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from openai import OpenAI
from getpass import getpass
from typing import List





# %%
# 1) Obtener API key (entorno -> Colab userdata -> getpass)
env_key = os.getenv("OPENAI_API_KEY")
if not env_key:
    try:
        from google.colab import userdata
        env_key = userdata.get("OPENAI_API_KEY")
    except Exception:
        pass
if not env_key:
    try:

        env_key = getpass.getpass("Pegá tu OPENAI_API_KEY (Enter para omitir y usar embeddings locales): ")
    except Exception:
        env_key = None
if env_key:
    os.environ["OPENAI_API_KEY"] = env_key



# %%
# Pedir clave si no está ya cargada
if not os.getenv("OPENAI_API_KEY"):
    print("No hay API key configurada.")
    env_key = getpass("👉 Pegá tu OPENAI_API_KEY aquí: ")
    os.environ["OPENAI_API_KEY"] = env_key
else:
    print("API key detectada.")

from openai import OpenAI
client = OpenAI()
print("Conectado a OpenAI ✔")

# %%
    #Inicializacion y conexion con OpenIA

# 2) Intentar cliente OpenAI
use_openai, client = False, None
if os.getenv("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        client = OpenAI()
        use_openai = True
        print("✅ OpenAI listo. Intentando sanity check…")
    except Exception as e:
        print("⚠️ No pude inicializar OpenAI:", e)
        use_openai = False
else:
    print("ℹ️ No hay OPENAI_API_KEY: usaré embeddings locales (gratuitos).")

# %%
# 3) Funciones de embeddings
#OpenIA usa text-embedding-3-small para generar vectores

def embed_texts_openai(texts: List[str], model="text-embedding-3-small", batch_size=100):
    assert client is not None
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        r = client.embeddings.create(model=model, input=batch)
        all_vecs.extend([d.embedding for d in r.data])
    return np.array(all_vecs, dtype=np.float32)

#  embed_texts_local de SentenceTransformers
#Usa modelo multilingüe local (gratuito)

def embed_texts_local(texts: List[str], model_name="distiluse-base-multilingual-cased-v2"):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name)
    return st.encode(texts, convert_to_numpy=True, batch_size=64,
                     show_progress_bar=True, normalize_embeddings=True)

#Estos embedding convierten texto a vector numerico para que los modelos puedan aprender patrones.

# %%
# 4) Sanity check OpenAI (detecta falta de cuota 429)
if use_openai:
    try:
        _ = embed_texts_openai(["hola mundo"])
        print("✅ OpenAI embeddings funcionando.")
    except Exception as e:
        print("⚠️ OpenAI no disponible:", e)
        print("→ Usaré embeddings locales automáticamente.")
        use_openai = False



# %%
#inicio de pipeline de procesamiento antes de cargar los datos su objetivo es tener una función unificada para generar embeddings
#(vectores de texto) y preparar el entorno de trabajo con Google Drive y las librerías necesarias.


# 5) Función unificada para el pipeline
def get_embeddings(texts: List[str]) -> np.ndarray:
    return embed_texts_openai(texts) if use_openai else embed_texts_local(texts)

print("Listo. Usa get_embeddings(texts) en el flujo.")

#from google.colab import drive
#drive.mount('/content/drive', force_remount=True)




# %%
#6 Ruta y columnas
RUTA_CSV = "C:/Users/akane/Downloads/CyberBullying.csv"  # ← ajusta si cambia
TEXT_COL  = "Text"
LABEL_COL = "CB_Label"

# Cargar
df = pd.read_csv(RUTA_CSV)
assert TEXT_COL in df.columns and LABEL_COL in df.columns, f"Faltan columnas '{TEXT_COL}' y/o '{LABEL_COL}'"

# Limpieza
def limpiar_texto(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"(@\w+)|(#\w+)", " ", t)
    #t = emoji.replace_emoji(t, replace="")
    t = re.sub(r"[^a-záéíóúñü'!? ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["clean_text"] = df[TEXT_COL].apply(limpiar_texto)
df = df.dropna(subset=["clean_text", LABEL_COL]).copy()
df[LABEL_COL] = df[LABEL_COL].astype(int)

# Split (estratificado)
X = df["clean_text"].tolist()
y = df[LABEL_COL].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# %%
#7 Generar embeddings con el backend disponible (OpenAI o local)
Xtr_emb = get_embeddings(X_train)
Xte_emb = get_embeddings(X_test)
print("Embeddings -> Train:", Xtr_emb.shape, " Test:", Xte_emb.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

# %%
#8 Logistic Regression
clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
clf.fit(Xtr_emb, y_train)

y_proba = clf.predict_proba(Xte_emb)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

roc = roc_auc_score(y_test, y_proba)
ap  = average_precision_score(y_test, y_proba)
print("=== LogisticRegression @0.5 ===")
print(classification_report(y_test, y_pred, digits=3))
print("Matriz:\n", confusion_matrix(y_test, y_pred))    
print(f"ROC-AUC: {roc:.3f}  |  PR-AUC: {ap:.3f}")

# %%
# =====9 MLP sobre embeddings =====

mlp = Sequential([
    Dense(256, activation="relu", input_shape=(Xtr_emb.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid"),
])
mlp.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
mlp.fit(Xtr_emb, y_train, validation_split=0.15, epochs=20, batch_size=64,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=1)

y_proba_mlp = mlp.predict(Xte_emb).ravel()
y_pred_mlp  = (y_proba_mlp >= 0.5).astype(int)
roc_mlp = roc_auc_score(y_test, y_proba_mlp)
ap_mlp  = average_precision_score(y_test, y_proba_mlp)

print("\n=== MLP @0.5 ===")
print(classification_report(y_test, y_pred_mlp, digits=3))
print("Matriz:\n", confusion_matrix(y_test, y_pred_mlp))
print(f"ROC-AUC: {roc_mlp:.3f}  |  PR-AUC: {ap_mlp:.3f}")

import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve
# Umbral óptimo por F1 (LR)
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
best_idx = int(np.argmax(f1))
best_thr = float(thresholds[max(best_idx-1, 0)])
print(f"Mejor umbral LR (F1): {best_thr:.3f}")

y_pred_opt = (y_proba >= best_thr).astype(int)
print("\n=== LR @umbral óptimo ===")
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred_opt, digits=3))
print("Matriz:\n", confusion_matrix(y_test, y_pred_opt))

# Curva ROC (LR)
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"LR ROC-AUC={roc:.3f}")
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC"); plt.legend(); plt.grid(True); plt.show()

# Curva PR (LR)
plt.figure()
plt.plot(recall, precision, label=f"LR PR-AUC={ap:.3f}")
plt.scatter(recall[best_idx], precision[best_idx], s=60, label=f"Mejor F1 @ {best_thr:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Curva Precision–Recall")
plt.legend(); plt.grid(True); plt.show()




# %%
print(globals().keys())
'X_test' in globals()
[x for x in globals().keys() if 'test' in x.lower()]


# %%
# 10 CSV de resultados (LR)
out = pd.DataFrame({
    "texto": train_test_split,
    "y_true": y_test,
    "y_proba_LR": y_proba,
    "y_pred_LR@0.50": (y_proba >= 0.5).astype(int),
    f"y_pred_LR@{best_thr:.3f}": (y_proba >= best_thr).astype(int),
    "y_proba_MLP": y_proba_mlp,
    "y_pred_MLP@0.50": y_pred_mlp
})
csv_path = "C:/Users/akane/Downloads/predicciones_cyberbullying.csv"
out.to_csv(csv_path, index=False, encoding="utf-8")
print("✅ CSV guardado en:", csv_path)

# PDF resumen
pdf_path = "C:/Users/akane/Downloads/resumen_ciberbullying.pdf"
rep_lr_050  = classification_report(y_test, (y_proba>=0.5).astype(int), digits=3, output_dict=True)
rep_lr_opt  = classification_report(y_test, (y_proba>=best_thr).astype(int), digits=3, output_dict=True)
rep_mlp_050 = classification_report(y_test, y_pred_mlp, digits=3, output_dict=True)
cm_lr_050   = confusion_matrix(y_test, (y_proba>=0.5).astype(int))
cm_lr_opt   = confusion_matrix(y_test, (y_proba>=best_thr).astype(int))
cm_mlp_050  = confusion_matrix(y_test, y_pred_mlp)

with PdfPages(pdf_path) as pdf:
    # Portada
    fig = plt.figure(figsize=(8.5, 11)); plt.axis('off')
    txt = (
        "Resumen ejecutivo — Evaluación de textos (OpenAI/Locales + LR + MLP)\n\n"
        f"LR → ROC-AUC: {roc:.3f} | PR-AUC: {ap:.3f} | Mejor umbral: {best_thr:.3f}\n"
        f"MLP → ROC-AUC: {roc_mlp:.3f} | PR-AUC: {ap_mlp:.3f}\n\n"
        "Métricas LR @0.50 (clase 1): "
        f"P={rep_lr_050['1']['precision']:.3f}  R={rep_lr_050['1']['recall']:.3f}  F1={rep_lr_050['1']['f1-score']:.3f}\n"
        "Métricas LR @umbral óptimo (clase 1): "
        f"P={rep_lr_opt['1']['precision']:.3f}  R={rep_lr_opt['1']['recall']:.3f}  F1={rep_lr_opt['1']['f1-score']:.3f}\n"
        "Métricas MLP @0.50 (clase 1): "
        f"P={rep_mlp_050['1']['precision']:.3f}  R={rep_mlp_050['1']['recall']:.3f}  F1={rep_mlp_050['1']['f1-score']:.3f}\n"
    )
    plt.text(0.05, 0.95, txt, va='top', ha='left', fontsize=11)
    pdf.savefig(fig); plt.close(fig)

    # Matrices de confusión
    for M, title in [(cm_lr_050, "LR — Matriz @0.50"),
                     (cm_lr_opt, f"LR — Matriz @thr≈{best_thr:.3f}"),
                     (cm_mlp_050, "MLP — Matriz @0.50")]:
        fig = plt.figure(figsize=(6,5))
        plt.imshow(M, cmap='Blues'); plt.title(title); plt.colorbar()
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                plt.text(j, i, M[i, j], ha="center", va="center")
        plt.xlabel("Predicho"); plt.ylabel("Real")
        plt.xticks([0,1],[0,1]); plt.yticks([0,1],[0,1])
        pdf.savefig(fig); plt.close(fig)

    # Curvas (LR)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = plt.figure()
    plt.plot(fpr, tpr, label=f"LR ROC-AUC={roc:.3f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Curva ROC"); plt.legend(); plt.grid(True)
    pdf.savefig(fig); plt.close(fig)

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1))
    fig = plt.figure()
    plt.plot(recall, precision, label=f"LR PR-AUC={ap:.3f}")
    plt.scatter(recall[best_idx], precision[best_idx], s=60, label="Mejor F1")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Curva PR"); plt.legend(); plt.grid(True)
    pdf.savefig(fig); plt.close(fig)

print("✅ PDF generado en:", pdf_path)



# %%
# === 11 Evaluador interactivo ===

def evaluar_frase(frase: str, modelo="LR", umbral=None):
    """
    Evalúa una frase nueva con el modelo entrenado.
    modelo: "LR" o "MLP"
    umbral: si None usa 0.5 o el mejor umbral de F1 para LR
    """
    if not frase.strip():
        print("La frase está vacía.")
        return
    
    # Limpieza 
    texto_limpio = limpiar_texto(frase)
    
    # Embeddings
    emb = get_embeddings([texto_limpio])
    
    # Predicción
    if modelo == "LR":
        proba = clf.predict_proba(emb)[:, 1][0]
        thr = best_thr if umbral is None else umbral
        pred = int(proba >= thr)
        print(f"\n Modelo: Logistic Regression")
    else:
        proba = float(mlp.predict(emb).ravel()[0])
        thr = 0.5 if umbral is None else umbral
        pred = int(proba >= thr)
        print(f"\n Modelo: MLP")
    
    # Resultado
    etiqueta = " CIBERBULLYING" if pred == 1 else " OK! No ciberbullying"
    print(f"\nFrase: {frase}")
    print(f"Probabilidad: {proba:.3f} | Umbral: {thr:.3f}")
    print(f"Resultado: {etiqueta}")

# === Modo interactivo simple ===
while True:
    frase = input("\nEscribí una frase para evaluar (o 'salir' para terminar): ")
    if frase.lower().strip() == "salir":
        break
    evaluar_frase(frase, modelo="LR")  # Se puede cambiar a "MLP" si queremos probar ese
    #evaluar_frase(frase, modelo="MLP")





