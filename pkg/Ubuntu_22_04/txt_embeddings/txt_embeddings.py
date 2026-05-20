#!/usr/bin/env python3
"""
txt_embeddings.py — Visualizador semántico de textos con embeddings + clustering
==================================================================================

Toma un fichero de textos (uno por línea), calcula embeddings con OpenAI,
reduce a 2D con UMAP, agrupa con KMeans (o clusters externos), genera un
HTML interactivo con Plotly y exporta resúmenes GPT-4 por cluster.

REQUISITOS
----------
    pip install openai umap-learn scikit-learn plotly pandas numpy
    export OPENAI_API_KEY="sk-..."

USO BÁSICO
----------
    # Clustering automático (5 clusters por defecto)
    python txt_embeddings.py --input textos.txt

    # Especificar número de clusters
    python txt_embeddings.py --input textos.txt --n_clusters 8

    # Usar clusters predefinidos (fichero con una etiqueta por línea, mismo orden que textos.txt)
    python txt_embeddings.py --input textos.txt --clusters etiquetas.txt

    # Cambiar rutas de salida
    python txt_embeddings.py --input textos.txt \
        --output_html mapa.html \
        --output_csv puntos.csv \
        --output_summary_csv resumen.csv

ARGUMENTOS
----------
    --input              Fichero de textos de entrada (una línea = un texto)
                         Default: textos.txt
    --clusters           (Opcional) Fichero con etiquetas de cluster, una por línea.
                         Si se omite, se aplica KMeans sobre el espacio UMAP.
    --n_clusters         Número de clusters para KMeans. Default: 5
    --output_html        Fichero HTML de salida con gráfico interactivo + tabla.
                         Default: embeddings_interactivos.html
    --output_csv         CSV con coordenadas 2D y cluster de cada punto.
                         Default: embeddings_procesados.csv
    --output_summary_csv CSV con resumen GPT-4 por cluster.
                         Default: resumen_clusters.csv

EJEMPLOS DE USO TÍPICOS
-----------------------
    1. Explorar temáticas en un corpus de reseñas:
       python txt_embeddings.py --input resenas.txt --n_clusters 6

    2. Validar clusters anotados manualmente:
       python txt_embeddings.py --input frases.txt --clusters categorias_manuales.txt

    3. Pipeline completo con rutas explícitas:
       python txt_embeddings.py \
           --input corpus/noticias.txt \
           --n_clusters 10 \
           --output_html resultados/mapa_noticias.html \
           --output_csv resultados/puntos.csv \
           --output_summary_csv resultados/temas.csv

FORMATO DEL FICHERO DE TEXTOS (--input)
----------------------------------------
    Una línea por texto. Líneas vacías se ignoran. Ejemplo:
        El modelo aprende representaciones densas del lenguaje.
        La música barroca usa contrapunto estricto.
        Los embeddings capturan similitud semántica.

FORMATO DEL FICHERO DE CLUSTERS (--clusters)
---------------------------------------------
    Una etiqueta por línea, mismo número de líneas que --input. Ejemplo:
        NLP
        Música
        NLP

SALIDAS
-------
    *.html   Gráfico Plotly interactivo (hover muestra texto) + tabla de resúmenes
    *.csv    Coordenadas x/y UMAP, texto original, cluster de cada punto
    *_summary.csv  Una fila por cluster: id, n_textos, resumen generado por GPT-4

NOTAS
-----
    - Los embeddings usan text-embedding-3-small (1536 dims).
    - Los resúmenes usan gpt-4; el prompt se trunca a ~750 tokens por cluster
      para respetar el límite TPM de la API.
    - UMAP usa random_state=42 para reproducibilidad.
"""

import os
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
from openai import OpenAI
import umap
from sklearn.cluster import KMeans

# =========================
# ARGUMENTOS
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="textos.txt", help="Fichero de textos")
parser.add_argument("--clusters", help="Fichero opcional con clusters predefinidos")
parser.add_argument("--output_html", default="embeddings_interactivos.html", help="HTML de salida")
parser.add_argument("--output_csv", default="embeddings_procesados.csv", help="CSV de salida de puntos")
parser.add_argument("--output_summary_csv", default="resumen_clusters.csv", help="CSV de resúmenes por cluster")
parser.add_argument("--n_clusters", type=int, default=5, help="Número de clusters si se usa KMeans")
args = parser.parse_args()

MODEL = "text-embedding-3-small"
BATCH_SIZE = 100
RESUMEN_MODEL = "gpt-4"

# =========================
# 1. Leer API Key
# =========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No se encontró la variable de entorno OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# =========================
# 2. Leer textos
# =========================
if not os.path.exists(args.input):
    raise FileNotFoundError(f"No se encontró el fichero: {args.input}")

lines = []
with open(args.input, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            lines.append(str(line))

if not lines:
    raise ValueError("El fichero de textos está vacío.")

print(f"Se cargaron {len(lines)} textos")

# =========================
# 3. Obtener embeddings
# =========================
def get_embeddings_batch(client, texts, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Procesando batch {i} - {i + len(batch)}")
        response = client.embeddings.create(
            model=MODEL,
            input=batch
        )
        all_embeddings.extend([e.embedding for e in response.data])
    return np.array(all_embeddings)

embeddings = get_embeddings_batch(client, lines, BATCH_SIZE)
print("Embeddings calculados")

# =========================
# 4. Reducir a 2D (UMAP)
# =========================
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)
print("Reducción dimensional completada")

# =========================
# 5. Obtener clusters
# =========================
if args.clusters:
    if not os.path.exists(args.clusters):
        raise FileNotFoundError(f"No se encontró el fichero de clusters: {args.clusters}")

    cluster_labels = []
    with open(args.clusters, "r", encoding="utf-8") as f:
        for line in f:
            cluster_labels.append(line.strip())

    if len(cluster_labels) != len(lines):
        raise ValueError(
            f"El número de clusters ({len(cluster_labels)}) "
            f"no coincide con el número de textos ({len(lines)})"
        )

    print("Clusters cargados desde fichero externo")

else:
    print(f"Calculando KMeans sobre espacio UMAP con {args.n_clusters} clusters")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(embedding_2d)

print("Clusters listos")

# =========================
# 6. DataFrame de puntos
# =========================
def recortar_texto(t, max_len=100):
    return t[:max_len] + "..." if len(t) > max_len else t

df = pd.DataFrame({
    "x": embedding_2d[:, 0],
    "y": embedding_2d[:, 1],
    "texto_original": lines,
    "texto_hover": [recortar_texto(t, 100) for t in lines],
    "cluster": [str(c) for c in cluster_labels]
})

# =========================
# 7. Exportar CSV de puntos
# =========================
df.to_csv(args.output_csv, index=False)
print(f"CSV de puntos generado: {args.output_csv}")

# =========================
# 8. Generar resúmenes por cluster
# =========================
MAX_PROMPT_CHARS = 3000  # ~750 tokens, seguro para TPM 10k

def preparar_textos_para_prompt(textos, max_chars=MAX_PROMPT_CHARS):
    """Selecciona textos truncando por caracteres totales para no reventar el TPM."""
    resultado = []
    acumulado = 0
    for t in textos:
        t_corto = t[:200]  # truncar cada texto individual a 200 chars
        if acumulado + len(t_corto) > max_chars:
            break
        resultado.append(t_corto)
        acumulado += len(t_corto)
    return resultado

resumenes = []
for cluster_id in sorted(set(cluster_labels)):
    cluster_texts = df[df["cluster"] == str(cluster_id)]["texto_original"].tolist()
    textos_prompt = preparar_textos_para_prompt(cluster_texts)
    prompt = (
        "Resume en 1-2 frases el contenido común de estos textos:\n\n"
        + "\n".join(textos_prompt)
    )
    response = client.chat.completions.create(
        model=RESUMEN_MODEL,
        messages=[
            {"role": "system", "content": "Eres un asistente que resume textos brevemente."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=150
    )
    resumen = response.choices[0].message.content.strip()
    resumenes.append({
        "cluster": str(cluster_id),
        "n_textos": len(cluster_texts),
        "resumen": resumen
    })
    print(f"Resumen cluster {cluster_id}: {resumen}")

df_resumen = pd.DataFrame(resumenes)
df_resumen.to_csv(args.output_summary_csv, index=False)
print(f"CSV de resumen por cluster generado: {args.output_summary_csv}")

# =========================
# 9. Plot interactivo + tabla de resúmenes en HTML
# =========================
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    title="Embeddings 2D (UMAP) + Clusters",
    custom_data=["texto_hover", "cluster"]
)

fig.update_traces(
    hovertemplate="<b>Cluster:</b> %{customdata[1]}<br>"
                  "<b>Texto:</b> %{customdata[0]}"
                  "<extra></extra>",
    marker=dict(size=8)
)

# Generar tabla HTML de resúmenes
tabla_html = df_resumen.to_html(index=False, escape=False)

# Guardar HTML combinado
with open(args.output_html, "w", encoding="utf-8") as f:
    f.write("<html><head><title>Embeddings UMAP + Clusters</title></head><body>\n")
    f.write("<h1>Gráfico interactivo</h1>\n")
    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("<h2>Resúmenes por cluster</h2>\n")
    f.write(tabla_html)
    f.write("</body></html>")

print(f"HTML final con tabla de resúmenes generado: {args.output_html}")

