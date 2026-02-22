import os
import mido
import numpy as np
from sklearn.cluster import KMeans
import random
import pygame
import time
import tempfile
import threading
import umap
import matplotlib.pyplot as plt
from mido import MidiFile, MidiTrack

# --- Configuración ---
midi_folder = '.'
reference_midis = []  # lista de MIDIs de referencia
samples_per_cluster = 3
play_seconds = 10
n_clusters = 20
output_file = 'midis_seleccionados.txt'
comments_file = 'midis_comentarios.txt'

# Opciones visualización y reproducción
umap_color_mode = "cluster"  # "cluster" o "score"
cluster_play_order = "extreme"  # "extreme" o "original"
use_random_offset = True        # reproducir MIDIs con offset aleatorio

# --- Inicializar pygame ---
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((700,140))
pygame.display.set_caption("Selección de Clusters de MIDI")
font = pygame.font.SysFont(None,24)
def draw_text(text):
    screen.fill((0,0,0))
    img = font.render(text, True, (255,255,255))
    screen.blit(img, (10,10))
    pygame.display.flip()

# --- Comentarios ---
comments = {}  # clave: ruta MIDI, valor: texto
def input_comment(file_path):
    text = ""
    entering = True
    draw_text(f"Comentario para {os.path.basename(file_path)}: (Enter para guardar)")
    while entering:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    entering = False
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                elif event.key <= 127:
                    text += event.unicode
        draw_text(f"Comentario para {os.path.basename(file_path)}: {text}")
        pygame.time.wait(50)
    return text

# --- Reproducción MIDI con eliminación de silencios iniciales y offset aleatorio ---
def play_midi(file_path, duration=10, random_offset=False):
    try:
        mid = MidiFile(file_path)
        # Buscar primer note_on para eliminar silencios iniciales
        first_note_time = None
        all_note_times = []
        for track in mid.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type=='note_on' and msg.velocity>0:
                    all_note_times.append(abs_time)
                    if first_note_time is None or abs_time<first_note_time:
                        first_note_time = abs_time

        if first_note_time is None:
            first_note_time = 0
        # Offset aleatorio
        offset_time = first_note_time
        if random_offset and all_note_times:
            offset_time = random.choice(all_note_times)

        # Crear midi temporal ajustando tiempos
        temp_mid = MidiFile()
        temp_mid.ticks_per_beat = mid.ticks_per_beat
        for track in mid.tracks:
            new_track = MidiTrack()
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.is_meta:
                    new_track.append(msg.copy())
                    continue
                if abs_time < offset_time:
                    continue
                dt = msg.time if abs_time != offset_time else 0
                new_track.append(msg.copy(time=dt))
            temp_mid.tracks.append(new_track)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmpfile:
            temp_mid.save(tmpfile.name)
            pygame.mixer.music.load(tmpfile.name)
            pygame.mixer.music.play()
        start_time = time.time()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
            if time.time() - start_time >= duration:
                pygame.mixer.music.stop()
                break
    except Exception as e:
        print(f"Error reproduciendo {file_path}: {e}")
        return

def play_midi_thread(file_path, duration=10, random_offset=False):
    t = threading.Thread(target=play_midi, args=(file_path,duration,random_offset))
    t.start()
    return t

# --- Vectorización y score emocional ---
def midi_to_vector(file_path):
    try:
        mid = mido.MidiFile(file_path)
    except:
        return None
    notes, velocities, instruments = [],[],set()
    n_pistas = len(mid.tracks)
    for track in mid.tracks:
        current_program = 0
        for msg in track:
            if msg.type=='program_change':
                current_program = msg.program
            if msg.type=='note_on' and msg.velocity>0:
                notes.append(msg.note)
                velocities.append(msg.velocity)
                instruments.add(current_program)
    if not notes:
        return None
    notes = np.array(notes)
    velocities = np.array(velocities)
    durations = np.array([msg.time if hasattr(msg,'time') and msg.time>0 else 1
                          for track in mid.tracks for msg in track if msg.type=='note_on' and msg.velocity>0])
    diffs = np.diff(notes)
    propor_asc_desc = np.sum(diffs>0)/len(diffs) if len(diffs)>0 else 0.5
    prop_intervalos_grandes = np.sum(np.abs(diffs)>5)/len(diffs) if len(diffs)>0 else 0
    densidad = len(notes)/sum(durations)
    velocity_var = np.var(velocities)
    n_mayor = sum(1 for n in notes if n%12 in [0,2,4,5,7,9,11])
    luminosidad = n_mayor/len(notes)
    vector = np.array([np.mean(notes), np.std(notes), np.max(notes)-np.min(notes),
                       densidad, np.mean(velocities), np.std(velocities),
                       propor_asc_desc, n_pistas, len(instruments), prop_intervalos_grandes])
    score = (velocity_var/128 + prop_intervalos_grandes + densidad/10 + luminosidad)/4
    return vector, score

# --- Cargar y vectorizar colección ---
midi_files = [os.path.join(midi_folder,f) for f in os.listdir(midi_folder) if f.endswith('.mid')]
vectors, valid_files, scores = [],[],[]
print("Vectorizando MIDIs...")
for f in midi_files:
    result = midi_to_vector(f)
    if result is not None:
        vec, score = result
        vectors.append(vec)
        valid_files.append(f)
        scores.append(score)
vectors = np.array(vectors)
scores = np.array(scores)
print(f"{len(valid_files)} MIDIs válidos.")

# --- Vectorizar MIDIs de referencia ---
ref_vectors = []
for f in reference_midis:
    result = midi_to_vector(f)
    if result is not None:
        vec,_ = result
        ref_vectors.append(vec)
ref_vectors = np.array(ref_vectors)

# --- Clustering ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(vectors)
cluster_centers = kmeans.cluster_centers_

# --- Ordenar clusters ---
if len(ref_vectors)>0:
    cluster_dists = []
    for c in range(n_clusters):
        dist = np.mean([np.linalg.norm(cvec - ref_vec) for cvec in [cluster_centers[c]] for ref_vec in ref_vectors])
        cluster_dists.append((c, dist))
    cluster_order = [c for c,_ in sorted(cluster_dists, key=lambda x:x[1])]
else:
    cluster_scores = []
    for c in range(n_clusters):
        idxs = np.where(labels==c)[0]
        avg_score = np.mean(scores[idxs])
        cluster_scores.append((c, avg_score))
    cluster_order = [c for c,_ in sorted(cluster_scores, key=lambda x:-x[1])]

# --- UMAP ---
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(vectors)

plt.ion()
fig, ax = plt.subplots(figsize=(12,8))

# Color según configuración
if umap_color_mode == "cluster":
    scatter = ax.scatter(embedding[:,0], embedding[:,1], c=labels, cmap='tab20', s=15, alpha=0.6)
    plt.colorbar(scatter,label="Cluster")
elif umap_color_mode == "score":
    scatter = ax.scatter(embedding[:,0], embedding[:,1], c=scores, cmap='plasma', s=15, alpha=0.6)
    plt.colorbar(scatter,label="Score emocional")
else:
    scatter = ax.scatter(embedding[:,0], embedding[:,1], c='gray', s=15, alpha=0.6)

# Highlight activo
highlight, = ax.plot([], [], 'ro', markersize=12)
text_info = ax.text(0.02,0.95,"", transform=ax.transAxes, fontsize=10, verticalalignment='top')

# Hover fijo en esquina inferior izquierda, sin flecha
hover_annotation = ax.annotate("", xy=(0.02,0.02), xycoords="axes fraction",
                               xytext=(0,0), textcoords="offset points",
                               bbox=dict(boxstyle="round", fc="yellow", alpha=0.8))
hover_annotation.set_visible(False)
plt.show()

def highlight_midi(idx):
    highlight.set_data(embedding[idx,0], embedding[idx,1])
    vec = vectors[idx]
    text_info.set_text(f"{os.path.basename(valid_files[idx])}\nCluster: {labels[idx]}\nScore: {np.round(scores[idx],2)}\nVector: {np.round(vec,2)}")
    fig.canvas.draw()
    fig.canvas.flush_events()

def on_hover(event):
    if event.inaxes==ax:
        x, y = event.xdata, event.ydata
        distances = np.linalg.norm(embedding - np.array([x, y]), axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx]<0.05:
            hover_annotation.set_visible(True)
            vec = vectors[min_idx]
            hover_annotation.set_text(f"{os.path.basename(valid_files[min_idx])}\nCluster: {labels[min_idx]}\nScore: {np.round(scores[min_idx],2)}\nVector: {np.round(vec,2)}")
            fig.canvas.draw_idle()
        else:
            hover_annotation.set_visible(False)
            fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", on_hover)

# --- Selección interactiva ---
selected_files=[]
def save_and_exit():
    with open(output_file,'w') as f:
        for mf in selected_files:
            f.write(f"{mf}\n")
    with open(comments_file,'w') as f:
        for midi_file, comment in comments.items():
            f.write(f"{os.path.basename(midi_file)}: {comment}\n")
    print(f"{len(selected_files)} MIDIs seleccionados. Guardado en {output_file}.")
    print(f"{len(comments)} comentarios guardados en {comments_file}.")
    pygame.quit()
    exit()

for cluster_id in cluster_order:
    cluster_files = [valid_files[i] for i in range(len(valid_files)) if labels[i]==cluster_id]
    already_played = set()
    draw_text(f"Cluster {cluster_id}: {len(cluster_files)} MIDIs. Reproduciendo muestras...")

    while True:
        to_play = [f for f in cluster_files if f not in already_played]

        # --- Reiniciar si todos reproducidos ---
        if not to_play:
            already_played = set()
            draw_text(f"Reiniciando cluster {cluster_id} desde el principio...")
            plt.pause(0.5)
            to_play = [f for f in cluster_files]

        # --- Ordenar por score emocional si cluster_play_order == "extreme"
        if cluster_play_order == "extreme" and to_play:
            cluster_scores = np.array([scores[valid_files.index(f)] for f in to_play])
            mean_score = np.mean(cluster_scores)
            dist_to_mean = np.abs(cluster_scores - mean_score)
            sorted_idx = np.argsort(-dist_to_mean)  # más extremo primero
            to_play = [to_play[i] for i in sorted_idx]

        selected_samples = to_play[:samples_per_cluster]

        for f in selected_samples:
            idx = valid_files.index(f)
            highlight_midi(idx)
            draw_text(f"Reproduciendo {os.path.basename(f)} ({play_seconds}s). <Esc> para saltar, C=Comentario")
            play_thread = play_midi_thread(f, duration=play_seconds, random_offset=use_random_offset)

            start_time = time.time()
            while time.time() - start_time < play_seconds and play_thread.is_alive():
                plt.pause(0.05)
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.mixer.music.stop()
                            play_thread.join()
                        elif event.key == pygame.K_c:
                            comment = input_comment(f)
                            comments[f] = comment
            play_thread.join()
            already_played.add(f)

        draw_text("S: mantener, N: descartar, C: continuar, Q: salir y guardar")
        waiting = True
        while waiting:
            plt.pause(0.05)
            for event in pygame.event.get():
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_s:
                        selected_files.extend(cluster_files)
                        waiting=False
                        break
                    elif event.key==pygame.K_n:
                        waiting=False
                        break
                    elif event.key==pygame.K_c:
                        waiting=False
                        break
                    elif event.key==pygame.K_q:
                        save_and_exit()
        if event.key in [pygame.K_s, pygame.K_n]:
            break

save_and_exit()
