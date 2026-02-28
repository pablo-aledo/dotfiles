#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           SELECT  v1.0                                       ║
║         Selección interactiva de MIDIs por clustering UMAP + KMeans         ║
║                                                                              ║
║  Vectoriza una colección de MIDIs, agrupa por similitud musical y           ║
║  presenta muestras de cada cluster para escucha y selección manual.         ║
║  Exporta la lista seleccionada y comentarios opcionales por archivo.        ║
║                                                                              ║
║  FLUJO:                                                                      ║
║    [1] Vectoriza todos los MIDIs de midi_folder                             ║
║    [2] Clustering KMeans → n_clusters grupos                                ║
║    [3] Proyección UMAP 2D con color por cluster o score emocional           ║
║    [4] Reproducción de muestras por cluster (pygame)                        ║
║    [5] Selección interactiva: S=mantener, N=descartar, C=continuar          ║
║    [6] Exporta lista seleccionada y comentarios                              ║
║                                                                              ║
║  USO:                                                                        ║
║    python select.py                          (usa carpeta actual)           ║
║    python select.py --folder ./midis/        (carpeta específica)           ║
║    python select.py --clusters 30 --samples 5                               ║
║    python select.py --ref ref1.mid ref2.mid  (ordenar por referencia)      ║
║    python select.py --color score --extended                                ║
║                                                                              ║
║  CONTROLES:                                                                  ║
║    S        — Mantener todos los MIDIs del cluster actual                   ║
║    N        — Descartar el cluster actual                                   ║
║    C        — Continuar (escuchar más muestras del cluster)                 ║
║    Esc      — Saltar la reproducción actual                                 ║
║    C (durante reproducción) — Añadir comentario al MIDI en curso           ║
║    Q        — Guardar y salir                                               ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --folder DIR         Carpeta con los MIDIs a explorar (default: .)      ║
║    --ref MIDI [...]     MIDIs de referencia para ordenar clusters           ║
║    --clusters N         Número de clusters KMeans (default: 20)            ║
║    --samples N          MIDIs a reproducir por cluster (default: 3)        ║
║    --seconds N          Duración máxima por reproducción (default: 10)     ║
║    --output FILE        Archivo de salida seleccionados (default: auto)    ║
║    --comments FILE      Archivo de comentarios (default: auto)             ║
║    --color MODE         Color UMAP: cluster|score (default: cluster)       ║
║    --order MODE         Orden reproducción: extreme|original (default: extreme)║
║    --random-offset      Reproducir desde punto aleatorio del MIDI          ║
║    --extended           Vectorización emocional completa (default: básica) ║
║                                                                              ║
║  SALIDAS:                                                                    ║
║    <folder>_seleccionados.txt  — Rutas de los MIDIs seleccionados          ║
║    <folder>_comentarios.txt    — Comentarios por MIDI                      ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy, scikit-learn, umap-learn, pygame, matplotlib   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
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

# ══════════════════════════════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="Selección interactiva de MIDIs por clustering UMAP + KMeans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--folder", default=".",
                   help="Carpeta con los MIDIs a explorar (default: .)")
    p.add_argument("--ref", nargs="+", default=[], metavar="MIDI",
                   help="MIDIs de referencia para ordenar clusters por similitud")
    p.add_argument("--clusters", type=int, default=20,
                   help="Número de clusters KMeans (default: 20)")
    p.add_argument("--samples", type=int, default=3,
                   help="MIDIs a reproducir por cluster (default: 3)")
    p.add_argument("--seconds", type=int, default=10,
                   help="Duración máxima de cada reproducción en segundos (default: 10)")
    p.add_argument("--output", default=None,
                   help="Archivo de salida para la lista seleccionada (default: <folder>_seleccionados.txt)")
    p.add_argument("--comments", default=None,
                   help="Archivo de salida para comentarios (default: <folder>_comentarios.txt)")
    p.add_argument("--color", default="cluster", choices=["cluster", "score"],
                   help="Modo de color UMAP: cluster o score (default: cluster)")
    p.add_argument("--order", default="extreme", choices=["extreme", "original"],
                   help="Orden de reproducción por cluster: extreme o original (default: extreme)")
    p.add_argument("--random-offset", action="store_true",
                   help="Reproducir MIDIs desde un punto aleatorio")
    p.add_argument("--extended", action="store_true",
                   help="Usar vectorización emocional completa (default: vectorización básica)")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  PYGAME — UTILIDADES DE PANTALLA
# ══════════════════════════════════════════════════════════════════════════════

def init_pygame():
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((700, 140))
    pygame.display.set_caption("Selección de Clusters de MIDI")
    font = pygame.font.SysFont(None, 24)
    return screen, font


def draw_text(screen, font, text):
    screen.fill((0, 0, 0))
    img = font.render(text, True, (255, 255, 255))
    screen.blit(img, (10, 10))
    pygame.display.flip()


# ══════════════════════════════════════════════════════════════════════════════
#  COMENTARIOS
# ══════════════════════════════════════════════════════════════════════════════

def input_comment(screen, font, file_path):
    text = ""
    entering = True
    draw_text(screen, font, f"Comentario para {os.path.basename(file_path)}: (Enter para guardar)")
    while entering:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    entering = False
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                elif event.key <= 127:
                    text += event.unicode
        draw_text(screen, font, f"Comentario para {os.path.basename(file_path)}: {text}")
        pygame.time.wait(50)
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUCCIÓN MIDI
# ══════════════════════════════════════════════════════════════════════════════

def play_midi(file_path, duration=10, random_offset=False):
    try:
        mid = MidiFile(file_path)
        first_note_time = None
        all_note_times = []
        for track in mid.tracks:
            abs_time = 0
            for msg in track:
                abs_time += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    all_note_times.append(abs_time)
                    if first_note_time is None or abs_time < first_note_time:
                        first_note_time = abs_time
        if first_note_time is None:
            first_note_time = 0
        offset_time = first_note_time
        if random_offset and all_note_times:
            offset_time = random.choice(all_note_times)

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


def play_midi_thread(file_path, duration=10, random_offset=False):
    t = threading.Thread(target=play_midi, args=(file_path, duration, random_offset))
    t.start()
    return t


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_vector(file_path, use_extended=False):
    try:
        mid = mido.MidiFile(file_path)
    except Exception:
        return None
    notes, velocities, instruments = [], [], set()
    n_pistas = len(mid.tracks)
    for track in mid.tracks:
        current_program = 0
        for msg in track:
            if msg.type == "program_change":
                current_program = msg.program
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append(msg.note)
                velocities.append(msg.velocity)
                instruments.add(current_program)
    if not notes:
        return None

    notes = np.array(notes)
    velocities = np.array(velocities)
    durations = np.array([
        msg.time if hasattr(msg, "time") and msg.time > 0 else 1
        for track in mid.tracks
        for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    ])
    diffs = np.diff(notes)
    propor_asc_desc = np.sum(diffs > 0) / len(diffs) if len(diffs) > 0 else 0.5
    prop_intervalos_grandes = np.sum(np.abs(diffs) > 5) / len(diffs) if len(diffs) > 0 else 0
    densidad = len(notes) / sum(durations)
    velocity_var = np.var(velocities)

    if use_extended:
        prop_intervalos_dis = np.sum(np.isin(np.abs(diffs), [1, 2, 6, 10])) / len(diffs) if len(diffs) > 0 else 0
        velocity_changes = np.mean(np.abs(np.diff(velocities)) / 127) if len(velocities) > 1 else 0
        n_mayor = sum(1 for n in notes if n % 12 in [0, 2, 4, 5, 7, 9, 11])
        luminosidad = n_mayor / len(notes)
        tonalidad_dominante = max([notes.tolist().count(n % 12) for n in notes]) / len(notes)
        fuera_tonalidad = 1 - tonalidad_dominante
        prop_extremos = np.sum((notes < 40) | (notes > 80)) / len(notes)
        perc_count = sum(1 for i in instruments if 115 <= i <= 127)
        string_count = sum(1 for i in instruments if 40 <= i <= 71)
        instr_perc = perc_count / len(instruments) if instruments else 0
        instr_str = string_count / len(instruments) if instruments else 0
        vector = np.array([
            np.mean(notes), np.std(notes), np.max(notes) - np.min(notes),
            densidad, np.mean(velocities), np.std(velocities),
            propor_asc_desc, prop_intervalos_grandes, prop_intervalos_dis,
            luminosidad, fuera_tonalidad,
            n_pistas, len(instruments), instr_perc, instr_str,
            prop_extremos, velocity_changes,
        ])
        score = (velocity_var / 128 + prop_intervalos_grandes + prop_intervalos_dis + densidad / 10 +
                 luminosidad + fuera_tonalidad + prop_extremos + velocity_changes) / 8
    else:
        vector = np.array([
            np.mean(notes), np.std(notes), np.max(notes) - np.min(notes),
            densidad, np.mean(velocities), np.std(velocities),
            propor_asc_desc, prop_intervalos_grandes, n_pistas, len(instruments),
        ])
        score = (velocity_var / 128 + prop_intervalos_grandes + densidad / 10) / 3

    return vector, score


# ══════════════════════════════════════════════════════════════════════════════
#  UMAP — VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def build_umap_plot(vectors, labels, scores, color_mode):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    embedding = reducer.fit_transform(vectors)

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 8))

    if color_mode == "cluster":
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab20", s=15, alpha=0.6)
        plt.colorbar(scatter, label="Cluster")
    elif color_mode == "score":
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=scores, cmap="plasma", s=15, alpha=0.6)
        plt.colorbar(scatter, label="Score emocional")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], c="gray", s=15, alpha=0.6)

    highlight, = ax.plot([], [], "ro", markersize=12)
    text_info = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=10, verticalalignment="bottom")
    plt.show()

    return fig, ax, embedding, highlight, text_info


def highlight_midi(idx, embedding, vectors, valid_files, labels, scores, highlight, text_info, fig):
    highlight.set_data(embedding[idx, 0], embedding[idx, 1])
    vec = vectors[idx]
    text_info.set_text(
        f"{os.path.basename(valid_files[idx])}\n"
        f"Cluster: {labels[idx]}\n"
        f"Score: {np.round(scores[idx], 2)}\n"
        f"Vector: {np.round(vec, 2)}"
    )
    fig.canvas.draw()
    fig.canvas.flush_events()


def make_hover_handler(ax, embedding, vectors, valid_files, labels, scores, text_info):
    def on_hover(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            distances = np.linalg.norm(embedding - np.array([x, y]), axis=1)
            min_idx = np.argmin(distances)
            if distances[min_idx] < 0.05:
                vec = vectors[min_idx]
                text_info.set_text(
                    f"{os.path.basename(valid_files[min_idx])}\n"
                    f"Cluster: {labels[min_idx]}\n"
                    f"Score: {np.round(scores[min_idx], 2)}\n"
                    f"Vector: {np.round(vec, 2)}"
                )
            else:
                text_info.set_text("")
            ax.figure.canvas.draw_idle()
    return on_hover


# ══════════════════════════════════════════════════════════════════════════════
#  SELECCIÓN INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════

def run_selection(
    cluster_order, labels, valid_files, scores, vectors, embedding,
    highlight, text_info, fig, screen, font,
    n_clusters, samples_per_cluster, play_seconds,
    use_random_offset, cluster_play_order,
    output_file, comments_file,
):
    selected_files = []
    comments = {}

    def save_and_exit():
        with open(output_file, "w") as f:
            for mf in selected_files:
                f.write(f"{mf}\n")
        with open(comments_file, "w") as f:
            for midi_file, comment in comments.items():
                f.write(f"{os.path.basename(midi_file)}: {comment}\n")
        print(f"{len(selected_files)} MIDIs seleccionados. Guardado en {output_file}.")
        print(f"{len(comments)} comentarios guardados en {comments_file}.")
        pygame.quit()
        sys.exit(0)

    for cluster_id in cluster_order:
        cluster_files = [valid_files[i] for i in range(len(valid_files)) if labels[i] == cluster_id]
        already_played = set()
        draw_text(screen, font, f"Cluster {cluster_id}: {len(cluster_files)} MIDIs. Reproduciendo muestras...")

        while True:
            to_play = [f for f in cluster_files if f not in already_played]

            if cluster_play_order == "extreme" and to_play:
                cluster_scores = np.array([scores[valid_files.index(f)] for f in to_play])
                mean_score = np.mean(cluster_scores)
                dist_to_mean = np.abs(cluster_scores - mean_score)
                sorted_idx = np.argsort(-dist_to_mean)
                to_play = [to_play[i] for i in sorted_idx]

            if not to_play:
                draw_text(screen, font, "No quedan más MIDIs por reproducir en este cluster. Reiniciando...")
                already_played.clear()
                continue

            selected_samples = to_play[:samples_per_cluster]

            for f in selected_samples:
                idx = valid_files.index(f)
                highlight_midi(idx, embedding, vectors, valid_files, labels, scores, highlight, text_info, fig)
                draw_text(screen, font, f"Reproduciendo {os.path.basename(f)} ({play_seconds}s). <Esc> saltar, C=Comentario")
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
                                comments[f] = input_comment(screen, font, f)
                play_thread.join()
                already_played.add(f)

            draw_text(screen, font, "S: mantener, N: descartar, C: continuar, Q: salir y guardar")
            waiting = True
            event = None
            while waiting:
                plt.pause(0.05)
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:
                            selected_files.extend(cluster_files)
                            waiting = False
                            break
                        elif event.key == pygame.K_n:
                            waiting = False
                            break
                        elif event.key == pygame.K_c:
                            waiting = False
                            break
                        elif event.key == pygame.K_q:
                            save_and_exit()

            if event is not None and event.key in [pygame.K_s, pygame.K_n]:
                break

    save_and_exit()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolver nombres de archivos de salida
    folder_stem = os.path.basename(os.path.abspath(args.folder))
    output_file   = args.output   or f"{folder_stem}_seleccionados.txt"
    comments_file = args.comments or f"{folder_stem}_comentarios.txt"

    # Vectorizar colección
    midi_files = [os.path.join(args.folder, f) for f in os.listdir(args.folder) if f.endswith(".mid")]
    if not midi_files:
        print(f"[ERROR] No se encontraron archivos .mid en: {args.folder}")
        sys.exit(1)

    vectors, valid_files, scores = [], [], []
    print("Vectorizando MIDIs...")
    total = len(midi_files)
    for i, f in enumerate(midi_files, 1):
        result = midi_to_vector(f, use_extended=args.extended)
        if result is not None:
            vec, score = result
            vectors.append(vec)
            valid_files.append(f)
            scores.append(score)
        if i % 100 == 0 or i == total:
            print(f"  {i} / {total}", end="\r")
    vectors = np.array(vectors)
    scores  = np.array(scores)
    print(f"\n  {len(valid_files)} MIDIs válidos.")

    # Vectorizar referencias
    ref_vectors = []
    for f in args.ref:
        result = midi_to_vector(f, use_extended=args.extended)
        if result is not None:
            ref_vectors.append(result[0])
    ref_vectors = np.array(ref_vectors) if ref_vectors else np.array([])

    # Clustering
    print("Calculando clustering KMeans...")
    kmeans = KMeans(n_clusters=args.clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    cluster_centers = kmeans.cluster_centers_

    # Ordenar clusters
    if len(ref_vectors) > 0:
        cluster_dists = []
        for c in range(args.clusters):
            dist = np.mean([np.linalg.norm(cluster_centers[c] - rv) for rv in ref_vectors])
            cluster_dists.append((c, dist))
        cluster_order = [c for c, _ in sorted(cluster_dists, key=lambda x: x[1])]
    else:
        cluster_scores = []
        for c in range(args.clusters):
            idxs = np.where(labels == c)[0]
            cluster_scores.append((c, np.mean(scores[idxs])))
        cluster_order = [c for c, _ in sorted(cluster_scores, key=lambda x: -x[1])]

    # UMAP + visualización
    print("Calculando proyección UMAP...")
    fig, ax, embedding, highlight, text_info = build_umap_plot(vectors, labels, scores, args.color)
    hover = make_hover_handler(ax, embedding, vectors, valid_files, labels, scores, text_info)
    fig.canvas.mpl_connect("motion_notify_event", hover)

    # Inicializar pygame
    screen, font = init_pygame()

    # Selección interactiva
    run_selection(
        cluster_order=cluster_order,
        labels=labels,
        valid_files=valid_files,
        scores=scores,
        vectors=vectors,
        embedding=embedding,
        highlight=highlight,
        text_info=text_info,
        fig=fig,
        screen=screen,
        font=font,
        n_clusters=args.clusters,
        samples_per_cluster=args.samples,
        play_seconds=args.seconds,
        use_random_offset=args.random_offset,
        cluster_play_order=args.order,
        output_file=output_file,
        comments_file=comments_file,
    )


if __name__ == "__main__":
    main()
