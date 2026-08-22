#!/usr/bin/env python3
"""
fingering_to_synthesia.py — Convierte la digitación generada por fingering_v3.py
a formato .synthesia (LocalFingerInfoList / FingerInfo), el sistema de "finger
hints" que usa Synthesia.

────────────────────────────────────────────────────────────────────────────
Entrada
────────────────────────────────────────────────────────────────────────────
Espera el JSON producido por:

    python fingering_v3.py archivo.mid --json digitacion.json

es decir, un objeto con la forma:

    {
      "midi_file": "...",
      "bpm": 120.0,
      "time_signature": "4/4",
      "measures": 32,
      "notes": [
        {"measure": 1, "beat": 0.0, "hand": "right", "note": "C4",
         "pitch": 60, "fingering": 1, ...},
        ...
      ]
    }

────────────────────────────────────────────────────────────────────────────
Formato Synthesia (gramática oficial, Synthesia-LLC/metadata-editor wiki)
────────────────────────────────────────────────────────────────────────────
  1-5        → mano izquierda, dedos 1(pulgar)-5(meñique)
  6-9, 0     → mano derecha,  dedos 1(pulgar)-5(meñique)  (6=R1 … 9=R4, 0=R5)
  -          → nota sin digitación (se salta)
  s          → sustitución de dedo sobre la misma nota (no se genera aquí)
  tN:        → cambia de track N (0-based); reinicia el compás a 1
  mN:        → salta al compás N (1-based) del track actual
  Por defecto: track 0, compás 1.

Este script asume el convenio más habitual: track 0 = mano derecha,
track 1 = mano izquierda — PERO ese número de track se refiere al índice
CRUDO del track en el fichero MIDI (0-based, tal cual aparece en el
fichero), NO al índice entre "solo los tracks que contienen notas". Muchos
ficheros MIDI tipo 1 tienen un primer track vacío (solo tempo/metadatos);
si tu archivo es de ese tipo, la mano derecha real está en el track 1 y la
izquierda en el 2, no en 0/1. Pasa --midi para que el script lo autodetecte
y te lo confirme en pantalla; si no, verifica los índices a mano.

────────────────────────────────────────────────────────────────────────────
Orden físico de notas simultáneas (acordes) — por qué --midi importa
────────────────────────────────────────────────────────────────────────────
Synthesia asigna cada símbolo de digitación a las notas de un track EN EL
ORDEN FÍSICO en que aparecen los eventos note-on en el archivo MIDI. Dentro
de un acorde (varias notas que empiezan en el mismo instante), ese orden
NO tiene por qué ser ascendente por altura: depende de cómo se escribió
originalmente el MIDI, y es habitual que se alternen o inviertan.

fingering_v3.py, en cambio, guarda en su JSON las notas de cada acorde
ordenadas por altura ascendente (conveniente para su propio algoritmo de
digitación, pero no es el orden físico real). Si este conversor usara esa
misma ordenación para generar el "fingers", cada vez que el orden físico
real no coincida con el ascendente, la etiqueta acabaría sobre la nota
equivocada del acorde — típicamente esto produce muchos más fallos en la
mano con más acordes/polifonía.

Por eso este script acepta opcionalmente --midi con el archivo MIDI
original: si se indica, vuelve a leer los eventos note-on tal cual están
en el fichero y reordena las notas de cada grupo simultáneo del JSON según
ese orden físico real antes de codificarlas. Sin --midi, se usa el orden
del JSON (ascendente por altura) como aproximación, lo cual puede producir
digitaciones mal asignadas en los acordes cuyo orden físico esté invertido.

────────────────────────────────────────────────────────────────────────────
El atributo "hash"
────────────────────────────────────────────────────────────────────────────
Synthesia identifica cada pieza por un hash interno (no documentado
públicamente) del propio archivo musical, NO por su ruta o nombre. Este
script no puede recalcular ese hash. Tienes dos opciones:

  1) Pasar --existing tu_archivo_synthesia_real.xml junto con --hash
     el_hash_de_esa_cancion (cópialo de la entrada <FingerInfo> que
     Synthesia ya generó al abrir esa canción por primera vez). El script
     actualizará solo esa entrada, dejando el resto del archivo intacto.

  2) No indicar --hash: se generará un fichero nuevo con un hash
     "placeholder" derivado del nombre del MIDI, y se avisará claramente
     de que casi seguro NO coincidirá con el que espera Synthesia. Tendrás
     que sustituirlo a mano por el real antes de usarlo.

────────────────────────────────────────────────────────────────────────────
Uso
────────────────────────────────────────────────────────────────────────────
    python fingering_to_synthesia.py digitacion.json -o salida.synthesia
    python fingering_to_synthesia.py digitacion.json -o salida.synthesia --hash a48d1b2c62fc908118333dee270986a3
    python fingering_to_synthesia.py digitacion.json -o merged.synthesia \
        --existing LocalFingerInfo.xml --hash a48d1b2c62fc908118333dee270986a3
    python fingering_to_synthesia.py digitacion.json -o salida.synthesia --track-right 0 --track-left 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Colores ANSI (desactivables)
# ─────────────────────────────────────────────────────────────────────────────

class _C:
    OK   = "\033[92m"
    WARN = "\033[93m"
    ERR  = "\033[91m"
    DIM  = "\033[2m"
    END  = "\033[0m"


def _use_color() -> bool:
    return sys.stdout.isatty()


def cprint(msg: str, color: str = "") -> None:
    if color and _use_color():
        print(f"{color}{msg}{_C.END}")
    else:
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Codificación de dedos Synthesia
# ─────────────────────────────────────────────────────────────────────────────

# mano izquierda: dedo 1..5 -> símbolo literal '1'..'5'
_LEFT_SYMBOLS = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
# mano derecha: dedo 1..5 -> '6','7','8','9','0'
_RIGHT_SYMBOLS = {1: "6", 2: "7", 3: "8", 4: "9", 5: "0"}

SKIP = "-"


def finger_symbol(hand: str, fingering: int) -> str:
    """Traduce (mano, dedo 1-5) al símbolo Synthesia. 0 / fuera de rango -> '-'."""
    table = _LEFT_SYMBOLS if hand == "left" else _RIGHT_SYMBOLS
    return table.get(fingering, SKIP)


# ─────────────────────────────────────────────────────────────────────────────
# Lectura del MIDI original para recuperar el orden físico real de eventos
# ─────────────────────────────────────────────────────────────────────────────

def _read_varlen(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            return value, pos


def _parse_midi_tracks(path: Path) -> list[list[tuple[int, int]]]:
    """
    Parser MIDI mínimo (sin dependencias externas): devuelve, para cada
    track del fichero, la lista de eventos note-on (velocity>0) como
    tuplas (tick_absoluto, pitch), EN EL ORDEN FÍSICO en que aparecen.

    Solo interpreta lo necesario (note on/off, delta-times, running status)
    para reconstruir ese orden; ignora el resto de eventos.
    """
    data = path.read_bytes()
    if data[0:4] != b"MThd":
        raise ValueError("No es un fichero MIDI válido (falta cabecera 'MThd')")
    header_len = int.from_bytes(data[4:8], "big")
    n_tracks = int.from_bytes(data[10:12], "big")
    pos = 8 + header_len

    tracks: list[list[tuple[int, int]]] = []
    for _ in range(n_tracks):
        if data[pos:pos + 4] != b"MTrk":
            raise ValueError("Fichero MIDI corrupto: falta marcador 'MTrk'")
        track_len = int.from_bytes(data[pos + 4:pos + 8], "big")
        track_end = pos + 8 + track_len
        p = pos + 8

        events: list[tuple[int, int]] = []
        abs_tick = 0
        running_status = None

        while p < track_end:
            delta, p = _read_varlen(data, p)
            abs_tick += delta

            status = data[p]
            if status < 0x80:
                # running status: reutiliza el último status byte
                if running_status is None:
                    raise ValueError("MIDI corrupto: running status sin status previo")
                status = running_status
            else:
                p += 1
                if status < 0xF0:
                    running_status = status

            if status == 0xFF:  # meta event
                meta_type = data[p]
                p += 1
                length, p = _read_varlen(data, p)
                p += length
                if meta_type == 0x2F:
                    break  # end of track
            elif status in (0xF0, 0xF7):  # sysex
                length, p = _read_varlen(data, p)
                p += length
            else:
                hi = status & 0xF0
                if hi in (0xC0, 0xD0):  # program change / channel pressure: 1 data byte
                    d1 = data[p]
                    p += 1
                else:
                    d1 = data[p]
                    d2 = data[p + 1]
                    p += 2
                    if hi == 0x90 and d2 > 0:  # note on con velocity>0
                        events.append((abs_tick, d1))
                    # note off (0x80) o note on con velocity 0 se ignoran:
                    # no aportan al orden físico de inicio de nota

        tracks.append(events)
        pos = track_end

    return tracks


def detect_note_tracks(all_tracks: list[list[tuple[int, int]]]) -> list[int]:
    """Índices CRUDOS (tal cual están en el fichero MIDI, incluyendo tracks
    vacíos de tempo/meta) de los tracks que contienen al menos una nota."""
    return [i for i, tr in enumerate(all_tracks) if tr]


def load_physical_note_order(midi_path: Path, right_track: int, left_track: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    """
    Lee el MIDI y devuelve (eventos_mano_derecha, eventos_mano_izquierda,
    ticks_per_beat), donde cada lista de eventos son tuplas
    (tick_absoluto, pitch) en orden físico real.

    IMPORTANTE: right_track/left_track son índices de track CRUDOS del
    fichero MIDI (0-based, tal cual aparecen en el fichero), NO índices
    entre "solo los tracks con notas". Todo apunta a que así es como
    Synthesia interpreta también el tN: de su propio formato de fingers:
    como el índice de track crudo, incluyendo tracks vacíos de tempo/meta
    si los hay. Usar el índice "reindexado sin vacíos" (como hace
    fingering_v3.py internamente al leer con pretty_midi) es un error
    fácil de cometer y que produce justo el síntoma de "la digitación
    aparece en la mano equivocada, o no aparece en absoluto".
    """
    with open(midi_path, "rb") as f:
        header = f.read(14)
    ticks_per_beat = int.from_bytes(header[12:14], "big")
    if ticks_per_beat & 0x8000:
        raise ValueError("Este fichero usa SMPTE en vez de ticks-per-beat; no soportado")

    all_tracks = _parse_midi_tracks(midi_path)

    if right_track >= len(all_tracks) or left_track >= len(all_tracks):
        raise ValueError(
            f"El MIDI solo tiene {len(all_tracks)} track(s) en total; "
            f"no se puede usar --track-right {right_track} / --track-left {left_track}"
        )
    if not all_tracks[right_track]:
        cprint(f"  Aviso: el track crudo {right_track} (mano derecha) no tiene ninguna nota en el MIDI.", _C.WARN)
    if not all_tracks[left_track]:
        cprint(f"  Aviso: el track crudo {left_track} (mano izquierda) no tiene ninguna nota en el MIDI.", _C.WARN)

    return all_tracks[right_track], all_tracks[left_track], ticks_per_beat


def group_physical_events(events: list[tuple[int, int]]) -> list[list[int]]:
    """Agrupa eventos (tick, pitch) consecutivos con el mismo tick, preservando
    el orden físico dentro de cada grupo y el orden temporal entre grupos."""
    groups: list[list[int]] = []
    current_tick = None
    for tick, pitch in events:
        if tick != current_tick:
            groups.append([])
            current_tick = tick
        groups[-1].append(pitch)
    return groups


def group_json_notes(notes: list[dict]) -> list[list[dict]]:
    """Agrupa notas del JSON (ya en orden) por (measure, beat) consecutivos idénticos."""
    groups: list[list[dict]] = []
    current_key = None
    for n in notes:
        key = (n["measure"], n["beat"])
        if key != current_key:
            groups.append([])
            current_key = key
        groups[-1].append(n)
    return groups


def reorder_by_physical_order(notes: list[dict], physical_events: list[tuple[int, int]], hand_label: str) -> list[dict]:
    """
    Reordena `notes` (una mano, ya en orden measure/beat) para que, dentro de
    cada grupo de notas simultáneas, sigan el orden físico real de los
    eventos note-on del MIDI en vez del orden por altura del JSON.

    Si el número de grupos o el multiconjunto de alturas de algún grupo no
    coincide entre el JSON y el MIDI, ese grupo concreto se deja tal cual
    venía (con un aviso), en vez de arriesgarse a una reordenación incorrecta.
    """
    json_groups = group_json_notes(notes)
    physical_groups = group_physical_events(physical_events)

    if len(json_groups) != len(physical_groups):
        cprint(
            f"  Aviso ({hand_label}): el número de grupos de notas simultáneas del JSON "
            f"({len(json_groups)}) no coincide con el del MIDI ({len(physical_groups)}); "
            f"se usará el orden del JSON sin corregir para esta mano.",
            _C.WARN,
        )
        return notes

    result: list[dict] = []
    n_reordered_groups = 0
    n_mismatched_groups = 0

    for j_group, p_group in zip(json_groups, physical_groups):
        if len(j_group) != len(p_group) or sorted(n["pitch"] for n in j_group) != sorted(p_group):
            # El contenido no coincide (no debería pasar si es el mismo MIDI);
            # se deja el grupo tal cual para no introducir un emparejamiento falso.
            n_mismatched_groups += 1
            result.extend(j_group)
            continue

        if len(j_group) == 1:
            result.append(j_group[0])
            continue

        pending = list(j_group)
        ordered = []
        for pitch in p_group:
            idx = next(i for i, n in enumerate(pending) if n["pitch"] == pitch)
            ordered.append(pending.pop(idx))
        if [n["pitch"] for n in ordered] != [n["pitch"] for n in j_group]:
            n_reordered_groups += 1
        result.extend(ordered)

    if n_reordered_groups:
        cprint(f"  Mano {hand_label}: {n_reordered_groups} acordes reordenados según el orden físico real del MIDI.", _C.OK)
    if n_mismatched_groups:
        cprint(
            f"  Aviso ({hand_label}): {n_mismatched_groups} grupo(s) con notas que no coinciden "
            f"exactamente entre JSON y MIDI; se dejaron sin reordenar.",
            _C.WARN,
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de la cadena "fingers" para una mano
# ─────────────────────────────────────────────────────────────────────────────

def encode_hand(notes: list[dict], hand: str, gap_encoding: str = "jump") -> tuple[str, int, int]:
    """
    Codifica la secuencia de notas de una mano como tokens Synthesia.

    gap_encoding controla qué hacer cuando, entre dos notas consecutivas de
    esta mano, hay uno o más compases sin ninguna nota:

      - "jump" (por defecto, y el único confirmado por la especificación
        oficial — ver ejemplo "Sevivon, Sov, Sov, Sov" en la wiki de
        Synthesia-LLC/metadata-editor: "m5: 6 m7: 60 m9: 8--0908" salta
        directamente sobre los compases vacíos 6 y 8): usa mN: para saltar
        al compás donde está la siguiente nota real.
      - "pad": alternativa experimental que rellena cada compás vacío con
        un '-'. NO está respaldada por la documentación oficial y, en las
        pruebas con una pieza real, produjo aún más pérdida de datos que
        "jump" al volver a guardarse desde Synthesia. Se deja solo para
        comparar/depurar; no usar salvo que estés investigando el problema.

    Devuelve (cadena_codificada, n_notas_codificadas, n_notas_sin_digitación).
    """
    if not notes:
        return "", 0, 0

    tokens: list[str] = []
    current_measure = 1
    n_skipped = 0
    first_note = True

    for n in notes:
        m = n["measure"]
        if m != current_measure or first_note:
            if gap_encoding == "pad" and m >= current_measure:
                # measures realmente vacías entre lo ya colocado y esta nota.
                # En la primerísima nota, current_measure=1 es solo la posición
                # de partida por defecto (no se ha colocado nada ahí todavía),
                # así que las medidas vacías son 1..m-1 (m-1 en total). Para
                # notas posteriores, current_measure SÍ tuvo ya una nota, así
                # que las vacías son las estrictamente intermedias (m - current - 1).
                empty_measures = (m - 1) if first_note else (m - current_measure - 1)
                if empty_measures > 0:
                    tokens.append(SKIP * empty_measures)
            elif gap_encoding != "pad":
                tokens.append(f" m{m}: ")
            current_measure = m
            first_note = False

        sym = finger_symbol(hand, n.get("fingering", 0))
        if sym == SKIP:
            n_skipped += 1
        tokens.append(sym)

    encoded = "".join(tokens).strip()
    return encoded, len(notes), n_skipped


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del atributo "fingers" completo (ambas manos)
# ─────────────────────────────────────────────────────────────────────────────

def build_fingers_string(
    notes: list[dict],
    track_right: int,
    track_left: int,
    midi_path: Path | None = None,
    gap_encoding: str = "jump",
) -> str:
    if track_right == track_left:
        raise ValueError("--track-right y --track-left no pueden ser el mismo track")

    right_notes = [n for n in notes if n["hand"] == "right"]
    left_notes  = [n for n in notes if n["hand"] == "left"]

    # El JSON de fingering_v3.py ya viene ordenado por (measure, beat, hand, pitch);
    # al filtrar por mano se conserva ese orden. Dentro de cada acorde, sin
    # embargo, ese orden es por altura ascendente, que no tiene por qué
    # coincidir con el orden físico real de los eventos note-on en el MIDI
    # (ver docstring del módulo). Si se indicó --midi, se corrige aquí.
    if midi_path is not None:
        try:
            right_events, left_events, _tpb = load_physical_note_order(midi_path, track_right, track_left)
        except ValueError as e:
            cprint(f"Aviso: no se pudo leer el orden físico del MIDI ({e}); se usará el orden del JSON.", _C.WARN)
        else:
            right_notes = reorder_by_physical_order(right_notes, right_events, "derecha")
            left_notes  = reorder_by_physical_order(left_notes, left_events, "izquierda")
    else:
        cprint(
            "  Aviso: no se indicó --midi; el orden de las notas dentro de cada acorde se toma "
            "por altura ascendente, que puede no coincidir con el orden físico real que espera "
            "Synthesia (ver docstring). Pasa --midi archivo.mid para una digitación exacta.",
            _C.WARN,
        )

    right_enc, n_right, skip_right = encode_hand(right_notes, "right", gap_encoding)
    left_enc,  n_left,  skip_left  = encode_hand(left_notes,  "left", gap_encoding)

    parts: list[str] = []

    if right_enc:
        prefix = "" if track_right == 0 else f"t{track_right}: "
        parts.append(f"{prefix}{right_enc}")

    if left_enc:
        # El track de la mano izquierda casi nunca es el 0 (por defecto), así
        # que casi siempre se emite el prefijo tN: explícito. Si por
        # configuración track_left == 0 y no hay mano derecha, se omite.
        prefix = "" if (track_left == 0 and not right_enc) else f"t{track_left}: "
        parts.append(f"{prefix}{left_enc}")

    fingers = " ".join(p for p in parts if p)

    total_skipped = skip_right + skip_left
    total_notes = n_right + n_left
    if total_notes:
        cprint(
            f"  Mano derecha: {n_right} notas codificadas ({skip_right} sin dedo asignado → '-')",
            _C.DIM,
        )
        cprint(
            f"  Mano izquierda: {n_left} notas codificadas ({skip_left} sin dedo asignado → '-')",
            _C.DIM,
        )
        if total_skipped:
            cprint(
                f"  Aviso: {total_skipped}/{total_notes} notas sin digitación válida (fingering=0) "
                f"se han marcado con '-'.",
                _C.WARN,
            )

    return fingers


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder de hash (best-effort, NO es el hash real de Synthesia)
# ─────────────────────────────────────────────────────────────────────────────

def placeholder_hash(midi_file: str) -> str:
    """
    Genera un identificador determinista a partir del nombre del MIDI para
    que el fichero de salida sea válido y reproducible. Esto NO es el
    algoritmo de hash real que usa Synthesia; solo sirve como marcador de
    posición hasta que el usuario lo sustituya por el hash real (ver
    docstring del módulo).
    """
    return hashlib.md5(midi_file.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Lectura / escritura del XML .synthesia
# ─────────────────────────────────────────────────────────────────────────────

def load_existing(path: Path) -> ET.ElementTree:
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "LocalFingerInfoList":
        raise ValueError(
            f"'{path}' no parece un fichero .synthesia válido "
            f"(raíz esperada <LocalFingerInfoList>, encontrada <{root.tag}>)"
        )
    return tree


def upsert_finger_info(tree: ET.ElementTree, hash_value: str, fingers: str) -> str:
    """
    Inserta o actualiza la entrada <FingerInfo hash="..."> dentro del árbol.
    Devuelve "updated" o "inserted" según lo que haya hecho.
    """
    root = tree.getroot()
    for elem in root.findall("FingerInfo"):
        if elem.get("hash") == hash_value:
            elem.set("fingers", fingers)
            elem.set("version", elem.get("version", "1"))
            return "updated"

    new_elem = ET.SubElement(root, "FingerInfo")
    new_elem.set("hash", hash_value)
    new_elem.set("version", "1")
    new_elem.set("fingers", fingers)
    return "inserted"


def new_tree(hash_value: str, fingers: str) -> ET.ElementTree:
    root = ET.Element("LocalFingerInfoList")
    root.set("version", "1")
    entry = ET.SubElement(root, "FingerInfo")
    entry.set("hash", hash_value)
    entry.set("version", "1")
    entry.set("fingers", fingers)
    return ET.ElementTree(root)


def write_tree(tree: ET.ElementTree, path: Path) -> None:
    root = tree.getroot()
    try:
        ET.indent(root, space="\t")
    except AttributeError:
        pass  # Python < 3.9: se escribe sin indentar, sigue siendo válido
    tree.write(path, encoding="UTF-8", xml_declaration=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convierte la salida JSON de fingering_v3.py a formato Synthesia (.synthesia)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("json_file", help="JSON exportado por fingering_v3.py (--json)")
    ap.add_argument("-o", "--output", required=True, help="Ruta del fichero .synthesia a generar")
    ap.add_argument(
        "--hash",
        default=None,
        help="Hash real de la canción en Synthesia (recomendado). Si se omite, "
             "se genera un placeholder que casi seguro habrá que sustituir a mano.",
    )
    ap.add_argument(
        "--existing",
        default=None,
        help="Fichero .synthesia real existente en el que fusionar/actualizar la entrada "
             "(preserva el resto de canciones guardadas).",
    )
    ap.add_argument(
        "--track-right",
        type=int,
        default=None,
        help="Índice de track CRUDO del MIDI (0-based, tal cual aparece en el fichero, incluyendo "
             "tracks vacíos de tempo/meta) para la mano derecha. Si se pasa --midi y no se indica "
             "esto, se autodetecta como el primer track con notas del fichero.",
    )
    ap.add_argument(
        "--track-left",
        type=int,
        default=None,
        help="Índice de track CRUDO del MIDI para la mano izquierda. Si se pasa --midi y no se "
             "indica esto, se autodetecta como el segundo track con notas del fichero.",
    )
    ap.add_argument(
        "--midi",
        default=None,
        help="Fichero MIDI original. Muy recomendado: permite (1) reordenar las notas de cada "
             "acorde según el orden físico real de los eventos note-on en vez de asumir orden "
             "ascendente por altura, y (2) autodetectar los índices de track correctos para "
             "--track-right/--track-left (incluyendo tracks vacíos de tempo, que Synthesia sí "
             "cuenta en su numeración y que si se ignoran hacen que la digitación de una mano "
             "se aplique a la otra, o a ningún sitio).",
    )
    ap.add_argument(
        "--gap-encoding",
        choices=["pad", "jump"],
        default="jump",
        help="Cómo codificar compases sin ninguna nota para una mano (típico con notas sostenidas "
             "que ocupan varios compases). 'jump' (por defecto, confirmado por la especificación "
             "oficial de Synthesia) salta directamente con mN: al siguiente compás con nota. "
             "'pad' rellena con '-' compás a compás; es experimental, no está respaldado por la "
             "documentación y no se recomienda.",
    )
    args = ap.parse_args()

    in_path = Path(args.json_file)
    if not in_path.exists():
        cprint(f"Error: no existe '{in_path}'", _C.ERR)
        sys.exit(1)

    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        cprint(f"Error: '{in_path}' no es JSON válido ({e})", _C.ERR)
        sys.exit(1)

    notes = data.get("notes")
    if not isinstance(notes, list) or not notes:
        cprint("Error: el JSON no contiene una lista 'notes' con contenido.", _C.ERR)
        sys.exit(1)

    midi_file = data.get("midi_file", "desconocido.mid")
    cprint(f"Procesando digitación de: {midi_file}")
    cprint(f"  Compases: {data.get('measures', '?')}   Compás: {data.get('time_signature', '?')}   BPM: {data.get('bpm', '?')}")

    midi_path = None
    if args.midi:
        midi_path = Path(args.midi)
        if not midi_path.exists():
            cprint(f"Error: no existe el fichero --midi '{midi_path}'", _C.ERR)
            sys.exit(1)

    track_right = args.track_right
    track_left = args.track_left

    if midi_path is not None and (track_right is None or track_left is None):
        try:
            all_tracks = _parse_midi_tracks(midi_path)
        except ValueError as e:
            cprint(f"Error leyendo tracks del MIDI: {e}", _C.ERR)
            sys.exit(1)
        note_track_idx = detect_note_tracks(all_tracks)
        if len(note_track_idx) < 2:
            cprint(
                f"Error: el MIDI solo tiene {len(note_track_idx)} track(s) con notas; "
                f"no se pueden autodetectar mano derecha/izquierda. Indica --track-right/--track-left a mano.",
                _C.ERR,
            )
            sys.exit(1)
        auto_right, auto_left = note_track_idx[0], note_track_idx[1]
        if track_right is None:
            track_right = auto_right
        if track_left is None:
            track_left = auto_left
        cprint(
            f"  Tracks del MIDI (crudo, 0-based): {len(all_tracks)} en total; con notas: {note_track_idx}. "
            f"Usando track {track_right} para mano derecha, track {track_left} para mano izquierda "
            f"(pasa --track-right/--track-left para forzar otra asignación).",
            _C.DIM,
        )
    else:
        if track_right is None:
            track_right = 0
        if track_left is None:
            track_left = 1
        if midi_path is None:
            cprint(
                "  Aviso: sin --midi no se puede verificar el índice de track real; se usará "
                f"--track-right {track_right} / --track-left {track_left} sin comprobar. Si el MIDI "
                "tiene un track vacío inicial (de tempo, muy habitual), esto es casi seguro incorrecto: "
                "pasa --midi para autodetectarlo.",
                _C.WARN,
            )

    try:
        fingers = build_fingers_string(notes, track_right, track_left, midi_path, args.gap_encoding)
    except ValueError as e:
        cprint(f"Error: {e}", _C.ERR)
        sys.exit(1)

    if not fingers:
        cprint("Error: no se ha podido codificar ninguna nota (¿JSON vacío o sin manos válidas?).", _C.ERR)
        sys.exit(1)

    hash_value = args.hash
    used_placeholder = False
    if not hash_value:
        hash_value = placeholder_hash(midi_file)
        used_placeholder = True

    out_path = Path(args.output)

    if args.existing:
        existing_path = Path(args.existing)
        if not existing_path.exists():
            cprint(f"Error: no existe el fichero --existing '{existing_path}'", _C.ERR)
            sys.exit(1)
        try:
            tree = load_existing(existing_path)
        except (ET.ParseError, ValueError) as e:
            cprint(f"Error leyendo '{existing_path}': {e}", _C.ERR)
            sys.exit(1)
        action = upsert_finger_info(tree, hash_value, fingers)
        cprint(f"Entrada {'actualizada' if action == 'updated' else 'insertada'} en '{existing_path.name}'.", _C.OK)
    else:
        tree = new_tree(hash_value, fingers)

    write_tree(tree, out_path)

    cprint(f"\nFichero Synthesia generado → {out_path}", _C.OK)
    cprint(f"hash: {hash_value}")
    if used_placeholder:
        cprint(
            "\nAVISO: no se indicó --hash, así que el hash anterior es un PLACEHOLDER\n"
            "derivado del nombre del MIDI, no el hash real que usa Synthesia.\n"
            "Sustitúyelo por el hash real de esta canción (búscalo en la entrada\n"
            "<FingerInfo> que Synthesia ya guardó al abrir el archivo por primera vez)\n"
            "antes de usar este fichero, o vuelve a ejecutar el script con --hash.",
            _C.WARN,
        )
    cprint(f"\nfingers=\"{fingers}\"", _C.DIM)


if __name__ == "__main__":
    main()
