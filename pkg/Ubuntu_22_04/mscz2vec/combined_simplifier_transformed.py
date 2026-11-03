#!/usr/bin/env python3
r"""
combined_simplifier_transformed.py — Simplificador de piano por grados con capa
de transformaciones musicales (MIDI → MIDI)
(versión 1.2: extensión de combined_simplifier.py).

NOVEDADES v5 (niveles): --level beginner|intermediate|advanced deriva versiones con un perfil
aprendido de 553 MIDIs graduados (ver bloque [NIVEL]); --target-grade 1..5 usa ese perfil (--legacy
para el motor clasico); --excerpt recorta extractos; --transpose es opt-in (en el corpus la
transposicion automatica coincidia con la real menos que no transponer).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Parte del motor estructural de combined_simplifier.py (suelo schenkeriano +
greedy por compás; se importa de él, así que ambos ficheros deben estar en la
misma carpeta) y le añade una capa de transformaciones que no se limita a
eliminar notas. Para cada --target-grade se toma el resultado estructural y se
aplican transformaciones en pasadas sucesivas, aceptando solo una que reduzca
estrictamente el grado global medido (prueba-y-acepta-si-mejora; hasta
--max-transform-passes pasadas o hasta que ninguna mejore). Si aun así no se
alcanza el grado, se recurre a la rejilla de respaldo de combined_simplifier.py.

Transformaciones disponibles:
  - fusión de notas repetidas
  - cuantización rítmica suave
  - alineación de manos (menos independencia entre ellas)
  - reducción de la extensión de los acordes
  - reducción de saltos
  - arpegiado de acordes densos
  - simplificación de acordes (recorte de voces)
  - adición opcional de notas de paso

────────────────────────────────────────────────────────────────────────────
Modos de uso (--transform-mode)
────────────────────────────────────────────────────────────────────────────
  off           Sin transformaciones: equivale al motor estructural puro
                (método "estructural", más rejilla si hace falta).
  conservative  Solo fusión de repetidas, cuantización suave y alineación de
                manos.
  balanced      (defecto) conservative + reducción de extensión (≤ 14
                semitonos), reducción de saltos (≤ 9) y arpegiado de acordes
                densos (bloques de 2 notas si el grado ≤ 2, de 3 si mayor).
  rewrite       balanced + simplificación de acordes (1 voz si el grado ≤ 1,
                2 si mayor) + cuantización más gruesa + notas de paso (el modo
                implica --allow-added-notes).

────────────────────────────────────────────────────────────────────────────
Entradas y salidas
────────────────────────────────────────────────────────────────────────────
  Entrada : un MIDI de piano (posicional).
  Salida  : <nombre>_grade<N>.mid por cada --target-grade en --outdir (o junto
            al MIDI) e informe por consola con el grado alcanzado y el método
            (p. ej. "estructural+transformaciones(balanced)", con
            "+rejilla" si hubo que recurrir a ella).

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
    midi                       Fichero MIDI de entrada (posicional).
    --target-grade N [N …]     Uno o más grados objetivo (obligatorio).
    --floor-level L            Nivel del suelo estructural: 'ursatz' (defecto) o
                               un índice entero (véase combined_simplifier.py).
    --split PITCH              Pitch MIDI que separa las manos (defecto 60).
    --outdir DIR               Carpeta de salida (defecto: la del MIDI).
    --transform-mode M         off | conservative | balanced (defecto) | rewrite.
    --allow-added-notes        Permite transformaciones que añaden notas de paso
                               (en rewrite ya vienen incluidas).
    --max-transform-passes N   Máximo de pasadas de transformaciones (defecto 8).

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  # Tres grados con el modo por defecto (balanced):
  python combined_simplifier_transformed.py obra.mid --target-grade 1 3 5

  # Reescritura completa (incluye notas de paso añadidas):
  python combined_simplifier_transformed.py obra.mid --target-grade 2 \
      --transform-mode rewrite --allow-added-notes

  # Transformaciones prudentes, solo ritmo y manos:
  python combined_simplifier_transformed.py obra.mid --target-grade 4 --transform-mode conservative

  # Sin transformaciones (comparar con el motor estructural puro):
  python combined_simplifier_transformed.py obra.mid --target-grade 4 --transform-mode off

  # Balanced con notas de paso, más pasadas y carpeta de salida:
  python combined_simplifier_transformed.py obra.mid --target-grade 3 \
      --allow-added-notes --max-transform-passes 12 --outdir out/ --split 60

────────────────────────────────────────────────────────────────────────────
Dependencias y relación con otras versiones
────────────────────────────────────────────────────────────────────────────
numpy y combined_simplifier.py (importado). Las versiones autocontenidas que
integran estas ideas son combined_simplifier_transformed_2.py (v2.0) y
combined_simplifier_hybrid.py / combined_simplifier_hybrid_v2.py.
"""

import sys
import argparse
import bisect
from pathlib import Path
from typing import List

try:
    from combined_simplifier import *
    import combined_simplifier as base
except ImportError:
    print(
        "[ERROR] Este programa transformado necesita 'combined_simplifier.py' "
        "en el mismo directorio.",
        file=sys.stderr,
    )
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  [TRANSFORMACIONES] Utilidades básicas
# ══════════════════════════════════════════════════════════════════════════════

def _clone_records(records: List["NoteRec"]) -> List["NoteRec"]:
    """Copia profunda de registros de nota."""
    out = []
    for r in records:
        n = Note(
            pitch=r.note.pitch,
            start=r.note.start,
            end=r.note.end,
            vel=r.note.vel,
            channel=r.note.channel,
        )
        out.append(
            NoteRec(
                note=n,
                hand=r.hand,
                bar=r.bar,
                voice_role=r.voice_role,
                function=r.function,
                weight=r.weight,
                floor_protected=r.floor_protected,
                is_octave_dup=r.is_octave_dup,
                candidate=r.candidate,
                orig_id=r.orig_id,
            )
        )
    return out


def _records_grade(records: List["NoteRec"], tc: "TimeContext") -> int:
    """Grado global de un conjunto de NoteRec."""
    rh = [r.note for r in records if r.hand == "rh"]
    lh = [r.note for r in records if r.hand == "lh"]
    g, _ = whole_piece_grade(rh, lh, tc)
    return g


def _sort_records(records: List["NoteRec"]) -> List["NoteRec"]:
    return sorted(records, key=lambda r: (r.note.start, r.note.pitch))


def _trim_overlaps(records: List["NoteRec"]) -> List["NoteRec"]:
    """Recorta superposiciones obvias, evitando tocar notas protegidas."""
    for hand in ("rh", "lh"):
        hs = sorted(
            [r for r in records if r.hand == hand],
            key=lambda r: (r.note.start, r.note.pitch),
        )
        for i in range(len(hs) - 1):
            cur = hs[i]
            nxt = hs[i + 1]
            if cur.note.end > nxt.note.start and not cur.floor_protected:
                cur.note.end = max(cur.note.start + 1, nxt.note.start)
    return records


# ══════════════════════════════════════════════════════════════════════════════
#  [TRANSFORMACIONES] Operaciones concretas
# ══════════════════════════════════════════════════════════════════════════════

def _merge_repeated_notes(records: List["NoteRec"], tc: "TimeContext") -> List["NoteRec"]:
    """
    Fusiona notas repetidas contiguas en la misma mano.
    Reduce ataques y, por tanto, velocidad percibida.
    """
    out = []
    for hand in ("rh", "lh"):
        hs = sorted(
            [r for r in records if r.hand == hand],
            key=lambda r: (r.note.start, r.note.pitch),
        )
        merged = []
        for r in hs:
            if (
                merged
                and merged[-1].note.pitch == r.note.pitch
                and merged[-1].note.end >= r.note.start - 1
            ):
                if r.floor_protected:
                    merged.append(r)
                else:
                    merged[-1].note.end = max(merged[-1].note.end, r.note.end)
            else:
                merged.append(r)
        out.extend(merged)
    return _sort_records(out)


def _quantize_onsets(
    records: List["NoteRec"],
    tc: "TimeContext",
    grid_ticks: int,
    protect_floor: bool = True,
) -> List["NoteRec"]:
    """
    Cuantiza onsets a una rejilla, respetando las notas protegidas por el suelo.
    """
    if grid_ticks <= 1:
        return records

    for r in records:
        if protect_floor and r.floor_protected:
            continue

        dur = max(1, r.note.end - r.note.start)
        q = int(round(r.note.start / float(grid_ticks)) * grid_ticks)
        if q < 0:
            q = 0

        r.note.start = q
        r.note.end = max(q + 1, q + dur)
        r.bar = tc.bar(r.note.start)

    _trim_overlaps(records)
    return _sort_records(records)


def _align_hands(
    records: List["NoteRec"],
    tc: "TimeContext",
    max_shift_ticks: int = None,
) -> List["NoteRec"]:
    """
    Alinea notas de la mano izquierda con onsets cercanos de la mano derecha
    para reducir independencia rítmica entre manos.
    """
    grid = max(1, tc.tpb // 4)
    if max_shift_ticks is None:
        max_shift_ticks = grid

    rh_onsets = sorted({r.note.start for r in records if r.hand == "rh"})
    if not rh_onsets:
        return records

    for r in records:
        if r.hand != "lh" or r.floor_protected:
            continue

        pos = bisect.bisect_left(rh_onsets, r.note.start)
        cands = []
        if pos < len(rh_onsets):
            cands.append(rh_onsets[pos])
        if pos > 0:
            cands.append(rh_onsets[pos - 1])
        if not cands:
            continue

        nearest = min(cands, key=lambda x: abs(x - r.note.start))
        shift = nearest - r.note.start

        if shift != 0 and abs(shift) <= max_shift_ticks:
            dur = max(1, r.note.end - r.note.start)
            r.note.start = nearest
            r.note.end = max(r.note.start + 1, r.note.start + dur)
            r.bar = tc.bar(r.note.start)

    _trim_overlaps(records)
    return _sort_records(records)


def _reduce_extension(
    records: List["NoteRec"],
    tc: "TimeContext",
    max_span: int = 14,
) -> List["NoteRec"]:
    """
    Reduce la extensión instantánea de acordes moviendo extremos una octava
    cuando eso disminuye el span y la nota no está protegida.
    """
    groups = {}
    for r in records:
        groups.setdefault((r.hand, r.note.start), []).append(r)

    for group in groups.values():
        if len(group) < 2:
            continue

        group.sort(key=lambda r: r.note.pitch)
        old_span = group[-1].note.pitch - group[0].note.pitch
        if old_span <= max_span:
            continue

        candidates = []
        top = group[-1]
        bottom = group[0]

        if not top.floor_protected:
            new_pitch = top.note.pitch - 12
            if 0 <= new_pitch <= 127:
                pitches = [r.note.pitch for r in group[:-1]] + [new_pitch]
                span = max(pitches) - min(pitches)
                if span < old_span:
                    candidates.append((span, top, new_pitch))

        if not bottom.floor_protected:
            new_pitch = bottom.note.pitch + 12
            if 0 <= new_pitch <= 127:
                pitches = [new_pitch] + [r.note.pitch for r in group[1:]]
                span = max(pitches) - min(pitches)
                if span < old_span:
                    candidates.append((span, bottom, new_pitch))

        if candidates:
            _, rec, new_pitch = min(candidates, key=lambda x: x[0])
            rec.note.pitch = new_pitch

    return _sort_records(records)


def _hand_span_at(group: List["NoteRec"]) -> int:
    if not group:
        return 0
    pitches = [r.note.pitch for r in group]
    return max(pitches) - min(pitches)


def _reduce_leaps(
    records: List["NoteRec"],
    tc: "TimeContext",
    max_leap: int = 9,
) -> List["NoteRec"]:
    """
    Reduce saltos melódicos grandes desplazando octavas la nota superior
    de cada evento cuando eso mejora la distancia con la nota superior anterior.
    """
    for hand in ("rh", "lh"):
        hand_recs = [r for r in records if r.hand == hand]
        onsets = sorted({r.note.start for r in hand_recs})

        groups_by_onset = {}
        top_by_onset = {}

        for onset in onsets:
            group = [r for r in hand_recs if r.note.start == onset]
            groups_by_onset[onset] = group
            top_by_onset[onset] = max(group, key=lambda r: r.note.pitch)

        prev_pitch = None

        for onset in onsets:
            top = top_by_onset[onset]

            if prev_pitch is not None and not top.floor_protected:
                leap = abs(top.note.pitch - prev_pitch)
                if leap > max_leap:
                    group = groups_by_onset[onset]
                    old_span = _hand_span_at(group)

                    best_pitch = top.note.pitch
                    best_diff = leap

                    for cand in (
                        top.note.pitch - 12,
                        top.note.pitch + 12,
                        top.note.pitch - 24,
                        top.note.pitch + 24,
                    ):
                        if not (0 <= cand <= 127):
                            continue

                        diff = abs(cand - prev_pitch)
                        if diff >= best_diff:
                            continue

                        pitches = [
                            r.note.pitch if r is not top else cand
                            for r in group
                        ]
                        new_span = max(pitches) - min(pitches)

                        if new_span <= max(old_span, 14):
                            best_pitch = cand
                            best_diff = diff

                    if best_pitch != top.note.pitch:
                        top.note.pitch = best_pitch

            prev_pitch = top.note.pitch

    return _sort_records(records)


def _arpeggiate_chords(
    records: List["NoteRec"],
    tc: "TimeContext",
    max_block: int = 3,
    grid_ticks: int = None,
) -> List["NoteRec"]:
    """
    Convierte bloques densos en arpegios, manteniendo un ancla en el onset
    y desplazando las notas restantes.
    """
    if grid_ticks is None:
        grid_ticks = max(1, tc.tpb // 8)

    groups = {}
    for r in records:
        groups.setdefault((r.hand, r.note.start), []).append(r)

    for (hand, onset), group in groups.items():
        if len(group) <= max_block:
            continue

        group.sort(key=lambda r: r.note.pitch)

        anchor = group[-1] if hand == "rh" else group[0]
        movable = [r for r in group if r is not anchor and not r.floor_protected]

        if not movable:
            continue

        order = sorted(movable, key=lambda r: r.note.pitch)

        for i, r in enumerate(order, start=1):
            dur = max(1, r.note.end - r.note.start)
            new_start = onset + i * grid_ticks
            r.note.start = new_start
            r.note.end = max(new_start + 1, new_start + dur)
            r.bar = tc.bar(new_start)

    _trim_overlaps(records)
    return _sort_records(records)


def _simplify_chords(
    records: List["NoteRec"],
    tc: "TimeContext",
    max_voices: int = 2,
) -> List["NoteRec"]:
    """
    Simplifica acordes conservando notas protegidas y, si falta espacio,
    extremos del acorde.
    """
    groups = {}
    for r in records:
        groups.setdefault((r.hand, r.note.start), []).append(r)

    out = []

    for (hand, onset), group in groups.items():
        group.sort(key=lambda r: r.note.pitch)

        protected = [r for r in group if r.floor_protected]
        chosen = list(protected)

        if len(chosen) < max_voices:
            if max_voices == 1:
                preferred = group[-1] if hand == "rh" else group[0]
                if preferred not in chosen:
                    chosen.append(preferred)
            else:
                for r in (group[0], group[-1]):
                    if r not in chosen and len(chosen) < max_voices:
                        chosen.append(r)

        if not chosen and group:
            chosen = [group[-1] if hand == "rh" else group[0]]

        out.extend(chosen)

    return _sort_records(out)


def _add_passing_notes(
    records: List["NoteRec"],
    tc: "TimeContext",
    max_leap: int = 9,
    grid_ticks: int = None,
) -> List["NoteRec"]:
    """
    Añade notas de paso simples entre saltos grandes si hay hueco temporal.
    Es una transformación opcional y conservadora: solo se acepta si luego
    mejora el grado global.
    """
    if grid_ticks is None:
        grid_ticks = max(1, tc.tpb // 4)

    new_records = list(records)

    for hand in ("rh", "lh"):
        hand_recs = [r for r in records if r.hand == hand]
        onsets = sorted({r.note.start for r in hand_recs})

        top_by_onset = {}
        for onset in onsets:
            group = [r for r in hand_recs if r.note.start == onset]
            top_by_onset[onset] = max(group, key=lambda r: r.note.pitch)

        prev_onset = None
        prev_pitch = None

        for onset in onsets:
            top = top_by_onset[onset]

            if prev_pitch is not None:
                leap = top.note.pitch - prev_pitch

                if abs(leap) > max_leap and onset - prev_onset >= 2 * grid_ticks:
                    start = prev_onset + grid_ticks
                    if start < onset:
                        sign = 1 if leap > 0 else -1
                        step = min(7, max(2, abs(leap) // 2))
                        mid_pitch = prev_pitch + sign * step

                        if 0 <= mid_pitch <= 127:
                            dur = max(grid_ticks, onset - start)
                            new_note = Note(
                                pitch=mid_pitch,
                                start=start,
                                end=start + dur,
                                vel=top.note.vel,
                                channel=top.note.channel,
                            )
                            new_records.append(
                                NoteRec(
                                    note=new_note,
                                    hand=hand,
                                    bar=tc.bar(start),
                                    voice_role="inner",
                                    function="PT",
                                    weight=0.2,
                                    floor_protected=False,
                                    is_octave_dup=False,
                                    candidate=True,
                                    orig_id=-1,
                                )
                            )

            prev_onset = onset
            prev_pitch = top.note.pitch

    _trim_overlaps(new_records)
    return _sort_records(new_records)


# ══════════════════════════════════════════════════════════════════════════════
#  [TRANSFORMACIONES] Orquestador
# ══════════════════════════════════════════════════════════════════════════════

def apply_musical_transformations(
    records: List["NoteRec"],
    tc: "TimeContext",
    target_grade: int,
    tpb: int,
    mode: str = "balanced",
    allow_added_notes: bool = False,
    max_passes: int = 8,
) -> Tuple[List["NoteRec"], int]:
    """
    Aplica transformaciones musicales aceptando solo aquellas que reducen
    el grado global estimado.
    """
    cur = _clone_records(records)
    grade = _records_grade(cur, tc)

    if mode == "off" or grade <= target_grade:
        return cur, grade

    conservative = [
        lambda rs: _merge_repeated_notes(rs, tc),
        lambda rs: _quantize_onsets(rs, tc, max(1, tpb // 4)),
        lambda rs: _align_hands(rs, tc),
    ]

    balanced = conservative + [
        lambda rs: _reduce_extension(rs, tc, max_span=14),
        lambda rs: _reduce_leaps(rs, tc, max_leap=9),
        lambda rs: _arpeggiate_chords(
            rs,
            tc,
            max_block=2 if target_grade <= 2 else 3,
            grid_ticks=max(1, tpb // 8),
        ),
    ]

    rewrite = balanced + [
        lambda rs: _simplify_chords(rs, tc, max_voices=1 if target_grade <= 1 else 2),
        lambda rs: _quantize_onsets(rs, tc, max(1, tpb // 2)),
    ]

    if mode == "conservative":
        transforms = conservative
    elif mode == "balanced":
        transforms = balanced
    elif mode == "rewrite":
        transforms = rewrite
    else:
        transforms = balanced

    if allow_added_notes or mode == "rewrite":
        transforms.append(
            lambda rs: _add_passing_notes(
                rs,
                tc,
                max_leap=9,
                grid_ticks=max(1, tpb // 4),
            )
        )

    for _ in range(max_passes):
        if grade <= target_grade:
            break

        improved = False

        for fn in transforms:
            cand = _clone_records(cur)
            try:
                cand = fn(cand)
            except Exception:
                continue

            if not cand:
                continue

            g = _records_grade(cand, tc)

            if g < grade:
                cur = cand
                grade = g
                improved = True
                break

        if not improved:
            break

    return cur, grade


# ══════════════════════════════════════════════════════════════════════════════
#  [ORQUESTACION] API principal transformada
# ══════════════════════════════════════════════════════════════════════════════

def combined_simplify_transformed(
    midi_path: str,
    target_grades: List[int],
    floor_level: str = "ursatz",
    split: int = 60,
    transform_mode: str = "balanced",
    allow_added_notes: bool = False,
    max_transform_passes: int = 8,
) -> Dict:
    """
    Versión transformada de combined_simplify().

    Flujo:
      1. reducción estructural original;
      2. transformaciones musicales no necesariamente eliminatorias;
      3. rejilla de respaldo si aún no se llega al objetivo.
    """
    result = base.reduce_graded(
        midi_path,
        target_grades,
        floor_level=floor_level,
        split=split,
    )

    tc, mid = result["tc"], result["mid"]
    n_original = len(result["records"])
    levels: Dict[int, LevelResult] = {}

    for t in sorted(set(target_grades), reverse=True):
        snap = result["level_snapshots"][t]

        if transform_mode == "off":
            transformed = _clone_records(snap)
            achieved = _records_grade(transformed, tc)
            method = "estructural"
        else:
            transformed, achieved = apply_musical_transformations(
                snap,
                tc,
                target_grade=t,
                tpb=mid.tpb,
                mode=transform_mode,
                allow_added_notes=allow_added_notes,
                max_passes=max_transform_passes,
            )
            method = f"estructural+transformaciones({transform_mode})"

        if achieved > t:
            reduced, achieved2, ok, steps = base.apply_grid_fallback_until_target(
                transformed,
                tc,
                mid.tpb,
                t,
            )
            if ok:
                method = f"{method}+rejilla"
            else:
                method = f"{method}+rejilla (limite tras {steps} peldaños)"

            transformed = reduced
            achieved = achieved2

        levels[t] = LevelResult(
            target_grade=t,
            achieved_grade=achieved,
            method=method,
            n_notes=len(transformed),
            n_notes_original=n_original,
            records=transformed,
        )

    return {
        "mid": mid,
        "tc": tc,
        "levels": levels,
        "result": result,
    }


def print_combined_report_transformed(midi_path: str, out: Dict):
    print(f"\n{'═' * 78}")
    print(f"COMBINED SIMPLIFIER TRANSFORMED v1.2 — {midi_path}")
    print(f"{'═' * 78}")
    print(f"  grado original estimado: {out['result']['grade_before_piece']}/8")

    for t in sorted(out["levels"], reverse=True):
        lv = out["levels"][t]
        pct = 100.0 * lv.n_notes / max(1, lv.n_notes_original)
        flag = "" if lv.achieved_grade <= t else "  [AVISO: no se alcanzo el target ni con la rejilla completa]"

        print(
            f"  target-grade {t}: alcanzado {lv.achieved_grade}/8  "
            f"metodo={lv.method}  notas={lv.n_notes}/{lv.n_notes_original} ({pct:.0f}%){flag}"
        )

    print(f"{'═' * 78}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  [NIVEL] DERIVACION DE VERSIONES beginner / intermediate / advanced (v5.0)
#
#  Capa derivada del analisis de un corpus de 553 MIDIs de piano graduados en
#  tres niveles (167 advanced, 231 intermediate, 154 beginner; 76 piezas con
#  version advanced + al menos una inferior). Lo que distingue de forma
#  robusta una version de otra (medido sobre los ficheros cuantizados, p10-p90):
#
#                            beginner        intermediate     advanced
#    voces por acorde (MD/MI)  ~1 (max 3)      1-3 (max 4)      3-4 (max 5)
#    onsets MD con acorde      5%              14% (p75 52%)    39%
#    dobles octavas            0%              0-10%            7-31%
#    ataque mas corto MD/MI    1/2 / 1 pulso   1/4 / 1/2 pulso  1/16 / 1/4
#    extension max acorde      MD 8  MI 7      MD 12 MI 12      MD 14 MI 18
#    MD pitch max (p10-p90)    72-84           74-90            77-100
#    MI pitch min (p10-p90)    37-55           31-45            26-41
#    notas/s pico (p50)        3.0             6.0              10.5
#    saltos MI > 9 st          0%              3%               23%
#
#  y, en 21 pares realmente alineables (misma pieza, mismas notas y tiempos),
#  la version inferior es casi siempre SELECCION de la avanzada (89-90% de sus
#  notas estan en la avanzada; ~6% desplazadas de octava; ~5% nuevas), con estas
#  reglas de seleccion: en acordes de >=3 notas de la MD se conserva la nota
#  superior (85% intermediate / 48% beginner), muy poco el resto; la MI se
#  adelgaza a 1.0-1.3 voces por ataque y se queda en los pulsos (ataques
#  conservados: en pulso 84%/73% MD, 21%/46% MI; en corchea 61%/45% MD; en
#  semicorchea 44%/29% MD).
#
#  derive_level_records() aplica esas reglas como pasos deterministas
#  (cuantizado -> voces/dobles/extension -> registro -> suelo ritmico ->
#  sostenidos -> tope de densidad). level_simplify() la orquesta; si el grado
#  del perfil sigue por encima del tope, recurre a la rejilla de respaldo.
# ══════════════════════════════════════════════════════════════════════════════

LEVEL_PROFILES = {
    "beginner": dict(
        target_grade=4,
        snap=0.25,                       # rejilla de cuantizado previo (pulsos)
        rh_voices=1, lh_voices=1,        # voces maximas por ataque (ajustado con los pares reales)
        rh_span=12, lh_span=12,          # extension maxima (semitonos)
        rh_range=(57, 84), lh_range=(38, 67),   # p10-p90 reales
        rh_min_ioi=0.5, lh_min_ioi=2.0,  # separacion minima entre ataques (pulsos)
        peak_nps=6.0,                    # notas/s maximas por mano (ventana 2 s)
        drop_octave_doubles=True,
        fill_rh=2.0, fill_lh=4.0,        # sostenido maximo (pulsos) al rellenar huecos
        excerpt_frac=0.40, excerpt_min=8, excerpt_max=32,
    ),
    "intermediate": dict(
        target_grade=5,
        snap=0.125,
        rh_voices=1, lh_voices=2,
        rh_span=12, lh_span=16,
        rh_range=(48, 90), lh_range=(31, 74),
        rh_min_ioi=0.25, lh_min_ioi=1.0,
        peak_nps=10.0,
        drop_octave_doubles=True,
        fill_rh=2.0, fill_lh=4.0,
        excerpt_frac=None,
    ),
    "advanced": dict(
        target_grade=8, identity=True, excerpt_frac=None,
    ),
}

# Rango real (p10, p50, p90) de rasgos sobre los ficheros cuantizados de cada
# nivel; solo para el informe de adecuacion al perfil.
LEVEL_REFERENCE = {
    "beginner":     dict(rh_chord_max=(1, 2, 3), lh_chord_max=(1, 1, 3), rh_span_max=(0, 3, 10.4), lh_span_max=(0, 0, 12),
                         rh_pitch_max=(72, 78, 84), lh_pitch_min=(36.8, 47, 55), rh_ioi_min=(0.25, 0.5, 1.0),
                         lh_ioi_min=(0.32, 1.0, 3.0), peak_nps=(2.0, 3.0, 7.6), black=(0.0, 0.11, 0.52)),
    "intermediate": dict(rh_chord_max=(1, 2, 4), lh_chord_max=(1, 2, 3), rh_span_max=(0, 4, 12), lh_span_max=(0, 7, 16),
                         rh_pitch_max=(74, 81, 90), lh_pitch_min=(31, 38, 45), rh_ioi_min=(0.11, 0.25, 0.5),
                         lh_ioi_min=(0.25, 0.5, 1.1), peak_nps=(3.5, 6.0, 10.1), black=(0.0, 0.16, 0.54)),
    "advanced":     dict(rh_chord_max=(3, 4, 5), lh_chord_max=(2, 4, 5), rh_span_max=(12, 12, 18), lh_span_max=(12, 12, 29),
                         rh_pitch_max=(77, 89, 100), lh_pitch_min=(26, 31, 41), rh_ioi_min=(0.05, 0.12, 0.5),
                         lh_ioi_min=(0.06, 0.25, 0.5), peak_nps=(6.5, 10.5, 22.0), black=(0.10, 0.31, 0.62)),
}

_LEVEL_ALIASES = {"beginner": "beginner", "principiante": "beginner", "basico": "beginner", "básico": "beginner",
                  "intermediate": "intermediate", "intermedio": "intermediate",
                  "advanced": "advanced", "avanzado": "advanced"}


def resolve_level(name: str) -> str:
    key = _LEVEL_ALIASES.get(str(name).strip().lower())
    if key is None:
        raise ValueError(f"nivel desconocido: {name!r} (usa beginner | intermediate | advanced)")
    return key


def _lv_clone(records):
    import dataclasses as _dc
    return [_dc.replace(r, note=_dc.replace(r.note)) for r in records]


def _lv_clusters(recs, tol=0):
    recs = sorted(recs, key=lambda r: (r.note.start, r.note.pitch))
    out = []
    for r in recs:
        if out and r.note.start - out[-1][0].note.start <= tol:
            out[-1].append(r)
        else:
            out.append([r])
    return out


def _lv_strength(tick, tc):
    """5 = principio de compas, 4 = pulso, 3 = corchea, 2 = semicorchea, 0 = fuera de rejilla."""
    try:
        bib = tc.beat_in_bar(tick, 4)
    except Exception:
        bib = (tick / tc.tpb) % 4
    if abs(bib) < 0.03 or abs(bib - 4) < 0.03:
        return 5
    if abs(bib - round(bib)) < 0.03:
        return 4
    if abs(bib * 2 - round(bib * 2)) < 0.06:
        return 3
    if abs(bib * 4 - round(bib * 4)) < 0.12:
        return 2
    return 0


def _lv_prio(cluster, hand, tc):
    """Prioridad de un ataque (cluster de notas simultaneas de una mano)."""
    role = "mel" if hand == "rh" else "bass"
    p = 2.0 * _lv_strength(cluster[0].note.start, tc)
    if any(r.voice_role == role for r in cluster):
        p += 3.0
    p += sum(r.weight for r in cluster) / max(1, len(cluster))
    return p


def _lv_pick(group, hand, cap, max_span, drop_dups):
    """Elige las voces que sobreviven en un ataque: ancla (nota superior en la
    MD, inferior en la MI), luego el otro extremo, luego por peso; respeta
    nº de voces y extension. Devuelve la lista de NoteRec conservadas."""
    notes = sorted(group, key=lambda r: r.note.pitch)
    if len(notes) == 1:
        return notes
    anchor = notes[-1] if hand == "rh" else notes[0]
    if drop_dups:
        seen, keep = set(), []
        for r in (reversed(notes) if hand == "rh" else notes):
            pc = r.note.pitch % 12
            if pc in seen:
                continue
            seen.add(pc)
            keep.append(r)
        notes = sorted(keep, key=lambda r: r.note.pitch)
    chosen = [anchor]
    other_end = notes[0] if hand == "rh" else notes[-1]
    ranked = [other_end] + sorted((r for r in notes if r is not anchor and r is not other_end),
                                  key=lambda r: -r.weight)
    for r in ranked:
        if len(chosen) >= cap:
            break
        if r is anchor or r in chosen:
            continue
        ps = [c.note.pitch for c in chosen] + [r.note.pitch]
        if max_span and max(ps) - min(ps) > max_span:
            continue
        chosen.append(r)
    return chosen


def _lv_fold(p, lo, hi):
    if p > hi:
        p -= 12 * ((p - hi + 11) // 12)
    elif p < lo:
        p += 12 * ((lo - p + 11) // 12)
    return p


def _lv_cap_density(cur, hand, tc, cap_nps, win=2.0):
    cl = _lv_clusters([r for r in cur if r.hand == hand], 0)
    if not cl:
        return cur, 0
    secs = [tc.sec(c[0].note.start) for c in cl]
    alive = [True] * len(cl)
    removed = 0
    k = 0
    while k < len(cl):
        if not alive[k]:
            k += 1
            continue
        members, count = [], 0
        for m in range(k, len(cl)):
            if secs[m] - secs[k] >= win:
                break
            if alive[m]:
                members.append(m)
                count += len(cl[m])
        if len(members) > 1 and count / win > cap_nps:
            worst = min(members, key=lambda m: (_lv_prio(cl[m], hand, tc), -secs[m]))
            alive[worst] = False
            removed += len(cl[worst])
            continue
        k += 1
    dead = {id(r) for c, a in zip(cl, alive) if not a for r in c}
    return [r for r in cur if id(r) not in dead], removed


def derive_level_records(records, tc, level, stats=None):
    """Deriva la version `level` a partir de los NoteRec de superficie
    (version avanzada). Devuelve (records, stats)."""
    prof = LEVEL_PROFILES[resolve_level(level)] if isinstance(level, str) else level
    st = stats if stats is not None else {}
    cur = _lv_clone(records)
    if prof.get("identity"):
        return cur, st
    tpb = tc.tpb

    # 1. cuantizado previo (las versiones graduadas viven sobre rejilla)
    if prof.get("snap"):
        g = max(1, int(round(tpb * prof["snap"])))
        for r in cur:
            n = r.note
            s = int(round(n.start / g)) * g
            e = int(round(n.end / g)) * g
            n.start, n.end = s, (e if e > s else s + g)

    # 2. voces por ataque, dobles octavas y extension
    n0 = len(cur)
    dead = set()
    for hand in ("rh", "lh"):
        cap, span = prof[f"{hand}_voices"], prof[f"{hand}_span"]
        for cl in _lv_clusters([r for r in cur if r.hand == hand], 0):
            keep = _lv_pick(cl, hand, cap, span, prof.get("drop_octave_doubles", True))
            kid = {id(r) for r in keep}
            dead.update(id(r) for r in cl if id(r) not in kid)
    cur = [r for r in cur if id(r) not in dead]
    st["voces_y_dobles_eliminadas"] = n0 - len(cur)

    # 3. registro: plegar por octavas al rango del nivel
    nfold = 0
    for r in cur:
        lo, hi = prof[f"{r.hand}_range"]
        q = _lv_fold(r.note.pitch, lo, hi)
        if q != r.note.pitch:
            r.note.pitch = q
            nfold += 1
    st["notas_plegadas_de_octava"] = nfold
    seen, dedup = set(), []
    for r in sorted(cur, key=lambda r: (r.note.start, r.note.pitch)):
        key = (r.hand, r.note.start, r.note.pitch)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    cur = dedup

    # 4. suelo ritmico por mano: separacion minima entre ataques
    n1 = len(cur)
    drop = set()
    for hand in ("rh", "lh"):
        min_ioi = int(round(tpb * prof[f"{hand}_min_ioi"]))
        kept = []
        for cl in _lv_clusters([r for r in cur if r.hand == hand], 0):
            if not kept or cl[0].note.start - kept[-1][0].note.start >= min_ioi:
                kept.append(cl)
                continue
            better = _lv_prio(cl, hand, tc) > _lv_prio(kept[-1], hand, tc)
            ok_prev = len(kept) < 2 or cl[0].note.start - kept[-2][0].note.start >= min_ioi
            if better and ok_prev:
                drop.update(id(r) for r in kept[-1])
                kept[-1] = cl
            else:
                drop.update(id(r) for r in cl)
    cur = [r for r in cur if id(r) not in drop]
    st["ataques_eliminados_por_ritmo"] = n1 - len(cur)

    # 5. tope de densidad (notas/s pico por mano)
    n2 = len(cur)
    for hand in ("rh", "lh"):
        cur, _ = _lv_cap_density(cur, hand, tc, prof["peak_nps"])
    st["notas_eliminadas_por_densidad"] = n2 - len(cur)

    # 6. sostenidos: cada nota dura hasta el siguiente ataque de su mano (con tope)
    for hand in ("rh", "lh"):
        cap_t = int(round(tpb * prof[f"fill_{hand}"]))
        cl = _lv_clusters([r for r in cur if r.hand == hand], 0)
        for i, c in enumerate(cl):
            nxt = cl[i + 1][0].note.start if i + 1 < len(cl) else None
            for r in c:
                n = r.note
                target = n.start + cap_t if nxt is None else min(nxt, n.start + cap_t)
                if n.end < target:
                    n.end = target

    # 7. superposiciones del mismo tono (MIDI valido) y compas
    for hand in ("rh", "lh"):
        last = {}
        for r in sorted((r for r in cur if r.hand == hand), key=lambda r: (r.note.start, r.note.pitch)):
            prev = last.get(r.note.pitch)
            if prev is not None and prev.note.end > r.note.start:
                prev.note.end = max(prev.note.start + 1, r.note.start)
            last[r.note.pitch] = r
    for r in cur:
        r.bar = tc.bar(r.note.start)
        r.floor_protected = False
    cur.sort(key=lambda r: (r.note.start, r.note.pitch))
    return cur, st


def _lv_hand_feats(recs, tc, hand):
    ns = [r.note for r in recs if r.hand == hand]
    if not ns:
        return dict(chord_max=0, span_max=0, pitch_max=0, pitch_min=0, ioi_min=0)
    by = {}
    for n in ns:
        by.setdefault(n.start, []).append(n.pitch)
    on = sorted(by)
    spans = [max(v) - min(v) for v in by.values() if len(v) > 1]
    iois = [(b - a) / tc.tpb for a, b in zip(on[:-1], on[1:])]
    return dict(chord_max=max(len(v) for v in by.values()), span_max=max(spans) if spans else 0,
                pitch_max=max(n.pitch for n in ns), pitch_min=min(n.pitch for n in ns),
                ioi_min=min(iois) if iois else 0)


def level_profile_report(records, tc, level):
    """Compara el resultado con el rango real (p10-p90) de ese nivel en el corpus."""
    level = resolve_level(level)
    ref = LEVEL_REFERENCE[level]
    rh, lh = _lv_hand_feats(records, tc, "rh"), _lv_hand_feats(records, tc, "lh")
    allns = [r.note for r in records]
    dur = sum(max(1, n.end - n.start) for n in allns) or 1
    black = sum(max(1, n.end - n.start) for n in allns if n.pitch % 12 in (1, 3, 6, 8, 10)) / dur
    from_rh = [r.note for r in records if r.hand == "rh"]
    from_lh = [r.note for r in records if r.hand == "lh"]
    peak = 0.0
    for ns in (from_rh, from_lh):
        ss = sorted(tc.sec(n.start) for n in ns)
        j = 0
        for i in range(len(ss)):
            while ss[i] - ss[j] > 2.0:
                j += 1
            peak = max(peak, (i - j + 1) / 2.0)
    vals = dict(rh_chord_max=rh["chord_max"], lh_chord_max=lh["chord_max"], rh_span_max=rh["span_max"],
                lh_span_max=lh["span_max"], rh_pitch_max=rh["pitch_max"], lh_pitch_min=lh["pitch_min"],
                rh_ioi_min=rh["ioi_min"], lh_ioi_min=lh["ioi_min"], peak_nps=peak, black=black)
    rows = []
    for k, (p10, p50, p90) in ref.items():
        v = vals[k]
        rows.append((k, v, p10, p50, p90, p10 - 1e-9 <= v <= p90 + 1e-9))
    return rows


def _lv_crop(records, tc, start_bar, n_bars):
    """Recorta a [start_bar, start_bar+n_bars) conservando los ticks originales
    (el desplazamiento a 0 se hace al exportar, ver _lv_export)."""
    t0 = tc.bar_range_ticks(start_bar)[0]
    t1 = tc.bar_range_ticks(start_bar + n_bars)[0]
    out = []
    for r in records:
        n = r.note
        if t0 <= n.start < t1:
            n.end = min(n.end, t1)
            out.append(r)
    return out, t0, t1


def _lv_shift_map(m, shift):
    if not shift:
        return list(m)
    before = [e for e in m if e[0] <= shift]
    after = [e for e in m if e[0] > shift]
    out = [(0,) + tuple(before[-1][1:])] if before else []
    out += [(e[0] - shift,) + tuple(e[1:]) for e in after]
    return out or list(m)[:1]


def _lv_export(records, mid, out_path, shift=0):
    import dataclasses as _dc
    if shift:
        recs = _lv_clone(records)
        for r in recs:
            r.note.start -= shift
            r.note.end -= shift
    else:
        recs = records
    export_grade_midi(recs, mid.tpb, out_path, _lv_shift_map(mid.tempo_map, shift),
                      _lv_shift_map(mid.timesig_map, shift))


def level_simplify(midi_path, levels, target_grade=None, floor_level="ursatz", split=60,
                   transpose=False, auto_meter=True, excerpt=None, excerpt_start=None,
                   use_fallback=True, max_transform_passes=8):
    """Deriva una version por nivel. `levels`: lista de beginner/intermediate/advanced.
    `target_grade`: si se da, sustituye el tope de grado del perfil.
    `excerpt`: None | 'auto' | nº de compases (extracto contiguo, como hacen la
    mayoria de versiones beginner reales). `transpose`: opt-in (en el corpus la
    transposicion automatica coincide menos con la real que no transponer)."""
    import inspect as _inspect
    params = _inspect.signature(reduce_graded).parameters
    kw = dict(floor_level=floor_level, split=split)
    if "pre_simplify" in params:
        kw["pre_simplify"] = True
    if "auto_meter" in params:
        kw["auto_meter"] = auto_meter
    if "auto_transpose" in params:
        kw["auto_transpose"] = bool(transpose)
    # grado 8 = nada que eliminar: nos da la superficie completa con roles/pesos
    result = reduce_graded(midi_path, [8], **kw)
    tc, mid = result["tc"], result["mid"]
    surface = result["level_snapshots"][8]
    n_original = len(surface)

    out_levels = {}
    for lv in [resolve_level(x) for x in levels]:
        prof = LEVEL_PROFILES[lv]
        shift, ex_info = 0, None
        recs = _lv_clone(surface)
        if excerpt:
            nb = max(1, result["n_bars"])
            first = excerpt_start or min((r.bar for r in recs), default=1)
            if excerpt == "auto":
                frac = prof.get("excerpt_frac")
                n_ex = nb if not frac else int(max(prof["excerpt_min"], min(prof["excerpt_max"], round(nb * frac))))
            else:
                n_ex = int(excerpt)
            if n_ex < nb - first + 1:
                recs, t0, t1 = _lv_crop(recs, tc, first, n_ex)
                shift = t0
                ex_info = (first, first + n_ex - 1)
        stats = {}
        derived, stats = derive_level_records(recs, tc, lv, stats)
        tgt = target_grade if target_grade else prof["target_grade"]
        rh = [r.note for r in derived if r.hand == "rh"]
        lh = [r.note for r in derived if r.hand == "lh"]
        grade, _ = whole_piece_grade(rh, lh, tc) if derived else (1, 0.0)
        method = f"nivel:{lv}"
        if grade > tgt and use_fallback and derived:
            reduced, g2, ok, steps = apply_grid_fallback_until_target(derived, tc, mid.tpb, tgt)
            derived, grade = reduced, g2
            method += "+rejilla" if ok else f"+rejilla (limite tras {steps} peldaños)"
        out_levels[lv] = LevelResult(tgt, grade, method, len(derived), n_original, derived)
        out_levels[lv].stats = stats
        out_levels[lv].shift = shift
        out_levels[lv].excerpt = ex_info
    return {"mid": mid, "tc": tc, "levels": out_levels, "result": result,
            "shift": 0, "excerpt": None}


def print_level_report(midi_path, out):
    tc = out["tc"]
    print(f"\n{'═' * 78}\n  NIVELES DERIVADOS — {midi_path}\n{'═' * 78}")
    print(f"  grado original estimado: {out['result']['grade_before_piece']}/8")
    for lv, res in out["levels"].items():
        pct = 100.0 * res.n_notes / max(1, res.n_notes_original)
        flag = "" if res.achieved_grade <= res.target_grade else "  [AVISO: grado por encima del tope]"
        ex = getattr(res, "excerpt", None)
        if ex:
            print(f"\n  [{lv}] extracto: compases {ex[0]}-{ex[1]}")
        print(f"\n  {lv}: grado {res.achieved_grade}/8 (tope {res.target_grade})  notas={res.n_notes}/{res.n_notes_original} ({pct:.0f}%)  metodo={res.method}{flag}")
        s = getattr(res, "stats", {})
        if s:
            print("     " + "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in s.items()))
        rows = level_profile_report(res.records, tc, lv)
        ok = sum(1 for r in rows if r[5])
        print(f"     perfil real del nivel (p10-p90 del corpus): {ok}/{len(rows)} rasgos dentro")
        for k, v, p10, p50, p90, good in rows:
            print(f"       {'✔' if good else '✘'} {k:13s} {v:7.2f}   [{p10:g} – {p90:g}]  (mediana {p50:g})")
    print(f"{'═' * 78}\n")


def add_level_arguments(ap):
    g = ap.add_argument_group("derivacion por nivel (v5)")
    g.add_argument("--level", nargs="+", metavar="L",
                   help="deriva versiones beginner / intermediate / advanced con el perfil aprendido del "
                        "corpus (en vez de --target-grade). Salida: <nombre>_<nivel>.mid")
    g.add_argument("--excerpt", metavar="N|auto",
                   help="[con --level] recorta un extracto contiguo: nº de compases o 'auto' (beginner "
                        "≈40%% de la pieza, 8-32 compases; los demas niveles no recortan)")
    g.add_argument("--excerpt-start", type=int, default=None, metavar="BAR",
                   help="[con --excerpt] compas inicial del extracto (defecto: primer compas con notas)")
    g.add_argument("--legacy", action="store_true",
                   help="usar el motor clasico tambien para --target-grade 1..5 (por defecto esos grados se "
                        "derivan con el perfil de nivel: 1-3 -> beginner, 4-5 -> intermediate; el motor "
                        "clasico solo alcanzaba grado 3 en el 2%% de las piezas del corpus)")
    g.add_argument("--transpose", action="store_true",
                   help="transportar a la tonalidad con menos teclas negras (OPT-IN desde v5: en el corpus "
                        "coincide menos con la transposicion real que no transponer)")
    return g


def level_main(args):
    out = level_simplify(args.midi, args.level, target_grade=(args.target_grade[0] if getattr(args, "target_grade", None) else None),
                         floor_level=args.floor_level, split=args.split, transpose=getattr(args, "transpose", False),
                         auto_meter=not getattr(args, "no_meter_detect", False),
                         excerpt=(int(args.excerpt) if args.excerpt and str(args.excerpt).isdigit() else args.excerpt),
                         excerpt_start=args.excerpt_start)
    print_level_report(args.midi, out)
    outdir = Path(args.outdir) if args.outdir else Path(args.midi).parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.midi).stem
    print("  Ficheros generados:")
    for lv, res in out["levels"].items():
        p = outdir / f"{stem}_{lv}.mid"
        _lv_export(res.records, out["mid"], str(p), getattr(res, "shift", 0))
        print(f"    · {p}  ({res.method})")
    print()
    return 0


def level_for_grade(t):
    return "beginner" if t <= 3 else ("intermediate" if t <= 5 else "advanced")


def grade_main(args, grades):
    """--target-grade bajos (<=5) via perfil de nivel; salida <nombre>_grade<N>.mid."""
    outdir = Path(args.outdir) if args.outdir else Path(args.midi).parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.midi).stem
    print("  [v5] --target-grade <= 5 derivado con perfil de nivel (usa --legacy para el motor clasico)")
    for t in sorted(set(grades), reverse=True):
        lv = level_for_grade(t)
        out = level_simplify(args.midi, [lv], target_grade=t, floor_level=args.floor_level, split=args.split,
                             transpose=getattr(args, "transpose", False),
                             auto_meter=not getattr(args, "no_meter_detect", False))
        print_level_report(args.midi, out)
        res = out["levels"][lv]
        p = outdir / f"{stem}_grade{t}.mid"
        _lv_export(res.records, out["mid"], str(p), getattr(res, "shift", 0))
        print(f"    · {p}  (grado {res.achieved_grade}/8; {res.method})")
    return 0


def main():
    ap = argparse.ArgumentParser(
        prog="combined_simplifier_transformed.py",
        description=(
            "Simplifica un MIDI de piano usando el motor estructural original "
            "más una capa de transformaciones musicales (cambio de registro, "
            "arpegiado, alineación de manos, reducción de saltos, etc.)."
        ),
    )

    ap.add_argument("midi")
    ap.add_argument("--target-grade", type=int, nargs="+", required=False)
    ap.add_argument("--floor-level", default="ursatz")
    ap.add_argument("--split", type=int, default=60)
    ap.add_argument("--outdir")
    ap.add_argument(
        "--transform-mode",
        choices=["off", "conservative", "balanced", "rewrite"],
        default="balanced",
    )
    ap.add_argument(
        "--allow-added-notes",
        action="store_true",
        help="Permite transformaciones que añaden notas de paso.",
    )
    ap.add_argument(
        "--max-transform-passes",
        type=int,
        default=8,
        help="Número máximo de pasadas de transformaciones.",
    )

    add_level_arguments(ap)
    args = ap.parse_args()
    if args.level:
        return level_main(args)
    if args.target_grade and not getattr(args, 'legacy', False) and not getattr(args, 'report_bars', False):
        _low = [t for t in args.target_grade if 1 <= t <= 5]
        if _low:
            grade_main(args, _low)
            args.target_grade = [t for t in args.target_grade if t not in _low]
            if not args.target_grade:
                return 0
    if not args.target_grade and not getattr(args, 'schenker_sequence', None):
        ap.error('hace falta --target-grade N [N ...] o --level L [L ...]')

    for g in args.target_grade:
        if not (1 <= g <= 8):
            print(f"[ERROR] --target-grade debe estar entre 1 y 8 (recibido: {g})", file=sys.stderr)
            return 1

    out = combined_simplify_transformed(
        args.midi,
        args.target_grade,
        floor_level=args.floor_level,
        split=args.split,
        transform_mode=args.transform_mode,
        allow_added_notes=args.allow_added_notes,
        max_transform_passes=args.max_transform_passes,
    )

    print_combined_report_transformed(args.midi, out)

    outdir = Path(args.outdir) if args.outdir else Path(args.midi).parent
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.midi).stem
    mid = out["mid"]

    print("  Ficheros generados:")
    for t in sorted(out["levels"], reverse=True):
        lv = out["levels"][t]
        out_path = outdir / f"{stem}_grade{t}.mid"
        export_grade_midi(lv.records, mid.tpb, str(out_path), mid.tempo_map, mid.timesig_map)
        print(f"    · {out_path}  ({lv.method})")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())