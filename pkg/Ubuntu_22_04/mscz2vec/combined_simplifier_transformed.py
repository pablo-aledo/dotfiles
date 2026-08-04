#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                COMBINED SIMPLIFIER TRANSFORMED v1.2                         ║
║                                                                              ║
║  Extensión del programa original combined_simplifier.py.                    ║
║                                                                              ║
║  Añade una capa de transformaciones musicales que no se limita a eliminar   ║
║  notas:                                                                      ║
║   - fusión de notas repetidas                                                 ║
║   - cuantización rítmica suave                                                ║
║   - alineación de manos para reducir independencia                            ║
║   - reducción de extensión de acordes                                         ║
║   - reducción de saltos                                                       ║
║   - arpegiado de acordes densos                                               ║
║   - simplificación de acordes                                                 ║
║   - adición opcional de notas de paso                                         ║
║                                                                              ║
║  Uso:                                                                        ║
║    python combined_simplifier_transformed.py obra.mid --target-grade 1 3 5  ║
║    python combined_simplifier_transformed.py obra.mid --target-grade 2      ║
║           --transform-mode rewrite --allow-added-notes                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
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
    ap.add_argument("--target-grade", type=int, nargs="+", required=True)
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

    args = ap.parse_args()

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