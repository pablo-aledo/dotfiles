#!/usr/bin/env python3
"""
orchestral_atlas.py — Orchestral Atlas: atlas de orquestación consultable
(instrumentos, registros, roles, combinaciones y reglas de equilibrio)
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
El ecosistema genera orquestaciones (orchestrator, piano_to_orchestra...) y
las puntúa (quality_scorer, playability_auditor), pero ninguna herramienta
EXPLICA por qué un timbre o un registro funciona. Este atlas es el material
de estudio: un catálogo curado, en el formato de chord_table / motif_table,
con lo que cualquier tratado de orquestación (Rimski-Kórsakov, Adler, Piston)
llama «carácter de registro»:

  · ZONAS DE REGISTRO por instrumento (grave / medio / agudo...) con su
    carácter tímbrico y su potencia de proyección (1 débil · 3 penetrante).
  · ROLES (melodía, contramelodía, armonía, bajo, ritmo, textura) con una
    puntuación de idoneidad 0-3 por instrumento.
  · COMBINACIONES de instrumentos (unísono, octava, capas) con su efecto,
    cuándo usarlas y sus precauciones.
  · REGLAS de equilibrio y de escritura (enmascaramiento, espaciado, etc.).

Es la fuente de datos que reutilizan orchestra_critic, orchestra_anatomy,
orchestra_compare, doubling_lab y library_map. Las claves de instrumento son
las de INSTRUMENT_DB de playability_auditor.py (violin, oboe, trompa...).

LIMITACIONES: es conocimiento curado y simplificado: sirve para decidir y
estudiar, no sustituye al tratado ni al curso. La «proyección» es una
heurística de equilibrio sobre MIDI; no mide el timbre real de tus librerías
de samples (calibra siempre por oído). Los registros se expresan como alturas
MIDI SONORAS (C4 = 60).

COMO MÓDULO:
  from orchestral_atlas import ATLAS, zone_of, projection, best_for_role,
                               find_combos, RULES, parse_pitch, instrument_key

DEPENDENCIAS: solo stdlib · playability_auditor.py (opcional: catálogo de
instrumentos y nombres de pista)

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  list      Instrumentos del atlas (filtra por familia o texto)
  show      Ficha de un instrumento: zonas, roles, virtudes y cautelas
  zone      En qué zona cae una nota (o varias) de un instrumento
  combos    Combinaciones recomendadas por rol, registro, instrumento o dinámica
  role      Qué instrumentos encajan mejor en un rol (y registro)
  compare   Dos instrumentos cara a cara
  rules     Reglas de equilibrio y escritura orquestal
  selftest  Comprueba la coherencia interna del catálogo

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  list      [--family F] [--grep TXT]
  show      INSTRUMENTO
  zone      INSTRUMENTO NOTA [NOTA ...]      (C4, F#3, 60...)
  combos    [--role R] [--register low|mid|high] [--with INSTR]
            [--dynamic soft|mid|loud] [--top N]
  role      ROL [--pitch NOTA | --register low|mid|high] [--top N]
  compare   A B [--pitch NOTA]
  rules     [--grep TXT]
  selftest

OPCIONES COMUNES:
  --json        Salida en JSON en vez de texto
  --no-color    Desactivar colores ANSI

CÓDIGOS DE SALIDA:  0 ok · 1 error de ejecución · 2 uso incorrecto
                    (instrumento/nota/rol desconocido) · 3 selftest fallido

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
Una línea por parámetro: <nombre> <tipo> [clave=valor …] [# descripción].
Tipos: int float str bool enum file dir list kv. Claves: default min max typical
unit values of io ext format required advanced when excludes alias. Sin min/max =
sin límite conocido. Se verifica contra el argparse con «suite_index.py lint
--check spec-vs-code».

@params v1
[*]
--json     bool  default=false  # Salida en JSON
--no-color bool  default=false
[list]
--family enum  values=cuerda_frotada|cuerda_pulsada|viento_madera|viento_metal|teclado|percusion_afinada|percusion|voz
--grep   str   # Filtra por texto
[show]
instrument str  required  # Clave o nombre (oboe, violonchelo, "corno inglés"...)
[zone]
instrument str  required
notes      list of=str required  # Notas: C4, F#3 o número MIDI
[combos]
--role     enum  values=melody|countermelody|harmony|bass|rhythm|texture
--register enum  values=low|mid|high
--with     str   # Combinaciones que incluyan este instrumento
--dynamic  enum  values=soft|mid|loud
--top      int   default=10 min=1
[role]
role       enum  values=melody|countermelody|harmony|bass|rhythm|texture required
--pitch    str   excludes=--register  # Nota (C4, 60)
--register enum  values=low|mid|high excludes=--pitch
--top      int   default=8 min=1
[compare]
a       str  required
b       str  required
--pitch str   # Nota en la que comparar
[rules]
--grep str   # Filtra por texto
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  python orchestral_atlas.py list --family viento_madera
  python orchestral_atlas.py show oboe
  python orchestral_atlas.py zone flauta C4 C5 C6 C7
  python orchestral_atlas.py combos --role melody --register mid --dynamic soft
  python orchestral_atlas.py combos --with violonchelo
  python orchestral_atlas.py role bass --register low
  python orchestral_atlas.py role melody --pitch G4
  python orchestral_atlas.py compare clarinete oboe --pitch C5
  python orchestral_atlas.py rules --grep enmascar
"""

import re
import sys
import json
import argparse
from typing import Dict, List, Optional

try:
    from playability_auditor import INSTRUMENT_DB, name_to_profile
except ImportError:                       # el atlas funciona también sin el auditor
    INSTRUMENT_DB = {}

    def name_to_profile(name):            # type: ignore
        return None

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ROLES = ["melody", "countermelody", "harmony", "bass", "rhythm", "texture"]
ROLE_ES = {"melody": "melodía", "countermelody": "contramelodía", "harmony": "armonía",
           "bass": "bajo", "rhythm": "ritmo", "texture": "textura/color"}
REGISTERS = ["low", "mid", "high"]
REGISTER_ES = {"low": "grave", "mid": "medio", "high": "agudo"}
DYNAMICS = ["soft", "mid", "loud"]
DYNAMIC_ES = {"soft": "suave (pp-mp)", "mid": "media (mp-f)", "loud": "fuerte (f-ff)"}
FAMILY_ES = {"cuerda_frotada": "cuerda frotada", "cuerda_pulsada": "cuerda pulsada",
             "viento_madera": "viento madera", "viento_metal": "viento metal",
             "teclado": "teclado", "percusion_afinada": "percusión afinada",
             "percusion": "percusión", "voz": "voz"}


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def pitch_name(p: int) -> str:
    return f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"


# ══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGO DE INSTRUMENTOS
# ══════════════════════════════════════════════════════════════════════════════
#  zones: (lo, hi, id, etiqueta, carácter, potencia 1-3) — contiguas y cubren
#         exactamente el rango físico de INSTRUMENT_DB.
#  roles: {rol: (idoneidad 0-3, nota)}
#  base:  proyección base del instrumento (0-1), factor de equilibrio heurístico.

ATLAS: Dict[str, dict] = {}


def _inst(key, es, family, base, zones, roles, strengths, cautions, dynamics, articulation):
    ATLAS[key] = dict(
        key=key, es=es, family=family, base=base,
        zones=[dict(lo=z[0], hi=z[1], id=z[2], label=z[3], character=z[4], power=z[5])
               for z in zones],
        roles={r: dict(score=v[0], note=v[1]) for r, v in roles.items()},
        strengths=strengths, cautions=cautions,
        dynamics=dynamics, articulation=articulation)


# ── CUERDA FROTADA ───────────────────────────────────────────────────────────
_inst("violin", "Violín", "cuerda_frotada", 0.60, [
    (55, 66, "g", "Grave (cuerda de Sol)", "oscuro, denso, casi de viola; potente con arco a fondo", 2),
    (67, 83, "medio", "Medio (cuerdas Re-La)", "el registro cantábile por excelencia: cálido, flexible y expresivo", 3),
    (84, 96, "agudo", "Agudo (cuerda de Mi)", "brillante, luminoso y penetrante; ideal para clímax líricos", 3),
    (97, 105, "sobreagudo", "Sobreagudo", "muy fino, tenso y casi etéreo; exige intonación y suena frágil", 2)],
    {"melody": (3, "voz melódica natural de la orquesta"),
     "countermelody": (3, "segundos violines o violines en registro medio"),
     "harmony": (3, "acordes sostenidos, divisi y pads"),
     "bass": (0, "fuera de su función"),
     "rhythm": (2, "ostinatos, spiccato y tremolo"),
     "texture": (3, "tremolo, armónicos, trinos, pizzicato")},
    ["cantabilidad y flexibilidad dinámica", "brillo en el agudo sin perder calidez",
     "gran variedad de articulaciones"],
    ["en solo tiene poca masa: la sección (cuerdas) suena más llena",
     "pasajes muy rápidos en el grave suenan emborronados"],
    "regulable de pp a ff en todo el rango; el agudo extremo es difícil en pp",
    "legato, détaché, spiccato, staccato, pizzicato, tremolo, con sordina, armónicos")

_inst("viola", "Viola", "cuerda_frotada", 0.50, [
    (48, 59, "c", "Grave (cuerda de Do)", "oscuro, nasal y penetrante; carácter propio inconfundible", 2),
    (60, 75, "medio", "Medio", "cálido, velado y melancólico; ideal para voces interiores", 2),
    (76, 88, "agudo", "Agudo", "tenso y más fino; pierde el carácter de viola y se parece a un violín débil", 1)],
    {"melody": (2, "melodías cálidas y sombrías en registro medio-grave"),
     "countermelody": (3, "voz interior expresiva"),
     "harmony": (3, "relleno armónico por excelencia"),
     "bass": (0, "fuera de su función"),
     "rhythm": (2, "figuraciones internas y ostinatos"),
     "texture": (2, "tremolo y sul ponticello oscuros")},
    ["color oscuro y cálido", "voz interior con carácter"],
    ["en el agudo queda detrás de los violines", "fácil de enterrar bajo viento o metal"],
    "más contenida que el violín; no proyecta tanto",
    "legato, détaché, spiccato, pizzicato, tremolo, con sordina")

_inst("violonchelo", "Violonchelo", "cuerda_frotada", 0.60, [
    (36, 47, "c", "Grave (cuerda de Do)", "potente y profundo; excelente como bajo, turbio en acordes cerrados", 2),
    (48, 62, "medio_grave", "Medio-grave", "cálido, redondo y lleno; ideal para contramelodías", 2),
    (63, 75, "tenor", "Tenor (cuerda de La)", "el registro más cantábile y expresivo, casi vocal", 3),
    (76, 84, "agudo", "Agudo", "tenso y apasionado; más fino y difícil", 2)],
    {"melody": (3, "melodías líricas y nobles en registro tenor"),
     "countermelody": (3, "voz grave expresiva"),
     "harmony": (2, "acordes de registro medio-grave"),
     "bass": (3, "bajo con calidez y cantabilidad"),
     "rhythm": (2, "ostinatos y pizzicato"),
     "texture": (2, "tremolo, armónicos, sul ponticello")},
    ["calidez vocal en registro tenor", "doble función: bajo y melodía"],
    ["acordes cerrados por debajo de C3 suenan turbios",
     "a ff el agudo se tensa y pierde redondez"],
    "muy flexible; el pp en el grave es muy bello",
    "legato, détaché, spiccato, pizzicato, tremolo, con sordina")

_inst("contrabajo", "Contrabajo", "cuerda_frotada", 0.45, [
    (28, 40, "grave", "Grave", "cimiento profundo; pierde definición con notas rápidas", 2),
    (41, 55, "medio", "Medio", "el registro más claro y articulado", 2),
    (56, 67, "agudo", "Agudo / armónicos", "tenso y poco potente; armónicos naturales sutiles", 1)],
    {"melody": (0, "solo en pasajes especiales"),
     "countermelody": (0, "no es su función"),
     "harmony": (1, "solo como fundamento de la armonía"),
     "bass": (3, "cimiento de la orquesta, a menudo con chelos a la octava"),
     "rhythm": (2, "ostinatos graves y pizzicato"),
     "texture": (1, "tremolo grave y armónicos")},
    ["profundidad y peso", "con chelos a la octava da un bajo sólido"],
    ["notas rápidas en el grave emborronan", "sin ataque (fagot, timbal) el bajo queda borroso"],
    "poco proyectado en pp; en ff el grave es muy poderoso",
    "arco largo, spiccato, pizzicato (con rebote), tremolo")

_inst("cuerdas", "Sección de cuerdas", "cuerda_frotada", 0.85, [
    (28, 47, "grave", "Cuerdas graves (chelos y contrabajos)", "cimiento cálido y sólido", 2),
    (48, 71, "medio", "Registro medio (violas, chelos, violines graves)", "el corazón de la sección: lleno, rico y cálido", 3),
    (72, 88, "agudo", "Agudo (violines)", "brillante y cantábile; la melodía natural de la orquesta", 3),
    (89, 105, "sobreagudo", "Sobreagudo", "etéreo, fino y luminoso; delicado en pp, penetrante en ff", 2)],
    {"melody": (3, "violines en registro medio-agudo"),
     "countermelody": (3, "violas y chelos"),
     "harmony": (3, "colchón armónico sostenido"),
     "bass": (3, "chelos y contrabajos"),
     "rhythm": (3, "ostinatos, spiccato, pizzicato"),
     "texture": (3, "tremolo, armónicos, trinos")},
    ["gama dinámica y de articulación completa", "capaz de sostener sin respirar"],
    ["a ff pueden quedar cubiertas por los metales", "muy fáciles de sobreescribir: no hagas que toquen siempre"],
    "de ppp a fff; la sección compensa el equilibrio con otras familias",
    "todas las del violín; con sordina, divisi, sul tasto/ponticello")

# ── MADERA ───────────────────────────────────────────────────────────────────
_inst("flautin", "Flautín", "viento_madera", 0.80, [
    (74, 83, "grave", "Grave", "débil, soplado y poco característico", 1),
    (84, 98, "medio", "Medio", "brillante y claro; refuerza la octava superior", 3),
    (99, 108, "agudo", "Agudo", "agudo y penetrante; corta cualquier textura", 3)],
    {"melody": (1, "solo en pasajes especiales y fuertes"),
     "countermelody": (0, "no es su función"),
     "harmony": (0, "no es su función"),
     "bass": (0, "fuera de su función"),
     "rhythm": (1, "figuras agudas y staccato"),
     "texture": (3, "brillo y centelleo en el sobreagudo")},
    ["suma brillo y chispa", "se oye a través de toda la orquesta"],
    ["cansa si se usa mucho", "el grave casi no suena"],
    "potente a f; el pp en el sobreagudo es muy difícil",
    "legato, staccato, trinos, picado doble")

_inst("flauta", "Flauta", "viento_madera", 0.55, [
    (60, 71, "grave", "Grave", "soplado, hueco y suave; queda tapado con facilidad", 1),
    (72, 84, "medio", "Medio", "cálido, redondo y claro; el centro de la flauta", 2),
    (85, 98, "agudo", "Agudo", "brillante y penetrante; suena bien a f, difícil a pp", 3)],
    {"melody": (3, "solista en registro medio-agudo sobre acompañamiento ligero"),
     "countermelody": (2, "línea ágil por encima de las cuerdas"),
     "harmony": (1, "solo en conjunto de maderas"),
     "bass": (0, "fuera de su función"),
     "rhythm": (1, "figuración rápida y staccato"),
     "texture": (3, "trinos, arpegios y adornos en el agudo")},
    ["agilidad y ligereza", "color claro y transparente"],
    ["en el grave (hasta B4) se oye poco: no la pongas bajo cuerdas f",
     "las notas largas en el agudo en pp son arriesgadas"],
    "fuerza dinámica muy dependiente del registro: débil en el grave, brillante en el agudo",
    "legato, staccato, picado doble/triple, trinos, frullato")

_inst("oboe", "Oboe", "viento_madera", 0.80, [
    (58, 64, "grave", "Grave", "áspero, rústico y algo nasal", 2),
    (65, 84, "medio", "Medio", "expresivo, penetrante y melancólico; el sonido nasal clásico", 3),
    (85, 93, "agudo", "Agudo", "fino y tenso; exige control", 2)],
    {"melody": (3, "solista penetrante: se oye bien sobre cuerdas"),
     "countermelody": (3, "voz expresiva"),
     "harmony": (1, "solo en conjunto de maderas"),
     "bass": (0, "fuera de su función"),
     "rhythm": (1, "staccato ligero"),
     "texture": (2, "color pastoral y melancólico")},
    ["timbre inconfundible y proyección", "gran expresividad en el registro medio"],
    ["el grave es áspero", "su color nasal domina si dobla otras líneas"],
    "buena proyección incluso en pp; difícil el crescendo en sobreagudo",
    "legato, staccato, trinos, picado")

_inst("corno_ingles", "Corno inglés", "viento_madera", 0.60, [
    (52, 59, "grave", "Grave", "oscuro, profundo y sombrío", 2),
    (60, 77, "medio", "Medio", "melancólico, nasal y evocador; caso típico de solo triste", 2),
    (78, 86, "agudo", "Agudo", "fino y tenso; pierde su carácter", 1)],
    {"melody": (3, "solo melancólico sobre un colchón suave"),
     "countermelody": (2, "voz sombría"),
     "harmony": (1, "solo en conjunto"),
     "bass": (0, "fuera de su función"),
     "rhythm": (0, "no es su función"),
     "texture": (2, "color nostálgico")},
    ["color melancólico muy reconocible"],
    ["queda tapado si el acompañamiento es denso", "movimientos rápidos poco naturales"],
    "dinámica media-baja; el ff es poco natural",
    "legato, staccato suave")

_inst("clarinete", "Clarinete", "viento_madera", 0.65, [
    (50, 67, "chalumeau", "Chalumeau (grave)", "oscuro, redondo y de madera; recuerda a la voz grave", 2),
    (68, 71, "garganta", "Notas de garganta", "las más débiles y apagadas: evítalas como solista", 1),
    (72, 83, "clarin", "Clarín (medio)", "brillante, claro y cantábile", 2),
    (84, 94, "agudo", "Agudo", "agudo y estridente; exige control", 2)],
    {"melody": (3, "gran cantábile en chalumeau y clarín"),
     "countermelody": (3, "línea oscura y agil"),
     "harmony": (2, "relleno de armonía en maderas"),
     "bass": (0, "salvo clarinete bajo (no incluido)"),
     "rhythm": (1, "staccato ágil"),
     "texture": (2, "arpegios y escalas rápidas")},
    ["la mayor gama dinámica de las maderas", "agilidad y cantabilidad"],
    ["las notas de garganta son débiles", "cambios bruscos de registro cambian mucho el color"],
    "de ppp a ff con facilidad; el pp es una de sus mejores armas",
    "legato, staccato, picado, trinos, glissando")

_inst("fagot", "Fagot", "viento_madera", 0.55, [
    (34, 45, "grave", "Grave", "pesado, zumbón y muy característico", 2),
    (46, 62, "medio", "Medio", "cálido y redondo; excelente bajo para maderas", 2),
    (63, 75, "agudo", "Agudo / tenor", "expresivo y plañidero; tenso en el límite", 1)],
    {"melody": (2, "melodía cómica o plañidera en registro tenor"),
     "countermelody": (2, "voz grave expresiva"),
     "harmony": (2, "relleno grave de las maderas"),
     "bass": (3, "bajo claro y articulado de las maderas"),
     "rhythm": (2, "staccato muy característico"),
     "texture": (1, "color")},
    ["buen ataque que define el bajo", "doble función: bajo y solista"],
    ["el agudo se tensa", "en pp grave puede ser poco audible"],
    "controlado en pp-mf; en ff puede sonar áspero",
    "legato, staccato, picado")

# ── METAL ────────────────────────────────────────────────────────────────────
_inst("trompeta", "Trompeta", "viento_metal", 1.00, [
    (54, 60, "grave", "Grave", "oscuro, poco seguro y poco brillante", 2),
    (61, 77, "medio", "Medio", "brillante, heroico y claro: fanfarria típica", 3),
    (78, 86, "agudo", "Agudo", "penetrante y cansado; exige resistencia", 3)],
    {"melody": (3, "melodía heroica o fanfarria; dominante"),
     "countermelody": (2, "con sordina, discreta"),
     "harmony": (2, "acordes de metales"),
     "bass": (0, "fuera de su función"),
     "rhythm": (2, "figuras marcadas y staccato"),
     "texture": (2, "notas largas y acentos")},
    ["brillo y autoridad", "se oye por encima de toda la orquesta"],
    ["cubre con facilidad a cuerdas y maderas", "el pp en el agudo es difícil"],
    "potente: de pp a ff pero se impone a partir de mf",
    "legato, staccato, marcato, con sordina, frullato")

_inst("trompa", "Trompa", "viento_metal", 0.85, [
    (35, 47, "grave", "Grave", "oscuro, profundo y algo sombrío", 2),
    (48, 72, "medio", "Medio (zona noble)", "cálido, noble y redondo; une maderas y metales", 3),
    (73, 77, "agudo", "Agudo", "tenso y arriesgado", 2)],
    {"melody": (3, "melodía noble y heroica sobre cuerdas"),
     "countermelody": (3, "voz cálida y llena"),
     "harmony": (3, "pad cálido y puente entre maderas y metales"),
     "bass": (1, "solo en registro grave y como refuerzo"),
     "rhythm": (2, "figuras de caza y llamadas"),
     "texture": (2, "stopped/con sordina, notas largas")},
    ["mezcla bien con maderas y con cuerdas", "gran gama dinámica"],
    ["puede cubrir a las cuerdas en ff", "melodías muy rápidas son poco naturales"],
    "de pp (suave y cálido) a ff (heroico)",
    "legato, staccato, stopped, con sordina, marcato")

_inst("trombon", "Trombón", "viento_metal", 1.00, [
    (40, 48, "grave", "Grave", "poderoso y oscuro", 3),
    (49, 67, "medio", "Medio", "noble, lleno y solemne", 3),
    (68, 77, "agudo", "Agudo", "brillante y tenso", 3)],
    {"melody": (2, "melodías solemnes y corales"),
     "countermelody": (1, "voz grave"),
     "harmony": (3, "coral solemne de tres trombones"),
     "bass": (2, "refuerzo del bajo en tutti"),
     "rhythm": (2, "acentos y sforzandi"),
     "texture": (2, "glissando y notas largas")},
    ["autoridad y solemnidad", "en grupo, coral muy sólido"],
    ["cubre a las maderas y a las cuerdas", "el staccato rápido es limitado"],
    "controlado de pp a ff; el pp es hermoso y profundo",
    "legato, staccato, marcato, glissando, con sordina")

_inst("tuba", "Tuba", "viento_metal", 0.90, [
    (26, 40, "grave", "Grave", "profundo y cimiento absoluto del metal", 3),
    (41, 52, "medio", "Medio", "lleno y redondo", 3),
    (53, 65, "agudo", "Agudo", "plañidero y tenso", 2)],
    {"melody": (1, "solo cómico o plañidero"),
     "countermelody": (0, "no es su función"),
     "harmony": (1, "fundamento del grupo de metales"),
     "bass": (3, "bajo del metal; refuerza trombones y contrabajos"),
     "rhythm": (1, "notas largas y acentos"),
     "texture": (1, "color grave")},
    ["bajo de los metales", "peso y profundidad"],
    ["movimiento muy rápido poco natural", "puede enturbiar si dobla al contrabajo sin definición"],
    "potente pero flexible; el pp grave es sorprendente",
    "legato, staccato, marcato")

_inst("viento_metal_gen", "Sección de metales", "viento_metal", 1.00, [
    (35, 52, "grave", "Graves (trombones, tuba, trompas)", "oscuro, solemne y poderoso", 3),
    (53, 72, "medio", "Medio", "pleno, noble y brillante", 3),
    (73, 86, "agudo", "Agudo (trompetas)", "brillante y penetrante", 3)],
    {"melody": (3, "heroica y llena"),
     "countermelody": (2, "con sordina"),
     "harmony": (3, "corales solemnes"),
     "bass": (2, "tuba y trombones bajos"),
     "rhythm": (2, "acentos marcados"),
     "texture": (2, "notas largas en crescendo")},
    ["potencia, brillo y autoridad", "tutti fuerte"],
    ["cubren todo a igual dinámica", "se cansan rápido"],
    "gran gama pero domina a partir de mf",
    "legato, marcato, staccato, con sordina")

# ── OTROS ────────────────────────────────────────────────────────────────────
_inst("arpa", "Arpa", "cuerda_pulsada", 0.35, [
    (24, 47, "grave", "Grave", "resonante, profundo y con mucha reverberación natural", 2),
    (48, 84, "medio", "Medio", "claro, brillante y de ataque limpio", 2),
    (85, 103, "agudo", "Agudo", "delicado, cristalino y de resonancia corta", 1)],
    {"melody": (1, "frases breves y de ataque muy marcado"),
     "countermelody": (1, "adornos"),
     "harmony": (2, "arpegios y acordes"),
     "bass": (1, "pedales de grave"),
     "rhythm": (2, "ostinatos"),
     "texture": (3, "glissandi, arpegios y armónicos")},
    ["transparente y delicado", "se oye aunque sea suave"],
    ["no sostiene las notas largas", "queda tapada por metales"],
    "poca dinámica máxima: de ppp a mf",
    "pluck, glissando, armónicos, bisbigliando")

_inst("celesta", "Celesta", "teclado", 0.35, [
    (60, 83, "medio", "Medio", "tintineante, dulce y mágico", 2),
    (84, 108, "agudo", "Agudo", "cristalino y fino; poca proyección", 1)],
    {"melody": (2, "melodía dulce y sencilla"),
     "countermelody": (0, "no es su función"),
     "harmony": (1, "acordes breves"),
     "bass": (0, "fuera de su función"),
     "rhythm": (1, "ostinatos suaves"),
     "texture": (3, "centelleo y magia")},
    ["color mágico e infantil"],
    ["poca proyección; se oye solo sobre texturas suaves"],
    "de pp a mf",
    "notas breves, pedal de resonancia")

_inst("piano", "Piano", "teclado", 0.70, [
    (21, 47, "grave", "Grave", "profundo, resonante y poderoso", 2),
    (48, 71, "medio", "Medio", "claro, lleno y neutro", 2),
    (72, 96, "agudo", "Agudo", "brillante y percusivo", 2),
    (97, 108, "sobreagudo", "Sobreagudo", "vidrioso y breve", 1)],
    {"melody": (2, "melodías de ataque nítido"),
     "countermelody": (2, "contrapunto claro"),
     "harmony": (3, "armonía completa"),
     "bass": (3, "bajo y armonía a la vez"),
     "rhythm": (3, "ostinatos y pulsos"),
     "texture": (2, "arpegios y adornos")},
    ["autosuficiente: melodía, armonía y bajo a la vez"],
    ["no sostiene el sonido: decae", "en orquesta puede sonar percusivo"],
    "amplia, pero decae rápido",
    "pedal, legato, staccato")

_inst("timbal", "Timbales", "percusion_afinada", 0.90, [
    (40, 60, "unico", "Rango afinable (D2-C4)", "profundo y resonante; da peso y pulso", 3)],
    {"melody": (0, "no es su función"), "countermelody": (0, "no es su función"),
     "harmony": (1, "fundamental y dominante afinados"),
     "bass": (2, "refuerzo de impacto del bajo"),
     "rhythm": (3, "pulso, redobles y acentos"),
     "texture": (2, "redobles suaves y trémolos")},
    ["pulso, impacto y dramatismo"],
    ["la afinación debe pertenecer a la armonía", "un redoble constante resulta cansado"],
    "de ppp a fff muy controlado",
    "golpes, redobles, sordina")

_inst("laminas", "Láminas (xilófono/marimba/vibráfono)", "percusion_afinada", 0.55, [
    (45, 71, "grave", "Grave", "cálido (marimba) y poco proyectado", 1),
    (72, 95, "medio", "Medio", "claro y percusivo", 2),
    (96, 108, "agudo", "Agudo", "brillante y seco", 2)],
    {"melody": (1, "frases cortas y percusivas"), "countermelody": (0, "no es su función"),
     "harmony": (1, "acordes breves"), "bass": (0, "fuera de su función"),
     "rhythm": (3, "ostinatos precisos"), "texture": (3, "brillo y color percusivo")},
    ["claridad rítmica y color brillante"],
    ["no sostienen notas largas"],
    "de pp a f",
    "baquetas duras/blandas, trémolo")

_inst("voz", "Voz / coro", "voz", 0.70, [
    (40, 55, "grave", "Grave (bajos)", "oscuro y sólido", 2),
    (56, 72, "medio", "Medio", "humano, cálido y directo", 3),
    (73, 84, "agudo", "Agudo", "brillante y emotivo", 3)],
    {"melody": (3, "la melodía más directa"), "countermelody": (2, "voz interior"),
     "harmony": (3, "coral y pad vocal"), "bass": (2, "bajos corales"),
     "rhythm": (1, "figuras sencillas"), "texture": (2, "ooh/aah como color")},
    ["conecta emocionalmente de forma inmediata"],
    ["necesita respirar", "el texto limita la agilidad"],
    "de pp a ff; el agudo a ff exige potencia",
    "legato, marcato, staccato, aah/ooh")

_inst("percusion", "Percusión (set/GM)", "percusion", 0.80, [
    (27, 87, "unico", "Rango GM", "sin altura definida; da ritmo y énfasis", 2)],
    {"melody": (0, "no es su función"), "countermelody": (0, "no es su función"),
     "harmony": (0, "no es su función"), "bass": (0, "no es su función"),
     "rhythm": (3, "pulso y acentos"), "texture": (2, "platillos y color")},
    ["ritmo y energía"],
    ["tapa las líneas finas si se escribe constante"],
    "amplia",
    "golpes, redobles, platillos suspendidos")


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINACIONES
# ══════════════════════════════════════════════════════════════════════════════
#  register: low|mid|high (donde funciona mejor) · dynamic: soft|mid|loud
#  blend: 0-3 (grado de fusión del timbre)

COMBOS: List[dict] = []


def _combo(cid, name, insts, relation, effect, use, caution, roles, register, dynamic, blend):
    COMBOS.append(dict(id=cid, name=name, insts=insts, relation=relation, effect=effect,
                       use=use, caution=caution, roles=roles, register=register,
                       dynamic=dynamic, blend=blend))


_combo("flauta_oboe", "Flauta + oboe", ["flauta", "oboe"], "unísono",
       "El oboe da cuerpo y mordiente a la flauta; la flauta suaviza el oboe.",
       "Melodía en registro medio-agudo con color claro y con foco.",
       "El oboe domina: baja su nivel unos 3-4 dB respecto a la flauta.",
       ["melody"], ["mid", "high"], ["soft", "mid"], 3)
_combo("flauta_clarinete", "Flauta + clarinete", ["flauta", "clarinete"], "octava/unísono",
       "Sonido hueco, sedoso y muy fundido.",
       "Melodía suave o contramelodía; la flauta arriba, el clarinete abajo.",
       "El clarinete puede quedar oculto en el registro de garganta (68-71).",
       ["melody", "countermelody"], ["mid"], ["soft"], 3)
_combo("oboe_clarinete", "Oboe + clarinete", ["oboe", "clarinete"], "unísono",
       "Mezcla clásica de maderas cálida y redonda.",
       "Melodías y respuestas en la sección de maderas.",
       "El oboe cubre al clarinete en el agudo: equilibra por oído.",
       ["melody", "countermelody"], ["mid"], ["soft", "mid"], 3)
_combo("oboe_fagot_2oct", "Oboe + fagot a dos octavas", ["oboe", "fagot"], "dos octavas",
       "Sonido hueco, pastoral y abierto (el timbre de «ventana abierta»).",
       "Contramelodía o armonía a dos voces con carácter arcaico.",
       "La octava doble deja un hueco armónico; rellena con una voz intermedia si hace falta.",
       ["countermelody", "harmony"], ["mid", "low"], ["soft"], 2)
_combo("violines_flauta_oct", "Violines + flauta una octava arriba", ["violin", "flauta"], "octava",
       "Brillo y centelleo sobre la melodía de cuerda.",
       "Melodía lírica con luminosidad extra.",
       "La flauta no debe bajar de C5; en el grave se pierde.",
       ["melody"], ["high"], ["soft", "mid"], 2)
_combo("cuerdas_oboe", "Cuerdas + oboe", ["cuerdas", "oboe"], "unísono",
       "Da foco y borde a una melodía de cuerdas.",
       "Melodía importante que debe oírse nítida en un tutti medio.",
       "Un oboe sobre toda la sección puede sonar duro: baja su nivel.",
       ["melody"], ["mid", "high"], ["mid"], 2)
_combo("chelos_trompas", "Violonchelos + trompas", ["violonchelo", "trompa"], "unísono/octava",
       "Calidez noble y romántica; el color de la melodía heroica lírica.",
       "Melodía o contramelodía amplia en el registro tenor.",
       "La trompa puede cubrir al chelo: déjala un nivel por debajo.",
       ["melody", "countermelody"], ["mid"], ["mid", "loud"], 3)
_combo("violas_clarinetes", "Violas + clarinetes", ["viola", "clarinete"], "unísono",
       "Color oscuro y sostenido.",
       "Armonía interior o contramelodía sombría.",
       "Si el clarinete cae en las notas de garganta, desaparece.",
       ["harmony", "countermelody"], ["mid", "low"], ["soft"], 3)
_combo("fagot_chelos_bajo", "Fagot + violonchelos + contrabajos", ["fagot", "violonchelo", "contrabajo"], "octavas",
       "Bajo definido: el fagot aporta ataque a la masa de las cuerdas.",
       "Líneas de bajo que necesitan claridad.",
       "En ff el fagot desaparece: añade tuba o trombón bajo.",
       ["bass"], ["low"], ["soft", "mid"], 3)
_combo("chelos_contrabajos_oct", "Violonchelos + contrabajos a la octava", ["violonchelo", "contrabajo"], "octava",
       "El cimiento clásico de la orquesta: sólido, cálido y definido.",
       "Bajo de cualquier textura de cuerdas.",
       "Sin ataque rítmico (fagot, timbal, pizzicato) el bajo queda borroso en notas rápidas.",
       ["bass"], ["low"], ["soft", "mid", "loud"], 3)
_combo("trompas_coral", "Trompas en armonía (pad)", ["trompa"], "acordes",
       "Colchón cálido que une maderas y metales.",
       "Notas largas de armonía bajo una melodía de maderas o cuerdas.",
       "Evita los extremos grave y agudo del registro.",
       ["harmony"], ["mid"], ["soft", "mid"], 3)
_combo("trombones_coral", "Trombones + trompas en coral", ["trombon", "trompa", "tuba"], "acordes",
       "Coral solemne y de gran peso.",
       "Momentos de ceremonia o clímax grave.",
       "Cuidado con acordes cerrados por debajo de C3: abre el espaciado.",
       ["harmony"], ["low", "mid"], ["mid", "loud"], 3)
_combo("trompeta_trombon", "Trompeta + trombón", ["trompeta", "trombon"], "octava/unísono",
       "Fanfarria brillante y potente.",
       "Llamadas, anuncios y clímax en metales.",
       "Cubren todo lo demás: reserva el resto de la orquesta para acompañar sin competir.",
       ["melody"], ["mid", "high"], ["loud"], 2)
_combo("pizz_fagot", "Cuerdas pizzicato + fagot staccato", ["cuerdas", "fagot"], "unísono",
       "Bajo ligero, definido y juguetón.",
       "Pasajes ligeros con carácter rítmico.",
       "Mantén el fagot en pp-mp: en f deja de fundirse.",
       ["bass", "rhythm"], ["low"], ["soft"], 3)
_combo("arpa_pizz", "Arpa + cuerdas en pizzicato", ["arpa", "cuerdas"], "capas",
       "Textura delicada y transparente.",
       "Acompañamiento suave con colores claros.",
       "Si los metales entran, el conjunto desaparece.",
       ["texture", "harmony"], ["mid"], ["soft"], 3)
_combo("celesta_flauta", "Celesta + flauta + arpa", ["celesta", "flauta", "arpa"], "capas",
       "Centelleo cristalino y mágico.",
       "Colores de fantasía o ensueño sobre un colchón suave.",
       "Poca proyección: solo con texturas ligeras.",
       ["texture"], ["high"], ["soft"], 3)
_combo("timbal_contrabajo", "Timbales + contrabajos", ["timbal", "contrabajo"], "unísono",
       "Bajo rítmico con ataque y peso.",
       "Pulso y tensión en pasajes dramáticos.",
       "Afina los timbales a la fundamental y a la dominante de la armonía.",
       ["rhythm", "bass"], ["low"], ["mid", "loud"], 3)
_combo("cuerdas_pad_flauta", "Flauta solista sobre colchón de cuerdas", ["flauta", "cuerdas"], "capas",
       "Un solista sobre un pad cálido: la textura más clara de la orquesta.",
       "Solo delicado y lírico.",
       "La flauta en grave (<C5) se pierde: sube la octava o reduce las cuerdas.",
       ["melody"], ["mid", "high"], ["soft"], 1)
_combo("solo_madera_pad", "Solo de madera sobre colchón de cuerdas", ["oboe", "clarinete", "corno_ingles", "cuerdas"], "capas",
       "Solista con identidad propia sobre una armonía suave.",
       "Cualquier melodía íntima o lírica.",
       "Evita que el pad y el solista estén en el mismo registro (se enmascaran).",
       ["melody"], ["mid"], ["soft", "mid"], 1)
_combo("flautin_violines_climax", "Flautín + violines al clímax", ["flautin", "violin"], "octava",
       "Brillo extremo en un clímax agudo.",
       "Culminaciones líricas muy agudas.",
       "Solo en f-ff; en pp el flautín no tiene suficiente cuerpo.",
       ["melody"], ["high"], ["loud"], 2)
_combo("tutti_octavas", "Tutti en octavas", ["cuerdas", "trompa", "trompeta", "flauta", "oboe", "trombon"], "octavas",
       "Gran masa y brillo de clímax.",
       "Momentos de máxima intensidad: la melodía en varias octavas.",
       "Si todo toca lo mismo, se pierde la claridad: deja una voz de bajo independiente.",
       ["melody"], ["mid", "high"], ["loud"], 2)
_combo("cuerdas_melodia_octavas", "Melodía en cuerdas a dos octavas", ["violin", "viola", "violonchelo"], "octavas",
       "La melodía lírica más llena y homogénea de la orquesta.",
       "Temas amplios de cuerdas; el estilo romántico.",
       "Cuidado si violonchelos bajan de C3: se enturbia.",
       ["melody"], ["mid", "high"], ["mid", "loud"], 3)
_combo("clarinete_chelo", "Clarinete + violonchelo", ["clarinete", "violonchelo"], "unísono",
       "Contramelodía oscura y cantábile.",
       "Voz interior sombría con carácter vocal.",
       "El clarinete en garganta se pierde; escribe en chalumeau o en clarín.",
       ["countermelody"], ["mid", "low"], ["soft", "mid"], 3)
_combo("laminas_arpa", "Láminas + arpa", ["laminas", "arpa"], "unísono",
       "Percusión melódica nítida y transparente.",
       "Ostinatos y figuras de color.",
       "No sostiene notas largas: usa figuraciones cortas.",
       ["texture", "rhythm"], ["mid", "high"], ["soft", "mid"], 3)


# ══════════════════════════════════════════════════════════════════════════════
#  REGLAS DE EQUILIBRIO Y ESCRITURA
# ══════════════════════════════════════════════════════════════════════════════

RULES: List[dict] = []


def _rule(rid, rule, why, tags):
    RULES.append(dict(id=rid, rule=rule, why=why, tags=tags))


_rule("melody_register",
      "La melodía debe estar en el registro más agudo de la textura o tener un timbre distinto del acompañamiento.",
      "El oído sigue la línea más aguda y la de timbre contrastante; si ninguna de las dos condiciones se cumple, la melodía se pierde.",
      ["balance", "melodia"])
_rule("masking_same_register",
      "Dos líneas del mismo timbre y de registro cercano (menos de 5 semitonos) se enmascaran entre sí.",
      "Los armónicos se solapan; el oído no separa las voces y la melodía parece una masa.",
      ["balance", "enmascaramiento"])
_rule("brass_vs_strings",
      "A igual dinámica, los metales cubren a las cuerdas y a las maderas: baja los metales uno o dos niveles o refuerza la cuerda.",
      "Los metales tienen más energía armónica y proyección que las cuerdas.",
      ["balance", "metales"])
_rule("winds_low",
      "Las maderas graves (flauta por debajo de C5, clarinete en garganta) son débiles: no las dejes de solistas bajo cuerdas f.",
      "Su proyección es mucho menor que la de la cuerda, y su color se pierde.",
      ["balance", "maderas", "registro"])
_rule("bass_clarity",
      "Un bajo claro es un instrumento (o una octava) por línea: evita doblar el bajo en terceras graves.",
      "Los intervalos de tercera en el grave producen batidos y enturbian.",
      ["bajo", "armonia"])
_rule("low_chords",
      "Los acordes cerrados por debajo de aproximadamente C3 (MIDI 48) suenan turbios: abre el espaciado (quinta y octava abajo, terceras arriba).",
      "Los armónicos graves se solapan y producen batidos.",
      ["armonia", "espaciado", "registro"])
_rule("spacing",
      "Entre las voces adyacentes superiores no dejes más de una octava, y reserva los huecos grandes para el grave.",
      "La serie armónica distribuye los intervalos de forma natural: octava, quinta, cuarta, tercera.",
      ["armonia", "espaciado"])
_rule("octave_doubling",
      "Duplicar a la octava refuerza sin turbiar; duplicar al unísono funde los timbres en uno nuevo.",
      "En la octava los armónicos coinciden; en el unísono los timbres se mezclan.",
      ["doblaje"])
_rule("natural_dynamics",
      "Respeta la dinámica natural de cada registro: flauta grave nunca ff, trompeta aguda difícilmente pp.",
      "Cada instrumento tiene zonas de potencia distintas; fuera de ellas el timbre se deforma.",
      ["dinamica", "registro"])
_rule("breathing",
      "Los vientos necesitan respirar: frases de más de 15-20 segundos exigen alternar instrumentos.",
      "El mockup puede reproducir notas infinitas, pero un oyente lo percibe como irreal.",
      ["vientos", "realismo"])
_rule("density",
      "No suenen todos a la vez: densidad creciente y capas que entran y salen dan claridad.",
      "Menos instrumentos con roles claros suenan más grandes que todos a la vez.",
      ["densidad", "forma"])
_rule("clear_roles",
      "Cada capa debe tener un rol claro (melodía, contramelodía, armonía, bajo, ritmo, textura); si dos hacen lo mismo, elimina una.",
      "Roles duplicados restan claridad y energía.",
      ["roles"])
_rule("section_contrast",
      "Contrasta secciones cambiando de familia (cuerdas → maderas) o de registro.",
      "El timbre es la forma más directa de marcar un cambio de sección.",
      ["forma", "timbre"])
_rule("sample_libs",
      "En librerías de samples calibra los niveles por oído: la dinámica real (CC1/CC11) no sustituye a la escritura equilibrada.",
      "Cada librería tiene niveles distintos por instrumento y articulación.",
      ["librerias", "mezcla"])
_rule("inner_voices",
      "En una sección de cuerdas cada voz debe moverse por el camino más corto; evita saltos grandes en las voces interiores.",
      "Los saltos grandes en voces interiores llaman la atención sobre un papel de relleno.",
      ["armonia", "conduccion"])


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def parse_pitch(s) -> int:
    """'C4', 'F#3', 'Bb2' o '60' → pitch MIDI (C4 = 60)."""
    if isinstance(s, int):
        return s
    t = str(s).strip()
    if re.fullmatch(r"-?\d+", t):
        return int(t)
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", t)
    if not m:
        raise ValueError(f"nota no válida: {s!r} (usa C4, F#3, Bb2 o un número MIDI)")
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[m.group(1).upper()]
    acc = {"#": 1, "b": -1, "": 0}[m.group(2)]
    return (int(m.group(3)) + 1) * 12 + base + acc


def register_of(pitch: int) -> str:
    """Registro general: low (<C3), mid (C3-B4), high (≥C5)."""
    return "low" if pitch < 48 else ("mid" if pitch < 72 else "high")


def _fold(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


_ALIASES = {"corno ingles": "corno_ingles", "corno inglés": "corno_ingles",
            "violonchelo": "violonchelo", "chelo": "violonchelo", "cello": "violonchelo",
            "trombon": "trombon", "trombón": "trombon", "metales": "viento_metal_gen",
            "brass": "viento_metal_gen", "strings": "cuerdas", "cuerda": "cuerdas",
            "flautin": "flautin", "flautín": "flautin", "timpani": "timbal",
            "timbales": "timbal", "harp": "arpa", "coro": "voz", "choir": "voz",
            "laminas": "laminas", "láminas": "laminas"}


def instrument_key(text: str) -> Optional[str]:
    """Clave del atlas para un nombre/clave/nombre de pista; None si no se reconoce."""
    if not text:
        return None
    t = str(text).strip()
    low = _fold(t)
    for k in ATLAS:
        if low == _fold(k):
            return k
    if low in _ALIASES or t.lower() in _ALIASES:
        return _ALIASES.get(t.lower(), _ALIASES.get(low))
    for k, v in ATLAS.items():
        if low == _fold(v["es"]):
            return k
    k = name_to_profile(t)
    if k and k in ATLAS:
        return k
    if k:                                     # perfil válido pero sin ficha propia
        return k if k in INSTRUMENT_DB else None
    return None


def zone_of(key: str, pitch: int) -> dict:
    """Zona de un instrumento para una nota: {id,label,character,power,lo,hi,in_range}.
    Fuera del rango físico devuelve la zona más cercana con in_range=False."""
    inst = ATLAS.get(key)
    if inst:
        zs = inst["zones"]
        for z in zs:
            if z["lo"] <= pitch <= z["hi"]:
                return dict(z, in_range=True)
        z = zs[0] if pitch < zs[0]["lo"] else zs[-1]
        return dict(z, in_range=False, label=z["label"] + " (fuera de rango)", power=1)
    prof = INSTRUMENT_DB.get(key)
    if prof:
        lo, hi = prof["range"]
        slo, shi = prof["sweet"]
        if pitch < lo or pitch > hi:
            return dict(lo=lo, hi=hi, id="fuera", label="Fuera de rango",
                        character="nota imposible para el instrumento", power=1,
                        in_range=False)
        if slo <= pitch <= shi:
            return dict(lo=slo, hi=shi, id="comodo", label="Zona cómoda",
                        character="tesitura idiomática", power=2, in_range=True)
        return dict(lo=lo, hi=hi, id="extremo", label="Extremo del registro",
                    character="registro tenso o poco característico", power=1,
                    in_range=True)
    return dict(lo=0, hi=127, id="desconocido", label="Instrumento desconocido",
                character="sin datos", power=2, in_range=True)


def projection(key: str, pitch: int, vel: int = 80) -> float:
    """Proyección estimada 0-1 (heurística de equilibrio): base × zona × velocity."""
    inst = ATLAS.get(key)
    base = inst["base"] if inst else 0.6
    z = zone_of(key, pitch)
    v = max(0, min(127, vel)) / 127.0
    return round(min(1.0, base * (z["power"] / 3.0) * (0.4 + 0.6 * v) * 1.35), 3)


def best_for_role(role: str, pitch: Optional[int] = None,
                  register: Optional[str] = None, top: int = 8) -> List[dict]:
    """Instrumentos ordenados por idoneidad para un rol (y altura/registro opcional)."""
    if role not in ROLES:
        raise ValueError(f"rol desconocido: {role} (válidos: {', '.join(ROLES)})")
    out = []
    for key, inst in ATLAS.items():
        r = inst["roles"][role]
        if r["score"] <= 0:
            continue
        score = float(r["score"])
        zone = None
        if pitch is not None:
            z = zone_of(key, pitch)
            if not z["in_range"]:
                continue
            zone = z
            score += z["power"] * 0.5
        elif register:
            centre = {"low": 40, "mid": 60, "high": 84}[register]
            lo = inst["zones"][0]["lo"]
            hi = inst["zones"][-1]["hi"]
            if not (lo <= centre + 8 and hi >= centre - 8):
                continue
            zone = zone_of(key, min(max(centre, lo), hi))
            score += zone["power"] * 0.5
        out.append(dict(key=key, es=inst["es"], score=round(score, 2),
                        role_score=r["score"], note=r["note"],
                        zone=(zone["label"] if zone else None)))
    out.sort(key=lambda d: (-d["score"], d["key"]))
    return out[:top]


def find_combos(role: Optional[str] = None, register: Optional[str] = None,
                with_inst: Optional[str] = None, dynamic: Optional[str] = None) -> List[dict]:
    res = []
    key = instrument_key(with_inst) if with_inst else None
    if with_inst and not key:
        raise ValueError(f"instrumento desconocido: {with_inst}")
    for c in COMBOS:
        if role and role not in c["roles"]:
            continue
        if register and register not in c["register"]:
            continue
        if dynamic and dynamic not in c["dynamic"]:
            continue
        if key and key not in c["insts"]:
            continue
        res.append(c)
    return res


def combos_for(key: str) -> List[dict]:
    return [c for c in COMBOS if key in c["insts"]]


# ══════════════════════════════════════════════════════════════════════════════
#  PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _bar(n, total=3, ch="●", off="○"):
    return ch * int(n) + off * (total - int(n))


def _need_key(text):
    k = instrument_key(text)
    if not k:
        raise ValueError(f"instrumento desconocido: {text!r} "
                         f"(prueba: {', '.join(sorted(ATLAS))})")
    if k not in ATLAS:
        raise ValueError(f"{text!r} se reconoce ({k}) pero no tiene ficha en el atlas")
    return k


def _wrap(text, indent, width=78):
    import textwrap
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def _emit(obj, args, text_fn):
    if args.json:
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        text_fn()
    return 0


def cmd_list(a):
    rows = []
    for k, v in ATLAS.items():
        if a.family and v["family"] != a.family:
            continue
        hay = _fold(k + " " + v["es"] + " " + v["family"])
        if a.grep and _fold(a.grep) not in hay:
            continue
        lo, hi = v["zones"][0]["lo"], v["zones"][-1]["hi"]
        rows.append(dict(key=k, es=v["es"], family=v["family"], range=[lo, hi]))

    def text():
        B, R = _c("B"), _c("R")
        print(f"\n{B}ORCHESTRAL ATLAS v{VERSION} · instrumentos ({len(rows)}){R}\n")
        for r in rows:
            print(f"  {r['key']:<18} {r['es']:<36} {FAMILY_ES.get(r['family'], r['family']):<18} "
                  f"{pitch_name(r['range'][0])}–{pitch_name(r['range'][1])}")
        print()
    return _emit(rows, a, text)


def cmd_show(a):
    k = _need_key(a.instrument)
    v = ATLAS[k]
    out = dict(v, combos=[c["id"] for c in combos_for(k)])

    def text():
        B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
        print(f"\n{'═' * 78}\n  {B}{v['es'].upper()}{R}  ({k}) · {FAMILY_ES.get(v['family'], v['family'])}"
              f" · proyección base {v['base']:.2f}\n{'═' * 78}")
        print(f"\n  {B}Zonas de registro{R}   (potencia 1 débil · 3 penetrante)")
        for z in v["zones"]:
            print(f"    {pitch_name(z['lo']):>4}–{pitch_name(z['hi']):<4} {_bar(z['power'])}  "
                  f"{C}{z['label']}{R}")
            print(_wrap(z["character"], "                    "))
        print(f"\n  {B}Roles{R}")
        for r in ROLES:
            rv = v["roles"][r]
            print(f"    {ROLE_ES[r]:<16} {_bar(rv['score'])}  {G}{rv['note']}{R}")
        print(f"\n  {_c('GRN')}Virtudes{R}")
        for s in v["strengths"]:
            print(_wrap("✓ " + s, "    "))
        print(f"\n  {_c('YEL')}Cautelas{R}")
        for s in v["cautions"]:
            print(_wrap("→ " + s, "    "))
        print(f"\n  Dinámica:      {v['dynamics']}")
        print(f"  Articulación:  {v['articulation']}")
        cs = combos_for(k)
        if cs:
            print(f"\n  {B}Combinaciones en las que participa{R}")
            for c in cs:
                print(f"    · {c['name']} ({c['relation']}) → python orchestral_atlas.py combos --with {k}")
                break
            for c in cs:
                print(f"    · {c['name']}: {c['effect']}")
        print()
    return _emit(out, a, text)


def cmd_zone(a):
    k = instrument_key(a.instrument)
    if not k:
        raise ValueError(f"instrumento desconocido: {a.instrument!r}")
    res = []
    for n in a.notes:
        p = parse_pitch(n)
        z = zone_of(k, p)
        res.append(dict(note=pitch_name(p), pitch=p, zone=z["label"], in_range=z["in_range"],
                        power=z["power"], character=z["character"],
                        projection_mf=projection(k, p, 80)))

    def text():
        B, R, G = _c("B"), _c("R"), _c("G")
        es = ATLAS[k]["es"] if k in ATLAS else INSTRUMENT_DB.get(k, {}).get("es", k)
        print(f"\n{B}{es}{R}")
        for r in res:
            mark = _c("GRN") if r["in_range"] and r["power"] >= 2 else _c("YEL")
            print(f"  {r['note']:<4} {mark}{r['zone']}{R}  {_bar(r['power'])}")
            print(_wrap(r["character"], "       "))
            print(f"       {G}proyección estimada a mf: {r['projection_mf']:.2f}{R}")
        print()
    return _emit(res, a, text)


def cmd_combos(a):
    cs = find_combos(a.role, a.register, a.__dict__.get("with_"), a.dynamic)[:a.top]

    def text():
        B, R, G = _c("B"), _c("R"), _c("G")
        print(f"\n{B}COMBINACIONES ({len(cs)}){R}\n")
        if not cs:
            print("  (ninguna con esos filtros)\n")
        for c in cs:
            ins = " + ".join(ATLAS[i]["es"] if i in ATLAS else i for i in c["insts"])
            print(f"  {B}{c['name']}{R}  [{c['relation']}]  fusión {_bar(c['blend'])}")
            print(f"    {ins}")
            print(_wrap("Efecto: " + c["effect"], "    "))
            print(_wrap("Úsala: " + c["use"], "    "))
            print(_wrap("Ojo: " + c["caution"], "    "))
            print(f"    {G}roles: {', '.join(ROLE_ES[r] for r in c['roles'])} · "
                  f"registro: {', '.join(REGISTER_ES[r] for r in c['register'])} · "
                  f"dinámica: {', '.join(DYNAMIC_ES[d] for d in c['dynamic'])}{R}\n")
    return _emit(cs, a, text)


def cmd_role(a):
    p = parse_pitch(a.pitch) if a.pitch else None
    res = best_for_role(a.role, pitch=p, register=a.register, top=a.top)

    def text():
        B, R, G = _c("B"), _c("R"), _c("G")
        sub = (f" en {pitch_name(p)}" if p is not None
               else (f" en registro {REGISTER_ES[a.register]}" if a.register else ""))
        print(f"\n{B}Mejores instrumentos para {ROLE_ES[a.role]}{sub}{R}\n")
        for r in res:
            z = f" · {r['zone']}" if r["zone"] else ""
            print(f"  {r['es']:<36} {_bar(r['role_score'])}{z}")
            print(f"      {G}{r['note']}{R}")
        print()
    return _emit(res, a, text)


def cmd_compare(a):
    ka, kb = _need_key(a.a), _need_key(a.b)
    p = parse_pitch(a.pitch) if a.pitch else None
    A, Bv = ATLAS[ka], ATLAS[kb]
    out = dict(a=ka, b=kb, pitch=p, roles={}, zones={})
    for r in ROLES:
        out["roles"][r] = [A["roles"][r]["score"], Bv["roles"][r]["score"]]
    if p is not None:
        out["zones"] = {ka: zone_of(ka, p), kb: zone_of(kb, p)}
        out["projection"] = {ka: projection(ka, p), kb: projection(kb, p)}

    def text():
        B, R, G = _c("B"), _c("R"), _c("G")
        print(f"\n{B}{A['es']}  vs  {Bv['es']}{R}\n")
        print(f"  {'':<16}{A['es'][:22]:<24}{Bv['es'][:22]:<24}")
        for r in ROLES:
            print(f"  {ROLE_ES[r]:<16}{_bar(A['roles'][r]['score']):<24}{_bar(Bv['roles'][r]['score']):<24}")
        print(f"  {'proyección base':<16}{A['base']:<24}{Bv['base']:<24}")
        if p is not None:
            za, zb = out["zones"][ka], out["zones"][kb]
            print(f"\n  {B}En {pitch_name(p)}:{R}")
            print(f"    {A['es']}: {za['label']} · {_bar(za['power'])} · proyección {out['projection'][ka]:.2f}")
            print(f"    {Bv['es']}: {zb['label']} · {_bar(zb['power'])} · proyección {out['projection'][kb]:.2f}")
        print()
    return _emit(out, a, text)


def cmd_rules(a):
    rs = [r for r in RULES
          if not a.grep or _fold(a.grep) in _fold(r["rule"] + " " + r["why"] + " " + " ".join(r["tags"]))]

    def text():
        B, R, G = _c("B"), _c("R"), _c("G")
        print(f"\n{B}REGLAS DE EQUILIBRIO Y ESCRITURA ({len(rs)}){R}\n")
        for r in rs:
            print(_wrap(f"[{r['id']}] {r['rule']}", "  "))
            print(_wrap("¿Por qué? " + r["why"], "      "))
            print()
    return _emit(rs, a, text)


def cmd_selftest(a):
    errs = []
    for k, v in ATLAS.items():
        if INSTRUMENT_DB:
            if k not in INSTRUMENT_DB:
                errs.append(f"{k}: no está en INSTRUMENT_DB")
            else:
                lo, hi = INSTRUMENT_DB[k]["range"]
                zs = v["zones"]
                if zs[0]["lo"] != lo or zs[-1]["hi"] != hi:
                    errs.append(f"{k}: zonas {zs[0]['lo']}-{zs[-1]['hi']} no cubren el rango {lo}-{hi}")
        for i, z in enumerate(v["zones"]):
            if z["lo"] > z["hi"]:
                errs.append(f"{k}: zona {z['id']} invertida")
            if i and z["lo"] != v["zones"][i - 1]["hi"] + 1:
                errs.append(f"{k}: hueco o solape entre zonas {i - 1} y {i}")
            if z["power"] not in (1, 2, 3):
                errs.append(f"{k}: potencia fuera de 1-3 en {z['id']}")
        if set(v["roles"]) != set(ROLES):
            errs.append(f"{k}: roles incompletos")
    for c in COMBOS:
        for i in c["insts"]:
            if i not in ATLAS:
                errs.append(f"combo {c['id']}: instrumento {i} sin ficha")
        if not set(c["roles"]) <= set(ROLES):
            errs.append(f"combo {c['id']}: rol desconocido")
        if not set(c["register"]) <= set(REGISTERS) or not set(c["dynamic"]) <= set(DYNAMICS):
            errs.append(f"combo {c['id']}: registro/dinámica desconocidos")
    if len({c["id"] for c in COMBOS}) != len(COMBOS):
        errs.append("ids de combo repetidos")
    if parse_pitch("C4") != 60 or parse_pitch("F#3") != 54 or parse_pitch("Bb2") != 46:
        errs.append("parse_pitch incorrecto")
    if instrument_key("Corno Inglés") != "corno_ingles":
        errs.append("instrument_key no resuelve 'Corno Inglés'")
    if zone_of("flauta", 62)["id"] != "grave" or zone_of("flauta", 90)["id"] != "agudo":
        errs.append("zone_of(flauta) incorrecto")
    if not projection("trompeta", 70, 100) > projection("flauta", 65, 100):
        errs.append("la trompeta debería proyectar más que la flauta grave")
    if not best_for_role("bass", register="low"):
        errs.append("best_for_role(bass, low) vacío")
    if not find_combos(role="melody", register="mid"):
        errs.append("find_combos vacío")
    ok = not errs
    if a.json:
        print(json.dumps(dict(ok=ok, instruments=len(ATLAS), combos=len(COMBOS),
                              rules=len(RULES), errors=errs), ensure_ascii=False, indent=2))
    else:
        for e in errs:
            print(f"  ✗ {e}")
        print(f"  {'OK' if ok else 'FALLO'} · {len(ATLAS)} instrumentos · "
              f"{len(COMBOS)} combinaciones · {len(RULES)} reglas")
    return 0 if ok else 3


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="Salida en JSON")
    common.add_argument("--no-color", action="store_true")

    ap = argparse.ArgumentParser(
        prog="orchestral_atlas.py",
        description="Atlas de orquestación: instrumentos, registros, roles y combinaciones.")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("list", parents=[common], help="Instrumentos del atlas")
    p.add_argument("--family", choices=sorted(FAMILY_ES))
    p.add_argument("--grep")
    p.set_defaults(fn=cmd_list)

    p = sp.add_parser("show", parents=[common], help="Ficha de un instrumento")
    p.add_argument("instrument")
    p.set_defaults(fn=cmd_show)

    p = sp.add_parser("zone", parents=[common], help="Zona de una nota en un instrumento")
    p.add_argument("instrument")
    p.add_argument("notes", nargs="+")
    p.set_defaults(fn=cmd_zone)

    p = sp.add_parser("combos", parents=[common], help="Combinaciones recomendadas")
    p.add_argument("--role", choices=ROLES)
    p.add_argument("--register", choices=REGISTERS)
    p.add_argument("--with", dest="with_")
    p.add_argument("--dynamic", choices=DYNAMICS)
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(fn=cmd_combos)

    p = sp.add_parser("role", parents=[common], help="Instrumentos idóneos para un rol")
    p.add_argument("role", choices=ROLES)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--pitch")
    g.add_argument("--register", choices=REGISTERS)
    p.add_argument("--top", type=int, default=8)
    p.set_defaults(fn=cmd_role)

    p = sp.add_parser("compare", parents=[common], help="Dos instrumentos cara a cara")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--pitch")
    p.set_defaults(fn=cmd_compare)

    p = sp.add_parser("rules", parents=[common], help="Reglas de equilibrio y escritura")
    p.add_argument("--grep")
    p.set_defaults(fn=cmd_rules)

    p = sp.add_parser("selftest", parents=[common], help="Coherencia interna del catálogo")
    p.set_defaults(fn=cmd_selftest)

    a = ap.parse_args()
    if a.no_color:
        _USE_COLOR = False
    try:
        return a.fn(a)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
    except Exception as e:                              # pragma: no cover
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
