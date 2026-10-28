# =============================================================================
# learning.dedup — deduplicación semántica de carpetas a partir de listados
#
# Entrada : una carpeta con ficheros de texto plano; cada fichero = una carpeta
#           del disco, cada línea = ruta completa de un fichero (salida de find).
# Salida  : $learning_dir/dedup/{report.txt,candidates.tsv,remove_script.sh}
#           (+ gray_zone.txt con dedup_debug=true)
#
# Uso     : cd carpeta_con_listados; learning.dedup            # usa $PWD
#           learning.dedup /ruta/listados [/ruta/salida] [--max-folders 200 ...]
#           (los argumentos a partir del 3º se pasan tal cual al Python)
#
# Dependencias: python3, numpy.
#   Modo embeddings (por defecto): sentence-transformers (CPU basta).
#   Opcionales : faiss-cpu (kNN más rápido), rapidfuzz (similitud léxica rápida).
#   Modo léxico (dedup_embed=false): scikit-learn.
# =============================================================================

# ---- Añadir al bloque de configuración del script ---------------------------
dedup_python=""                # vacío: usa el venv de learning.dedup_setup si existe, si no python3
dedup_venv=""                  # vacío: $learning_dir/dedup/venv
dedup_embed=true
dedup_model=paraphrase-multilingual-MiniLM-L12-v2
dedup_item_threshold=0.85      # τ: similitud mínima para considerar equivalentes dos ficheros
dedup_cover_threshold=0.95     # θ: cobertura ponderada mínima para proponer borrar una carpeta
dedup_idf_max_df=0.02          # títulos en más de este % de carpetas = genéricos (peso 0)
dedup_min_items=3              # mínimo de ítems no genéricos para poder proponer una carpeta
dedup_min_tokens=2             # títulos con menos palabras solo casan por igualdad exacta
dedup_knn_k=30                 # vecinos por título (holgado: las hermanas ocupan sitio)
dedup_max_coverers=3           # máximo de carpetas conservadas que pueden cubrir a una candidata
dedup_digit_cap=0.60           # tope de similitud si los números del título difieren
dedup_use_subdirs=false        # true: los subdirectorios forman parte del título
dedup_plain=false              # true: cada línea es un texto libre (p. ej. un tema), no una ruta;
                               #       el nombre del fichero es el nombre de la carpeta
dedup_base_dir=""              # con dedup_plain: directorio donde están las carpetas reales
dedup_dest=""                  # vacío: <padre>/_to_delete junto a cada carpeta
dedup_debug=false
dedup_strip_ext=""             # vacío: lista por defecto del Python
dedup_skip_ext=""              # vacío: lista por defecto del Python
# -----------------------------------------------------------------------------

learning.dedup(){
    local dir="${1:-$PWD}"
    local out="${2:-$learning_dir/dedup}"
    local -a args

    mkdir -p "$out"
    learning.dedup_write_py "$out/dedup.py"

    args=( --dir "$dir" --out "$out"
           --model "${dedup_model:-paraphrase-multilingual-MiniLM-L12-v2}"
           --tau "${dedup_item_threshold:-0.85}"
           --theta "${dedup_cover_threshold:-0.95}"
           --max-df "${dedup_idf_max_df:-0.02}"
           --min-items "${dedup_min_items:-3}"
           --min-tokens "${dedup_min_tokens:-2}"
           --knn-k "${dedup_knn_k:-30}"
           --max-coverers "${dedup_max_coverers:-3}"
           --digit-cap "${dedup_digit_cap:-0.60}" )

    [ -n "$dedup_strip_ext" ]                   && args+=( --strip-ext "$dedup_strip_ext" )
    [ -n "$dedup_skip_ext" ]                    && args+=( --skip-ext "$dedup_skip_ext" )
    [ -n "$dedup_dest" ]                        && args+=( --dest "$dedup_dest" )
    [ "${dedup_embed:-true}" = "false" ]        && args+=( --no-embed )
    [ "${dedup_use_subdirs:-false}" = "true" ]  && args+=( --use-subdirs )
    [ "${dedup_plain:-false}" = "true" ]        && args+=( --plain )
    [ -n "$dedup_base_dir" ]                    && args+=( --base-dir "$dedup_base_dir" )
    [ "${dedup_debug:-false}" = "true" ]        && args+=( --debug )

    "$(learning.dedup_pybin)" "$out/dedup.py" "${args[@]}" "${@:3}"
}

# Intérprete que usará learning.dedup: $dedup_python > venv de dedup_setup > python3
learning.dedup_pybin(){
    local venv="${dedup_venv:-$learning_dir/dedup/venv}"
    if [ -n "$dedup_python" ]; then
        echo "$dedup_python"
    elif [ -x "$venv/bin/python" ]; then
        echo "$venv/bin/python"
    else
        echo python3
    fi
}

# Comprueba qué hay disponible sin instalar nada. Devuelve 0 si al menos el modo léxico funciona.
learning.dedup_check(){
    local py
    py="$(learning.dedup_pybin)"
    echo "intérprete: $py"
    "$py" - "${dedup_model:-paraphrase-multilingual-MiniLM-L12-v2}" << 'PYEOF'
import sys, importlib
print("python %d.%d.%d" % sys.version_info[:3])
ok = {}
for mod, need in (("numpy", "obligatorio"), ("sklearn", "modo léxico"), ("rapidfuzz", "opcional (más rápido)"),
                  ("faiss", "opcional (--faiss)"), ("sentence_transformers", "modo embeddings")):
    try:
        m = importlib.import_module(mod)
        ok[mod] = True
        print("  [ok]  %-22s %s" % (mod, getattr(m, "__version__", "")))
    except Exception:
        ok[mod] = False
        print("  [--]  %-22s falta  (%s)" % (mod, need))
model_ok = False
if ok["sentence_transformers"]:
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(sys.argv[1], device="cpu", local_files_only=True)
        model_ok = True
        print("  [ok]  modelo %s en caché local" % sys.argv[1])
    except Exception:
        print("  [--]  modelo %s no está descargado" % sys.argv[1])
lex = ok["numpy"] and ok["sklearn"]
emb = ok["numpy"] and ok["sentence_transformers"]
print("modo léxico (dedup_embed=false): %s" % ("LISTO" if lex else "NO disponible"))
print("modo embeddings (por defecto):   %s" % ("LISTO" if emb and model_ok else
      "falta el modelo (se descargará en la primera ejecución)" if emb else "NO disponible"))
sys.exit(0 if lex or emb else 1)
PYEOF
}

# Prepara el entorno. Uso: learning.dedup_setup [--lite] [--no-faiss] [--no-venv] [--check]
#   (sin opciones) venv aislado con numpy, scikit-learn, rapidfuzz, torch (CPU),
#                  sentence-transformers, faiss-cpu y descarga del modelo
#   --lite     solo numpy, scikit-learn, rapidfuzz: modo léxico, sin torch ni modelo
#   --no-faiss no instala faiss-cpu
#   --no-venv  instala con "pip --user" en el python3 del sistema en vez de un venv
#   --check    solo informa de lo que hay, no instala nada
learning.dedup_setup(){
    local lite=false faiss=true usevenv=true check=false arg py
    for arg in "$@"
    do
        case "$arg" in
            --lite)     lite=true ;;
            --no-faiss) faiss=false ;;
            --no-venv)  usevenv=false ;;
            --check)    check=true ;;
            *) echo "opción desconocida: $arg"; return 1 ;;
        esac
    done

    [ $check = true ] && { learning.dedup_check; return $?; }

    local venv="${dedup_venv:-$learning_dir/dedup/venv}"
    local sys_py
    sys_py="$(command -v python3)" || { echo "ERROR: python3 no está instalado"; return 1; }
    "$sys_py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' \
        || { echo "ERROR: se necesita Python >= 3.9 ($("$sys_py" --version 2>&1))"; return 1; }

    local -a pipopt
    if [ $usevenv = true ]; then
        # Un venv a medias (p. ej. creado sin python3-venv: existe bin/python pero no pip) se recrea
        if [ -x "$venv/bin/python" ] && [ -f "$venv/pyvenv.cfg" ] \
            && ! "$venv/bin/python" -m pip --version >/dev/null 2>&1
        then
            echo ">> el venv existente está incompleto (sin pip); se recrea"
            rm -rf "$venv"
        fi
        if [ ! -x "$venv/bin/python" ]; then
            echo ">> creando venv en $venv"
            mkdir -p "$(dirname "$venv")"
            if ! "$sys_py" -m venv "$venv" || ! "$venv/bin/python" -m pip --version >/dev/null 2>&1
            then
                [ -f "$venv/pyvenv.cfg" ] && rm -rf "$venv"
                echo "ERROR: no se pudo crear el venv con pip."
                echo "       En Debian/Ubuntu: sudo apt install python3-venv  (y vuelve a ejecutar)"
                echo "       Alternativa sin venv: learning.dedup_setup --lite --no-venv"
                return 1
            fi
        else
            echo ">> reutilizando venv $venv"
        fi
        py="$venv/bin/python"
        pipopt=()
    else
        py="$sys_py"
        pipopt=( --user )
        "$py" -m pip --version >/dev/null 2>&1 || { echo "ERROR: pip no disponible (sudo apt install python3-pip)"; return 1; }
    fi

    "$py" -m pip install --upgrade pip "${pipopt[@]}" >/dev/null 2>&1

    echo ">> instalando numpy, scikit-learn, rapidfuzz"
    "$py" -m pip install "${pipopt[@]}" numpy scikit-learn rapidfuzz \
        || { echo "ERROR: falló la instalación base"; return 1; }

    if [ $lite = false ]; then
        echo ">> instalando torch (CPU) — descarga grande, puede tardar"
        "$py" -m pip install "${pipopt[@]}" torch --index-url https://download.pytorch.org/whl/cpu \
            || "$py" -m pip install "${pipopt[@]}" torch \
            || { echo "ERROR: falló torch. Puedes usar: learning.dedup_setup --lite"; return 1; }
        echo ">> instalando sentence-transformers"
        "$py" -m pip install "${pipopt[@]}" sentence-transformers \
            || { echo "ERROR: falló sentence-transformers"; return 1; }
        if [ $faiss = true ]; then
            echo ">> instalando faiss-cpu (opcional)"
            "$py" -m pip install "${pipopt[@]}" faiss-cpu \
                || echo "AVISO: faiss-cpu no se pudo instalar; se usará el kNN exacto con numpy"
        fi
        echo ">> descargando el modelo ${dedup_model:-paraphrase-multilingual-MiniLM-L12-v2}"
        if ! "$py" - "${dedup_model:-paraphrase-multilingual-MiniLM-L12-v2}" << 'PYEOF'
import sys
from sentence_transformers import SentenceTransformer
m = SentenceTransformer(sys.argv[1], device="cpu")
v = m.encode(["prueba de instalación"], normalize_embeddings=True)
print("modelo cargado, dimensión", v.shape[1])
PYEOF
        then
            echo "AVISO: no se pudo descargar el modelo ahora (¿sin red?); se intentará en la primera ejecución"
        fi
    fi

    mkdir -p "$learning_dir/dedup"
    echo
    echo ">> verificación"
    learning.dedup_check
}

learning.dedup_write_py(){
    cat > "$1" << 'PYEOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learning.dedup — propone carpetas a eliminar por contención semántica.

Idea: cada carpeta es un conjunto de títulos normalizados. Una carpeta A es
eliminable si (casi) todos sus títulos con información (peso IDF) tienen un
equivalente en carpetas MAYORES que se conservan. Se procesa de mayor a menor,
así el conjunto de carpetas conservadas es definitivo cuando se evalúa A y no
hay cascadas (A cubierta por B, B borrada después por C).
"""
import argparse, glob, math, os, random, re, shlex, sys, time, unicodedata, zlib
from collections import defaultdict
import numpy as np

DEFAULT_STRIP = ("mp4 mkv avi mov wmv flv webm m4v mpg mpeg ts mp3 flac wav ogg m4a aac wma "
                 "pdf epub mobi azw3 djvu doc docx ppt pptx txt srt vtt sub idx ass "
                 "zip rar 7z converted part")
DEFAULT_SKIP = "jpg jpeg png gif bmp nfo url db ini ds_store sfv md5 sha1 lnk thumbs"

EXT_RE = re.compile(r'\.([A-Za-z0-9]{1,10})$')
NUM_PUNCT = re.compile(r'^\s*(\d{1,4})\s*[.\-_)\]:]+\s*')
NUM_SPACE = re.compile(r'^\s*(\d{1,3})\s+')


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', required=True, help='carpeta con los ficheros de listado')
    p.add_argument('--out', required=True, help='carpeta de salida (distinta de --dir)')
    p.add_argument('--model', default='paraphrase-multilingual-MiniLM-L12-v2')
    p.add_argument('--tau', type=float, default=0.85, help='similitud mínima entre ficheros equivalentes')
    p.add_argument('--theta', type=float, default=0.95, help='cobertura ponderada mínima')
    p.add_argument('--max-df', type=float, default=0.02, help='fracción de carpetas a partir de la cual un título es genérico')
    p.add_argument('--min-items', type=int, default=3)
    p.add_argument('--min-tokens', type=int, default=2)
    p.add_argument('--knn-k', type=int, default=30)
    p.add_argument('--cand-floor', type=float, default=0.60, help='similitud mínima de los candidatos del kNN')
    p.add_argument('--max-coverers', type=int, default=3)
    p.add_argument('--digit-cap', type=float, default=0.60)
    p.add_argument('--strip-ext', default=DEFAULT_STRIP)
    p.add_argument('--skip-ext', default=DEFAULT_SKIP)
    p.add_argument('--use-subdirs', action='store_true')
    p.add_argument('--plain', action='store_true',
                   help='cada fichero = una carpeta, cada línea = un ítem de texto libre (p. ej. temas); '
                        'el nombre del fichero es el nombre de la carpeta')
    p.add_argument('--base-dir', default='',
                   help='con --plain: directorio donde están las carpetas reales (se antepone en remove_script.sh)')
    p.add_argument('--no-embed', action='store_true', help='solo n-gramas de caracteres (sklearn), sin modelo')
    p.add_argument('--no-lex', action='store_true', help='no rescatar pares por similitud léxica')
    p.add_argument('--faiss', action='store_true', help='kNN con faiss HNSW (aproximado) en vez de numpy exacto')
    p.add_argument('--max-folders', type=int, default=0, help='procesar solo los N primeros listados (orden alfabético)')
    p.add_argument('--batch', type=int, default=128, help='batch de embeddings')
    p.add_argument('--dest', default='', help='destino fijo para mvd (vacío: <padre>/_to_delete)')
    p.add_argument('--debug', action='store_true')
    return p.parse_args()


# ----------------------------------------------------------------------------
# Normalización
# ----------------------------------------------------------------------------
def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def simple_norm(s):
    return re.sub(r'[\W_]+', ' ', strip_accents(s).lower()).strip()


def last_ext(name):
    m = EXT_RE.search(name)
    return m.group(1).lower() if m else ''


def normalize_name(name, strip_ext):
    """Devuelve (clave_normalizada, número_inicial_o_None)."""
    s = name
    while True:
        m = EXT_RE.search(s)
        if m and m.group(1).lower() in strip_ext:
            s = s[:m.start()]
        else:
            break
    num = None
    for rx in (NUM_PUNCT, NUM_SPACE):
        m = rx.match(s)
        if m and s[m.end():].strip():
            num = int(m.group(1))
            s = s[m.end():]
            break
    s = re.sub(r'\s*\(\d{1,2}\)\s*$', '', s)
    s = re.sub(r'[\s_-]+(copy|copia)(\s*\d*)?\s*$', '', s, flags=re.I)
    return simple_norm(s), num


# ----------------------------------------------------------------------------
# Lectura de listados
# ----------------------------------------------------------------------------
BULLET_RE = re.compile(r'^\s*(?:[-*•–]+|\d{1,3}[.)])\s+')


def parse_plain(path):
    """Modo --plain: nombre de fichero = nombre de carpeta; cada línea = un ítem de texto libre."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        raw = [l.rstrip('\r\n') for l in f]
    seen, entries = set(), []
    for l in raw:
        txt = BULLET_RE.sub('', l).strip()
        key = simple_norm(txt)
        if not key or key in seen:
            continue
        seen.add(key)
        entries.append((key, txt, None))
    if not entries:
        return None
    nm = os.path.basename(path)
    return dict(root=nm, name=nm, listing=nm, entries=entries)


def parse_listing(path, args, strip_ext, skip_ext):
    """Devuelve una LISTA de carpetas (normalmente una).
    Si el listado mezcla varias carpetas hermanas (p. ej. generado con un glob por prefijo),
    se separa una carpeta por cada subcarpeta de primer nivel bajo el padre común."""
    if args.plain:
        L = parse_plain(path)
        return [L] if L else []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [l.rstrip('\r\n').rstrip('/') for l in f]
    lines = [l for l in lines if l.strip()]
    if not lines:
        return []
    parents = set()
    for l in lines:
        p = os.path.dirname(l)
        while p and p != '/' and p not in parents:
            parents.add(p)
            p = os.path.dirname(p)
    files = [l for l in lines if l not in parents]
    if not files:
        return []
    groups = {}
    if lines[0] in parents and all(f.startswith(lines[0] + '/') for f in files):
        groups[lines[0]] = files
    else:
        try:
            P = os.path.commonpath([os.path.dirname(f) for f in files])
        except ValueError:
            P = os.path.dirname(files[0])
        pre = P.rstrip('/') + '/'
        loose = 0
        for f in files:
            rest = f[len(pre):] if f.startswith(pre) else f
            if '/' not in rest:
                loose += 1
                continue
            groups.setdefault(pre + rest.split('/', 1)[0], []).append(f)
        if not groups:
            groups[P] = files
        elif loose:
            log('AVISO: %s: %d ficheros sueltos fuera de subcarpetas, ignorados' % (os.path.basename(path), loose))
        if len(groups) > 1:
            log('AVISO: %s contiene %d carpetas distintas: se separan' % (os.path.basename(path), len(groups)))
    out = []
    for root, fl in groups.items():
        L = build_folder(root, fl, os.path.basename(path), args, strip_ext, skip_ext)
        if L:
            out.append(L)
    return out


def build_folder(root, files, listing, args, strip_ext, skip_ext):
    fname_norm = simple_norm(os.path.basename(root))
    entries = []
    for f in files:
        rel = f[len(root) + 1:] if f.startswith(root + '/') else f
        parts = rel.split('/')
        name = parts[-1]
        if last_ext(name) in skip_ext:
            continue
        key, num = normalize_name(name, strip_ext)
        if not key:
            key = simple_norm(name)
        if not key:
            continue
        if args.use_subdirs and len(parts) > 1:
            subs = [simple_norm(x) for x in parts[:-1]]
            subs = [x for x in subs if x and x != fname_norm]
            if subs:
                key = ' / '.join(subs + [key])
        entries.append((key, name, num))
    if not entries:
        return None
    return dict(root=root, name=os.path.basename(root) or root,
                listing=listing, entries=entries)


# ----------------------------------------------------------------------------
# Embeddings con caché por partes
# ----------------------------------------------------------------------------
def embed_keys(keys_sub, args):
    slug = re.sub(r'[^A-Za-z0-9_.-]', '_', args.model)
    cdir = os.path.join(args.out, 'emb_cache', slug)
    os.makedirs(cdir, exist_ok=True)
    parts = sorted(glob.glob(os.path.join(cdir, 'part_*.npz')))
    row = {}
    mats = []
    off = 0
    for pth in parts:
        z = np.load(pth, allow_pickle=False)
        ks = z['keys'].tolist()
        for i, k in enumerate(ks):
            row[k] = off + i
        mats.append(z['emb'])
        off += len(ks)
    missing = [k for k in keys_sub if k not in row]
    log('embeddings: %d en caché, %d por calcular' % (len(keys_sub) - len(missing), len(missing)))
    if missing:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model, device='cpu')
        CH = 20000
        for s in range(0, len(missing), CH):
            chunk = missing[s:s + CH]
            emb = model.encode(chunk, batch_size=args.batch, normalize_embeddings=True,
                               show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
            idx = len(glob.glob(os.path.join(cdir, 'part_*.npz')))
            np.savez(os.path.join(cdir, 'part_%06d.npz' % idx), keys=np.array(chunk), emb=emb)
            for i, k in enumerate(chunk):
                row[k] = off + i
            mats.append(emb)
            off += len(chunk)
            log('  checkpoint: %d/%d' % (min(s + CH, len(missing)), len(missing)))
    allm = np.vstack(mats) if len(mats) > 1 else mats[0]
    return np.ascontiguousarray(allm[[row[k] for k in keys_sub]], dtype=np.float32)


# ----------------------------------------------------------------------------
# kNN
# ----------------------------------------------------------------------------
def topk_block(S, start, kk, floor, oi, oj, os_):
    b, n = S.shape
    if kk < n:
        idx = np.argpartition(S, n - kk, axis=1)[:, n - kk:]
    else:
        idx = np.broadcast_to(np.arange(n), (b, n))
    vals = np.take_along_axis(S, idx, axis=1)
    rows = np.arange(start, start + b)[:, None]
    m = (vals >= floor) & (idx != rows)
    r, c = np.nonzero(m)
    oi.append((r + start).astype(np.int64))
    oj.append(np.asarray(idx[r, c]).astype(np.int64))
    os_.append(np.asarray(vals[r, c]).astype(np.float32))


def knn_dense(E, k, floor, use_faiss):
    n = E.shape[0]
    kk = min(k + 1, n)
    oi, oj, os_ = [], [], []
    if use_faiss:
        import faiss
        index = faiss.IndexHNSWFlat(E.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.add(E)
        index.hnsw.efSearch = max(64, kk * 2)
        B = 20000
        for s in range(0, n, B):
            D, I = index.search(E[s:s + B], kk)
            rows = np.arange(s, s + D.shape[0])[:, None]
            m = (I >= 0) & (D >= floor) & (I != rows)
            r, c = np.nonzero(m)
            oi.append((r + s).astype(np.int64)); oj.append(I[r, c].astype(np.int64)); os_.append(D[r, c].astype(np.float32))
            log('  kNN faiss %d/%d' % (min(s + B, n), n))
    else:
        B = int(max(16, min(2048, 6e7 // max(n, 1))))
        ET = np.ascontiguousarray(E.T)
        for s in range(0, n, B):
            S = E[s:s + B] @ ET
            topk_block(S, s, kk, floor, oi, oj, os_)
            if (s // B) % 20 == 0:
                log('  kNN %d/%d' % (min(s + B, n), n))
    if not oi:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(oi), np.concatenate(oj), np.concatenate(os_)


def knn_lexical(keys_sub, k, floor):
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), sublinear_tf=True,
                        dtype=np.float32).fit_transform(keys_sub).tocsr()
    XT = X.T.tocsr()
    n = X.shape[0]
    kk = min(k + 1, n)
    oi, oj, os_ = [], [], []
    B = int(max(16, min(512, 3e7 // max(n, 1))))
    for s in range(0, n, B):
        S = (X[s:s + B] @ XT).toarray()
        topk_block(S, s, kk, floor, oi, oj, os_)
        if (s // B) % 20 == 0:
            log('  kNN léxico %d/%d' % (min(s + B, n), n))
    if not oi:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(oi), np.concatenate(oj), np.concatenate(os_)


def canon_pairs(i, j, s, n):
    if len(i) == 0:
        return i, j, s
    a = np.minimum(i, j)
    b = np.maximum(i, j)
    key = a * n + b
    order = np.lexsort((-s, key))
    ks = key[order]
    first = np.ones(len(ks), dtype=bool)
    first[1:] = ks[1:] != ks[:-1]
    sel = order[first]
    return a[sel], b[sel], s[sel]


def _tri(s):
    s = ' ' + s + ' '
    return {s[i:i + 3] for i in range(len(s) - 2)}


def get_lexf():
    try:
        from rapidfuzz import fuzz
        return lambda x, y: fuzz.ratio(x, y) / 100.0
    except Exception:
        def jacc(x, y):
            a, b = _tri(x), _tri(y)
            return len(a & b) / float(len(a | b))
        return jacc


def digit_sig(k):
    return zlib.crc32(' '.join(sorted(re.findall(r'\d+', k))).encode('utf-8'))


def score_pairs(a, b, cos, el_keys, args):
    sim = cos.astype(np.float32).copy()
    if not args.no_lex and len(a):
        lexf = get_lexf()
        for t in np.nonzero(sim < args.tau)[0]:
            l = lexf(el_keys[a[t]], el_keys[b[t]])
            if l > sim[t]:
                sim[t] = l
    if len(a):
        dsig = np.array([digit_sig(k) for k in el_keys], dtype=np.int64)
        mism = dsig[a] != dsig[b]
        sim[mism] = np.minimum(sim[mism], args.digit_cap)
    return sim


# ----------------------------------------------------------------------------
# Cobertura
# ----------------------------------------------------------------------------
def one_to_one(plist, covered, w):
    used = set()
    match = {}
    gain = 0.0
    for s, x, y in plist:
        if x in covered or x in match or y in used:
            continue
        match[x] = (y, s)
        used.add(y)
        gain += w[x]
    return gain, match


def evaluate(A, kept, folders, folders_of, nbr, w, args):
    fa = folders[A]
    items = [x for x in fa['items'] if w[x] > 0]
    if len(items) < args.min_items:
        return None
    total = float(sum(w[x] for x in items))
    pairs = defaultdict(list)
    for x in items:
        eq = [(x, 1.0)]
        eq.extend(nbr.get(x, ()))
        for y, s in eq:
            for B in folders_of[y]:
                if B != A and kept[B]:
                    pairs[B].append((s, x, y))
    if not pairs:
        return dict(cov=0.0, chosen=[], covered={}, items=items, total=total, residual=list(items))
    pot = {B: sum(w[x] for x in set(p[1] for p in lst)) for B, lst in pairs.items()}
    cand = sorted(pot, key=lambda B: -pot[B])[:60]
    for B in cand:
        pairs[B].sort(key=lambda t: -t[0])
    covered, chosen, cov_w = {}, [], 0.0
    for _ in range(args.max_coverers):
        best = None
        for B in cand:
            if B in chosen:
                continue
            gain, match = one_to_one(pairs[B], covered, w)
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, B, match)
        if best is None:
            break
        gain, B, match = best
        for x, (y, s) in match.items():
            covered[x] = (B, y, s)
        chosen.append(B)
        cov_w += gain
        if cov_w / total >= args.theta:
            break
    residual = [x for x in items if x not in covered]
    return dict(cov=cov_w / total, chosen=chosen, covered=covered, items=items,
                total=total, residual=residual)


def tier_of(res):
    if res['cov'] >= 0.99 and len(res['chosen']) == 1:
        return 1
    if len(res['chosen']) <= 2:
        return 2
    return 3


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    if os.path.abspath(args.out) == os.path.abspath(args.dir):
        sys.exit('--out no puede ser la propia carpeta de listados')
    os.makedirs(args.out, exist_ok=True)
    import importlib.util
    if not args.no_embed and importlib.util.find_spec('sentence_transformers') is None:
        sys.exit('ERROR: falta sentence-transformers (modo embeddings).\n'
                 '  - instálalo con:  learning.dedup_setup\n'
                 '  - o usa el modo léxico:  dedup_embed=false; learning.dedup ...')
    strip_ext = set(args.strip_ext.split())
    skip_ext = set(args.skip_ext.split())

    # 1. Parseo ---------------------------------------------------------------
    names = sorted(f for f in os.listdir(args.dir)
                   if not f.startswith('.') and os.path.isfile(os.path.join(args.dir, f)))
    if args.max_folders:
        names = names[:args.max_folders]
    log('listados a leer: %d' % len(names))

    key2id, keys = {}, []
    folders = []
    parsed, by_root_p, n_merged = [], {}, 0
    for nm in names:
        try:
            Ls = parse_listing(os.path.join(args.dir, nm), args, strip_ext, skip_ext)
        except Exception as e:
            log('AVISO: no se pudo leer %s: %s' % (nm, e))
            continue
        for L in Ls:
            prev = by_root_p.get(L['root'])
            if prev is None:
                by_root_p[L['root']] = L
                parsed.append(L)
            else:   # la misma carpeta de disco aparece en varios listados: se usa el más completo
                n_merged += 1
                if len(L['entries']) > len(prev['entries']):
                    prev['entries'] = L['entries']
                if L['listing'] not in prev['listing'].split(' + '):
                    prev['listing'] += ' + ' + L['listing']
    if n_merged:
        log('AVISO: %d carpetas aparecen en más de un listado (misma ruta): se usa la versión más completa' % n_merged)
    for L in parsed:
        ids, orig, nums = set(), {}, []
        for key, name, num in L['entries']:
            kid = key2id.get(key)
            if kid is None:
                kid = len(keys)
                key2id[key] = kid
                keys.append(key)
            ids.add(kid)
            orig.setdefault(kid, name)
            if num is not None:
                nums.append(num)
        folders.append(dict(id=len(folders), name=L['name'], root=L['root'], listing=L['listing'],
                            n_files=len(L['entries']), items=sorted(ids), orig=orig,
                            max_num=max(nums) if nums else 0, n_nums=len(set(nums))))
    N = len(folders)
    if N == 0:
        sys.exit('no hay carpetas útiles')
    total_files = sum(f['n_files'] for f in folders)
    log('carpetas: %d  ficheros: %d  títulos únicos: %d' % (N, total_files, len(keys)))

    # 2. Índice invertido y pesos --------------------------------------------
    folders_of = [[] for _ in keys]
    for f in folders:
        for kid in f['items']:
            folders_of[kid].append(f['id'])
    df = np.array([len(x) for x in folders_of], dtype=np.float64)
    gen_thr = max(int(math.ceil(args.max_df * N)), 8)
    generic = df > gen_thr
    w = np.log(1.0 + N / df)
    w[generic] = 0.0
    w = w.tolist()
    log('títulos genéricos (df > %d): %d' % (gen_thr, int(generic.sum())))

    # 3. Equivalencias difusas entre títulos ---------------------------------
    nbr = defaultdict(list)
    el = [i for i in range(len(keys)) if w[i] > 0 and len(keys[i].split()) >= args.min_tokens]
    log('títulos elegibles para búsqueda difusa: %d' % len(el))
    n_edges = 0
    if len(el) >= 2:
        el_keys = [keys[i] for i in el]
        if args.no_embed:
            log('kNN léxico (n-gramas de caracteres)')
            i, j, s = knn_lexical(el_keys, args.knn_k, args.cand_floor)
        else:
            E = embed_keys(el_keys, args)
            log('kNN denso sobre %d vectores de %d dim' % E.shape)
            i, j, s = knn_dense(E, args.knn_k, args.cand_floor, args.faiss)
            del E
        a, b, c = canon_pairs(i, j, s, len(el))
        log('pares candidatos: %d' % len(a))
        sim = score_pairs(a, b, c, el_keys, args)
        if args.debug:
            hist, edges = np.histogram(sim, bins=np.arange(args.cand_floor, 1.0001, 0.05))
            log('histograma de similitud (candidatos tras reglas):')
            for h, e0, e1 in zip(hist, edges[:-1], edges[1:]):
                log('   %.2f-%.2f  %d' % (e0, e1, h))
            zone = np.nonzero((sim >= args.tau - 0.08) & (sim < args.tau + 0.04))[0]
            random.seed(0)
            pick = random.sample(list(zone), min(80, len(zone)))
            with open(os.path.join(args.out, 'gray_zone.txt'), 'w', encoding='utf-8') as g:
                for t in sorted(pick, key=lambda t: sim[t]):
                    g.write('%.3f  %s  |  %s\n' % (sim[t], el_keys[a[t]], el_keys[b[t]]))
            log('muestra de zona gris -> gray_zone.txt (%d pares en la zona)' % len(zone))
        keep = np.nonzero(sim >= args.tau)[0]
        for t in keep:
            ga, gb = el[a[t]], el[b[t]]
            nbr[ga].append((gb, float(sim[t])))
            nbr[gb].append((ga, float(sim[t])))
        n_edges = len(keep)
    log('aristas de equivalencia (sim >= %.2f): %d' % (args.tau, n_edges))

    # 4. Carpetas anidadas (no tocar si una contiene a otra) ------------------
    by_root = defaultdict(list)
    for f in folders:
        by_root[f['root']].append(f['id'])
    nested = [set() for _ in range(N)]
    for r, ids in ([] if args.plain else by_root.items()):
        if len(ids) > 1:
            for i in ids:
                nested[i] |= set(ids) - {i}
        p = os.path.dirname(r)
        while p and p != '/':
            if p in by_root:
                for x in ids:
                    for y in by_root[p]:
                        nested[x].add(y)
                        nested[y].add(x)
            p = os.path.dirname(p)
    n_nested = sum(1 for s in nested if s)
    if n_nested:
        log('AVISO: %d carpetas anidadas o con raíz repetida: no se proponen' % n_nested)

    # 5. Selección voraz de mayor a menor ------------------------------------
    order = sorted(range(N), key=lambda i: (-folders[i]['n_files'], -len(folders[i]['items']),
                                            folders[i]['name'].lower(), i))
    kept = [False] * N
    removed = {}
    for pos, A in enumerate(order):
        res = None if nested[A] else evaluate(A, kept, folders, folders_of, nbr, w, args)
        if res is not None and res['chosen'] and res['cov'] >= args.theta:
            removed[A] = res
        else:
            kept[A] = True
        if (pos + 1) % 1000 == 0:
            log('evaluadas %d/%d  candidatas: %d' % (pos + 1, N, len(removed)))
    n_rm_files = sum(folders[A]['n_files'] for A in removed)
    log('candidatas a eliminar: %d de %d carpetas (%d de %d ficheros, %.1f%%)' %
        (len(removed), N, n_rm_files, total_files, 100.0 * n_rm_files / max(total_files, 1)))

    # 6. Salidas --------------------------------------------------------------
    cands = sorted(removed.items(),
                   key=lambda kv: (tier_of(kv[1]), -kv[1]['cov'], -folders[kv[0]]['n_files']))
    rep = open(os.path.join(args.out, 'report.txt'), 'w', encoding='utf-8')
    tsv = open(os.path.join(args.out, 'candidates.tsv'), 'w', encoding='utf-8')
    scr = open(os.path.join(args.out, 'remove_script.sh'), 'w', encoding='utf-8')
    tsv.write('tier\tcobertura\tficheros\tnombre\truta\tcubridoras\tno_cubiertos\n')
    scr.write('# Generado por learning.dedup. REVISAR antes de ejecutar (source este_fichero).\n')
    scr.write('# Tier 1 activo; tiers 2 y 3 comentados: descomenta solo lo que revises.\n')
    scr.write('DEST="${DEST:-%s}"\n' % args.dest)
    scr.write('mvd(){\n    local src="$1" dst\n'
              '    [ -d "$src" ] || { echo "no existe: $src"; return 1; }\n'
              '    if [ -n "$DEST" ]; then dst="$DEST"; else dst="$(dirname "$src")/_to_delete"; fi\n'
              '    mkdir -p "$dst" && mv -n -- "$src" "$dst/" && echo "movida: $src"\n}\n')
    rep.write('learning.dedup  tau=%.2f theta=%.2f max_df=%.3f min_items=%d max_coverers=%d\n' %
              (args.tau, args.theta, args.max_df, args.min_items, args.max_coverers))
    rep.write('carpetas=%d  candidatas=%d  ficheros candidatos=%d/%d\n\n' %
              (N, len(removed), n_rm_files, total_files))
    cur = None
    for A, res in cands:
        fa = folders[A]
        t = tier_of(res)
        if t != cur:
            cur = t
            hdr = {1: 'TIER 1: cobertura >= 99% con una sola carpeta cubridora',
                   2: 'TIER 2: hasta 2 carpetas cubridoras',
                   3: 'TIER 3: cubierta por una unión de carpetas (revisar con cuidado)'}[t]
            rep.write('=' * 78 + '\n' + hdr + '\n' + '=' * 78 + '\n')
            scr.write('\n# ' + hdr + '\n')
        cover_names = []
        per = defaultdict(int)
        for x, (B, y, s) in res['covered'].items():
            per[B] += 1
        for B in res['chosen']:
            cover_names.append('%s (%d ficheros): %d ítems' % (folders[B]['name'], folders[B]['n_files'], per[B]))
        resid = sorted(res['residual'], key=lambda x: -w[x])
        compl = ''
        if fa['max_num']:
            compl = '   numeración: %d títulos numerados, máx %d (%.0f%% completa)' % (
                fa['n_nums'], fa['max_num'], 100.0 * fa['n_nums'] / fa['max_num'])
        rep.write('\n[T%d] %s  (%d ficheros)  cobertura %.1f%%  (%d ítems con peso)\n' %
                  (t, fa['name'], fa['n_files'], 100 * res['cov'], len(res['items'])))
        if args.plain:
            rep.write('   carpeta: %s\n' % (os.path.join(args.base_dir, fa['root']) if args.base_dir else fa['root']))
        else:
            rep.write('   ruta   : %s\n   listado: %s\n' % (fa['root'], fa['listing']))
        if compl:
            rep.write(compl + '\n')
        rep.write('   cubierta por:\n')
        for cn in cover_names:
            rep.write('     - %s\n' % cn)
        fuzzy = [(x, v) for x, v in res['covered'].items() if v[2] < 0.999]
        if fuzzy:
            rep.write('   emparejamientos difusos (%d), muestra:\n' % len(fuzzy))
            for x, (B, y, s) in sorted(fuzzy, key=lambda kv: kv[1][2])[:6]:
                rep.write('     %.2f  %s  <->  %s\n' % (s, fa['orig'][x], folders[B]['orig'][y]))
        if resid:
            rep.write('   NO cubiertos (%d), los más informativos:\n' % len(resid))
            for x in resid[:12]:
                rep.write('     - %s\n' % fa['orig'][x])
        tsv.write('%d\t%.4f\t%d\t%s\t%s\t%s\t%d\n' % (
            t, res['cov'], fa['n_files'], fa['name'], fa['root'],
            ' ; '.join(folders[B]['name'] for B in res['chosen']), len(resid)))
        scr.write('# %s | cob %.1f%% | cubierta por: %s | no cubiertos: %d\n' % (
            fa['name'], 100 * res['cov'], ' + '.join(folders[B]['name'] for B in res['chosen']), len(resid)))
        target = os.path.join(args.base_dir, fa['root']) if (args.plain and args.base_dir) else fa['root']
        scr.write('%smvd %s\n' % ('' if t == 1 else '# ', shlex.quote(target)))
    rep.close(); tsv.close(); scr.close()
    log('salida en %s: report.txt, candidates.tsv, remove_script.sh' % args.out)


if __name__ == '__main__':
    main()
PYEOF
}
