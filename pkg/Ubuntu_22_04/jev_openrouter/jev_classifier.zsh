# ---------------------------------------------------------------------------
# jev_classifier: clasificador zero-shot con Jev (TypeSafe) a traves de OpenRouter
#
# Los temas son la ULTIMA PALABRA de cada linea de $learning_dir/topics/topics.
# Como ese fichero es un arbol (de hasta 5 niveles) con ~1500 etiquetas, no se
# pregunta todo de golpe: se desciende por la jerarquia con una pregunta por
# nivel (raiz -> science -> development -> languages -> rust), nunca con mas de
# ~115 opciones en cada una. Un nodo que tambien es etiqueta por si mismo (linea
# "science math") ofrece la opcion "math (general)" para quedarse ahi.
#
# Activacion (cabecera del script, junto a las demas variables):
#
#   cached_classifier_method=jev_classifier   # con cache por md5 del documento
#   # o bien: classifier_method=jev_classifier  (sin cache)
#
# Variables opcionales (todas tienen valor por defecto):
#
#   jev_model=typesafe/jev-1.14      # ~typesafe/jev-latest sigue la ultima version
#   jev_min_confidence=0             # probabilidad minima del camino completo
#   jev_max_bytes=12000              # bytes del documento enviados en cada nivel
#   jev_instructions="..."           # pregunta base que se hace a Jev
#   jev_url=https://openrouter.ai/api/alpha/decisions
#
# Requiere:  export OPENROUTER_API_KEY=...
# ---------------------------------------------------------------------------

# $1 = transform | probs | check, $2 = fichero de texto (no hace falta en check).
# La configuracion viaja en variables de entorno, no interpolada en el codigo Python.
jev_classifier.query(){
    [ "$1" = check ] || [ -f "$2" ] || { echo "jev_classifier: no existe el fichero: $2" >&2; return 1; }

    JEV_TOPICS="$learning_dir/topics/topics" \
    JEV_MODEL="${jev_model:-typesafe/jev-1.14}" \
    JEV_URL="${jev_url:-https://openrouter.ai/api/alpha/decisions}" \
    JEV_MIN_CONF="${jev_min_confidence:-0}" \
    JEV_MAX_BYTES="${jev_max_bytes:-12000}" \
    JEV_INSTRUCTIONS="${jev_instructions}" \
    python - "$1" "$2" <<'PY'
import json, os, sys, time, urllib.request, urllib.error

mode = sys.argv[1]
doc = sys.argv[2]
env = os.environ
MAX_OPTIONS = 255   # limite por pregunta de los clientes oficiales de TypeSafe

def fail(msg):
    sys.stderr.write('jev_classifier: %s\n' % msg)
    sys.exit(1)

# ---- arbol de temas: cada linea de topics/topics es un camino, tema = ultima palabra
class Node:
    def __init__(self, name):
        self.name = name
        self.children = {}
        self.is_tag = False

def load_tree(fname):
    root, n = Node(''), 0
    try:
        with open(fname, encoding='utf-8') as f:
            for line in f:
                w = line.split()
                if not w:
                    continue
                if len(w) > 1:
                    w[0] = w[0].lower()   # tolera "Science" frente a "science" en la raiz
                node = root
                for word in w:
                    if word not in node.children:
                        node.children[word] = Node(word)
                    node = node.children[word]
                node.is_tag = True
                n += 1
    except OSError:
        fail('no se puede leer %s' % fname)
    if not root.children:
        fail('%s esta vacio' % fname)
    return root, n

def options_of(node, is_root):
    return len(node.children) + (1 if node.is_tag and not is_root else 0)

root, nlines = load_tree(env['JEV_TOPICS'])

if mode == 'check':
    worst, wpath, stack = 0, '', [(root, [])]
    while stack:
        node, path = stack.pop()
        k = options_of(node, not path)
        if k > worst:
            worst, wpath = k, ' > '.join(path) or '(raiz)'
        stack.extend((c, path + [name]) for name, c in node.children.items())
    sys.stderr.write('jev_classifier: %d temas, mayor numero de opciones en una pregunta: %d (%s)\n'
                     % (nlines, worst, wpath))
    if worst > MAX_OPTIONS:
        fail('%s tiene mas de %d opciones' % (wpath, MAX_OPTIONS))
    sys.exit(0)

key = env.get('OPENROUTER_API_KEY')
if not key:
    fail('falta la variable OPENROUTER_API_KEY')

try:
    with open(doc, 'rb') as f:
        raw = f.read(int(env['JEV_MAX_BYTES']))
except OSError as e:
    fail('no se puede leer %s: %s' % (doc, e))
text = raw.decode('utf-8', 'replace').strip()
if not text:
    sys.stderr.write('jev_classifier: documento vacio, sin respuesta\n')
    sys.exit(0)

base = env.get('JEV_INSTRUCTIONS') or \
    'Which topic best describes the main subject of this document? Choose the single best fit.'

def describe(node, path):
    d = ' > '.join(path)
    kids = list(node.children)
    if kids:
        d += ' (' + ', '.join(kids[:12]) + (', ...' if len(kids) > 12 else '') + ')'
    return d

def ask(instructions, criteria):
    body = json.dumps({
        'model': env['JEV_MODEL'],
        'state': text,
        'questions': {'topic': {'type': 'choice', 'instructions': instructions,
                                'criteria': criteria}},
    }).encode('utf-8')
    req = urllib.request.Request(env['JEV_URL'], data=body, method='POST', headers={
        'Authorization': 'Bearer ' + key,
        'Content-Type': 'application/json',
        'User-Agent': 'learning-jev/1.0',
    })
    # Reintentos solo para errores transitorios (limite de peticiones, 5xx, red)
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            last = 'HTTP %d: %s' % (e.code, e.read().decode('utf-8', 'replace')[:500])
            if e.code not in (408, 429, 500, 502, 503, 504):
                fail(last)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = str(e)
        except ValueError:
            fail('la respuesta no es JSON valido')
        if attempt < 3:
            time.sleep(2 ** attempt)
    else:
        fail('sin respuesta tras reintentos (%s)' % last)
    try:
        ans = resp['answers']['topic']
        choice = ans['choice']
    except (KeyError, TypeError):
        fail('respuesta inesperada: %s' % json.dumps(resp)[:300])
    if choice not in criteria:
        fail('Jev devolvio una opcion que no estaba en la lista: %r' % choice)
    probs = ans.get('probabilities') or {}
    p = probs.get(choice)
    if p is None:
        p = ans.get('confidence', 1.0)
    runner = sorted(((v, k) for k, v in probs.items() if k != choice), reverse=True)[:1]
    return choice, float(p), ('%s:%.2f' % (runner[0][1], runner[0][0]) if runner else '')

# ---- descenso por el arbol
node, path, total, trace = root, [], 1.0, []
while node.children:
    options = {name: describe(c, path + [name]) for name, c in node.children.items()}
    stop = None
    if path and node.is_tag:
        stop = node.name + ' (general)'
        options[stop] = ('General: %s as a whole, when none of the more specific options fit'
                         % ' > '.join(path))
    if len(options) == 1:           # un unico camino posible: no hace falta preguntar
        name = next(iter(options))
        node, path = node.children[name], path + [name]
        continue
    instr = base
    if path:
        instr += (' The document is already known to belong to: %s. '
                  'Choose the most specific option that fits.' % ' > '.join(path))
    choice, p, runner = ask(instr, options)
    total *= p
    trace.append((p, choice, runner))
    if choice == stop:
        break
    node, path = node.children[choice], path + [choice]

tag = node.name

if mode == 'probs':
    for p, c, r in trace:
        print('%.4f\t%s\t%s' % (p, c, r))
    print('total\t%.4f\t%s' % (total, tag))
elif total >= float(env['JEV_MIN_CONF']):
    print(tag)
else:
    # Sin tema: learning.queue_mv ignora las clasificaciones vacias y el
    # documento se queda en la cola
    sys.stderr.write('jev_classifier: probabilidad %.3f < %s para %s (%s)\n'
                     % (total, env['JEV_MIN_CONF'], doc, tag))
PY
}

jev_classifier.transform(){
    jev_classifier.query transform "$1"
}

# Probabilidad de cada nivel del camino (con la segunda opcion) y el total
jev_classifier.probabilities(){
    jev_classifier.query probs "$1"
}

# Jev no se entrena: fit solo comprueba que topics/topics se puede leer y que
# ninguna pregunta supera el limite de opciones.
jev_classifier.fit(){
    jev_classifier.query check
}
