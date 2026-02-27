"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TENSION DESIGNER  v1.0                                    ║
║         Diseñador visual interactivo de curvas de tensión                    ║
║                                                                              ║
║  Permite dibujar a mano la curva de tensión (y otras curvas emocionales)    ║
║  de una obra completa antes de generar el MIDI. Las curvas diseñadas se     ║
║  convierten en flags --mt-* para midi_dna_unified.py y se pasan             ║
║  directamente al pipeline.                                                   ║
║                                                                              ║
║  CURVAS EDITABLES:                                                           ║
║  [T] Tensión          — curva principal de tensión armónica 0-1             ║
║  [A] Actividad/Densidad — densidad rítmica 0-1 (→ --mt-density)            ║
║  [R] Registro         — registro melódico 0-1 (→ --mt-register)            ║
║  [H] Armonía          — complejidad armónica 0-1 (→ --mt-harmony-complexity)║
║  [S] Swing            — intensidad de swing 0-1 (→ --mt-swing)             ║
║                                                                              ║
║  CONTROLES DE TECLADO:                                                       ║
║    T/A/R/H/S    — seleccionar curva activa                                   ║
║    Click+drag   — dibujar / editar curva activa                              ║
║    C            — limpiar curva activa                                       ║
║    P            — añadir preset a la curva activa                            ║
║    G            — generar MIDI con las curvas actuales                       ║
║    E            — exportar curvas a JSON                                     ║
║    L            — cargar curvas desde JSON                                   ║
║    Z            — deshacer último cambio                                     ║
║    Q / Esc      — salir                                                      ║
║                                                                              ║
║  PRESETS DE TENSIÓN:                                                         ║
║    arch         — sube al centro y baja (arco)                               ║
║    crescendo    — sube progresivamente                                       ║
║    decrescendo  — baja progresivamente                                       ║
║    plateau      — plano en el centro, puntas bajas                           ║
║    late_climax  — clímax en el último tercio                                 ║
║    wave         — forma de ola (sube-baja-sube-baja)                        ║
║    neutral      — curva plana en 0.5                                         ║
║                                                                              ║
║  USO:                                                                        ║
║    python tension_designer.py fuente.mid [fuente2.mid]                      ║
║    python tension_designer.py a.mid b.mid --bars 32                         ║
║    python tension_designer.py a.mid --load curvas.json                      ║
║    python tension_designer.py a.mid --preset arch --auto-generate           ║
║    python tension_designer.py a.mid --no-gui --preset arch --bars 24        ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --bars N       Número de compases de la obra (default: 16)               ║
║    --load FILE    Cargar curvas desde JSON al inicio                         ║
║    --preset NAME  Aplicar preset de tensión al inicio                       ║
║    --no-gui       Modo sin interfaz: genera directamente con el preset       ║
║    --output FILE  MIDI de salida (default: tension_out.mid)                  ║
║    --mode MODE    Modo de midi_dna_unified (default: auto)                  ║
║    --verbose      Informe detallado                                          ║
║                                                                              ║
║  DEPENDENCIAS: matplotlib, pygame (solo para GUI), midi_dna_unified.py      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import subprocess
import time
import copy
import warnings

# Suprimir warnings de compatibilidad NumPy/SciPy al importar
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np

try:
    import matplotlib
    matplotlib.use('TkAgg')  # Backend interactivo
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.widgets import Button, RadioButtons
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import midi_dna_unified as dna_mod
    from midi_dna_unified import UnifiedDNA
    DNA_OK = True
except ImportError:
    DNA_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  PRESETS DE CURVAS
# ══════════════════════════════════════════════════════════════════════════════

TENSION_PRESETS = {
    'arch': lambda n: [
        0.1 + 0.85 * np.sin(np.pi * i / (n - 1))
        for i in range(n)
    ],
    'crescendo': lambda n: [
        0.05 + 0.9 * (i / (n - 1))
        for i in range(n)
    ],
    'decrescendo': lambda n: [
        0.95 - 0.9 * (i / (n - 1))
        for i in range(n)
    ],
    'plateau': lambda n: [
        0.2 + 0.7 * np.sin(np.pi * i / (n - 1)) ** 0.3
        for i in range(n)
    ],
    'late_climax': lambda n: [
        0.1 + 0.85 * np.sin(np.pi * i / (1.4 * (n - 1))) ** 2
        for i in range(n)
    ],
    'wave': lambda n: [
        0.4 + 0.5 * np.sin(2 * np.pi * i / (n - 1))
        for i in range(n)
    ],
    'neutral': lambda n: [0.5] * n,
    'dramatic': lambda n: [
        0.1 if i < n // 8 else
        0.1 + 0.85 * ((i - n//8) / (0.6 * (n - 1))) if i < int(0.7 * n) else
        0.95 - 0.8 * ((i - int(0.7 * n)) / (0.3 * (n - 1)))
        for i in range(n)
    ],
}

CURVE_COLORS = {
    'tension':    ('#e74c3c', 'Tensión [T]'),
    'activity':   ('#3498db', 'Actividad [A]'),
    'register':   ('#2ecc71', 'Registro [R]'),
    'harmony':    ('#9b59b6', 'Armonía [H]'),
    'swing':      ('#f39c12', 'Swing [S]'),
}

CURVE_KEYS = list(CURVE_COLORS.keys())

# ══════════════════════════════════════════════════════════════════════════════
#  DISEÑADOR DE CURVAS
# ══════════════════════════════════════════════════════════════════════════════

class CurveDesigner:
    def __init__(self, n_bars=16, initial_curves=None):
        self.n_bars = n_bars
        self.curves = {
            name: [0.5] * n_bars
            for name in CURVE_COLORS
        }
        if initial_curves:
            for k, v in initial_curves.items():
                if k in self.curves and len(v) == n_bars:
                    self.curves[k] = list(v)

        self.active_curve = 'tension'
        self.drawing = False
        self.history = []      # pila de deshacer
        self.fig = None
        self.ax = None
        self.line_objects = {}
        self.generated_midis = []

    # ── Inicialización de la figura ───────────────────────────────────────────

    def setup_figure(self):
        """Crea la interfaz matplotlib."""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('TENSION DESIGNER v1.0')
        self.fig.patch.set_facecolor('#1a1a2e')

        # Área principal del gráfico
        self.ax = self.fig.add_axes([0.05, 0.18, 0.72, 0.75])
        self.ax.set_facecolor('#16213e')
        self.ax.set_xlim(-0.5, self.n_bars - 0.5)
        self.ax.set_ylim(-0.05, 1.05)
        self.ax.set_xlabel('Compás', color='white', fontsize=11)
        self.ax.set_ylabel('Valor 0–1', color='white', fontsize=11)
        self.ax.set_title('TENSION DESIGNER — Click+drag para dibujar',
                          color='white', fontsize=13, pad=10)
        self.ax.tick_params(colors='white')
        self.ax.spines[:].set_color('#4a4a6a')
        self.ax.grid(True, color='#2a2a4a', linewidth=0.7, linestyle='--')

        # Líneas de referencia
        for ref in [0.25, 0.5, 0.75]:
            self.ax.axhline(ref, color='#3a3a5a', linewidth=0.5, linestyle=':')

        # Marcadores de compás
        for i in range(self.n_bars):
            self.ax.axvline(i, color='#2a2a4a', linewidth=0.3)

        # Dibujar líneas iniciales
        xs = list(range(self.n_bars))
        for name, (color, label) in CURVE_COLORS.items():
            line, = self.ax.plot(
                xs, self.curves[name],
                color=color, linewidth=2.5, alpha=0.75,
                marker='o', markersize=3, label=label
            )
            self.line_objects[name] = line

        # Resaltar curva activa
        self._highlight_active()

        # Leyenda
        self.ax.legend(loc='upper right', facecolor='#1a1a2e',
                       labelcolor='white', fontsize=9, framealpha=0.8)

        # ── Botones de acción ─────────────────────────────────────────────────
        btn_color = '#0f3460'
        btn_hover = '#16213e'

        def _make_btn(rect, label, callback):
            ax_btn = self.fig.add_axes(rect)
            ax_btn.set_facecolor(btn_color)
            btn = Button(ax_btn, label, color=btn_color, hovercolor=btn_hover)
            btn.label.set_color('white')
            btn.label.set_fontsize(9)
            btn.on_clicked(callback)
            return btn

        self.btn_clear    = _make_btn([0.79, 0.78, 0.09, 0.05], 'Limpiar [C]',   self._on_clear)
        self.btn_generate = _make_btn([0.79, 0.70, 0.09, 0.05], 'Generar [G]',   self._on_generate)
        self.btn_export   = _make_btn([0.79, 0.62, 0.09, 0.05], 'Exportar [E]',  self._on_export)
        self.btn_undo     = _make_btn([0.79, 0.54, 0.09, 0.05], 'Deshacer [Z]',  self._on_undo)

        # Presets
        ax_presets_label = self.fig.add_axes([0.79, 0.47, 0.09, 0.04])
        ax_presets_label.set_facecolor('#1a1a2e')
        ax_presets_label.axis('off')
        ax_presets_label.text(0.5, 0.5, 'PRESETS', ha='center', va='center',
                              color='#aaaacc', fontsize=8)

        preset_names = list(TENSION_PRESETS.keys())
        for i, pname in enumerate(preset_names[:4]):
            row = i // 2
            col = i % 2
            rect = [0.79 + col * 0.048, 0.32 + (1 - row) * 0.07, 0.044, 0.05]
            btn = _make_btn(rect, pname[:7], lambda _, p=pname: self._apply_preset(p))
            setattr(self, f'btn_preset_{pname}', btn)

        for i, pname in enumerate(preset_names[4:8]):
            row = (i + 4) // 2
            col = (i + 4) % 2
            rect = [0.79 + col * 0.048, 0.32 + (1 - row) * 0.07, 0.044, 0.05]
            btn = _make_btn(rect, pname[:7], lambda _, p=pname: self._apply_preset(p))
            setattr(self, f'btn_preset_{pname}', btn)

        # Selector de curva activa
        ax_radio = self.fig.add_axes([0.79, 0.03, 0.20, 0.22])
        ax_radio.set_facecolor('#16213e')
        labels = [CURVE_COLORS[k][1] for k in CURVE_KEYS]
        self.radio = RadioButtons(ax_radio, labels, active=0,
                                  activecolor='#e74c3c')
        for label in self.radio.labels:
            label.set_color('white')
            label.set_fontsize(8)
        self.radio.on_clicked(self._on_radio_change)

        # Barra de estado
        self.status_ax = self.fig.add_axes([0.05, 0.04, 0.72, 0.06])
        self.status_ax.set_facecolor('#0f3460')
        self.status_ax.axis('off')
        self.status_text = self.status_ax.text(
            0.02, 0.5,
            "Click+drag para dibujar  |  T/A/R/H/S = cambiar curva  |  G = generar  |  Q = salir",
            transform=self.status_ax.transAxes,
            color='#aaaaff', fontsize=9, va='center'
        )

        # Eventos
        self.fig.canvas.mpl_connect('button_press_event',   self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event',      self._on_key)

    def _highlight_active(self):
        """Resalta la curva activa aumentando su grosor y opacidad."""
        for name, line in self.line_objects.items():
            if name == self.active_curve:
                line.set_linewidth(4.0)
                line.set_alpha(1.0)
                line.set_markersize(5)
            else:
                line.set_linewidth(1.5)
                line.set_alpha(0.45)
                line.set_markersize(2)
        self.fig.canvas.draw_idle()

    def _update_status(self, text):
        self.status_text.set_text(text)
        self.fig.canvas.draw_idle()

    # ── Dibujo de curvas ──────────────────────────────────────────────────────

    def _xy_to_curve(self, x, y):
        """Convierte coordenadas de pantalla a índice de compás y valor."""
        bar = int(round(x))
        bar = max(0, min(self.n_bars - 1, bar))
        val = max(0.0, min(1.0, y))
        return bar, val

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        self._save_history()
        self.drawing = True
        bar, val = self._xy_to_curve(event.xdata, event.ydata)
        self.curves[self.active_curve][bar] = val
        self._redraw_curve(self.active_curve)

    def _on_release(self, event):
        self.drawing = False

    def _on_motion(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        bar, val = self._xy_to_curve(event.xdata, event.ydata)
        self.curves[self.active_curve][bar] = val
        self._redraw_curve(self.active_curve)

    def _redraw_curve(self, name):
        xs = list(range(self.n_bars))
        self.line_objects[name].set_ydata(self.curves[name])
        self.fig.canvas.draw_idle()

    # ── Historial ─────────────────────────────────────────────────────────────

    def _save_history(self):
        self.history.append(copy.deepcopy(self.curves))
        if len(self.history) > 50:
            self.history.pop(0)

    def _on_undo(self, event=None):
        if self.history:
            self.curves = self.history.pop()
            for name in CURVE_COLORS:
                self._redraw_curve(name)
            self._update_status("Deshacer aplicado.")

    # ── Botones ───────────────────────────────────────────────────────────────

    def _on_clear(self, event=None):
        self._save_history()
        self.curves[self.active_curve] = [0.5] * self.n_bars
        self._redraw_curve(self.active_curve)
        self._update_status(f"Curva '{self.active_curve}' limpiada.")

    def _apply_preset(self, preset_name, event=None):
        """Aplica un preset a la curva activa."""
        self._save_history()
        if preset_name in TENSION_PRESETS:
            vals = TENSION_PRESETS[preset_name](self.n_bars)
            self.curves[self.active_curve] = [float(v) for v in vals]
            self._redraw_curve(self.active_curve)
            self._update_status(f"Preset '{preset_name}' aplicado a '{self.active_curve}'.")

    def _on_radio_change(self, label):
        for key, (color, lbl) in CURVE_COLORS.items():
            if lbl == label:
                self.active_curve = key
                break
        self._highlight_active()
        self._update_status(f"Curva activa: {self.active_curve}")

    def _on_generate(self, event=None):
        """Exporta curvas y lanza la generación."""
        self._update_status("Generando MIDI… por favor espera.")
        self.fig.canvas.draw()
        curves_dict = self._get_curves_dict()
        # Llama a la función de generación (implementada en el main con los MIDIs cargados)
        if hasattr(self, '_generate_callback') and self._generate_callback:
            try:
                out_path = self._generate_callback(curves_dict)
                self._update_status(f"✓ MIDI generado: {out_path}")
                self.generated_midis.append(out_path)
            except Exception as e:
                self._update_status(f"✗ Error: {str(e)[:80]}")
        else:
            self._update_status("No hay callback de generación configurado.")

    def _on_export(self, event=None):
        """Exporta las curvas a JSON."""
        curves_dict = self._get_curves_dict()
        path = 'tension_curves.json'
        with open(path, 'w') as f:
            json.dump(curves_dict, f, indent=2)
        self._update_status(f"✓ Curvas exportadas: {path}")

    def _get_curves_dict(self):
        """Convierte las curvas internas al formato --mt-* de midi_dna_unified."""
        result = {}
        for name, vals in self.curves.items():
            # Generar string de spec para midi_dna_unified
            # Muestrear puntos de control (cada 2 compases aprox.)
            step = max(1, self.n_bars // 8)
            control_points = []
            for i in range(0, self.n_bars, step):
                control_points.append(f"{i}:{vals[i]:.3f}")
            # Asegurar punto final
            if not control_points or not control_points[-1].startswith(str(self.n_bars - 1)):
                control_points.append(f"{self.n_bars - 1}:{vals[-1]:.3f}")
            result[name] = {
                'values': [float(v) for v in vals],
                'mt_spec': ', '.join(control_points),
            }
        return result

    # ── Eventos de teclado ────────────────────────────────────────────────────

    def _on_key(self, event):
        key_map = {
            't': 'tension',
            'a': 'activity',
            'r': 'register',
            'h': 'harmony',
            's': 'swing',
        }
        if event.key.lower() in key_map:
            self.active_curve = key_map[event.key.lower()]
            self._highlight_active()
            # Sincronizar radio button
            idx = CURVE_KEYS.index(self.active_curve)
            self.radio.set_active(idx)
            self._update_status(f"Curva activa: {self.active_curve}")
        elif event.key.lower() == 'c':
            self._on_clear()
        elif event.key.lower() == 'g':
            self._on_generate()
        elif event.key.lower() == 'e':
            self._on_export()
        elif event.key.lower() == 'z':
            self._on_undo()
        elif event.key.lower() in ('q', 'escape'):
            plt.close(self.fig)

    def run(self, generate_callback=None):
        """Abre la GUI y espera interacción."""
        self._generate_callback = generate_callback
        self.setup_figure()
        plt.show()
        return self._get_curves_dict()


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSIÓN DE CURVAS A FLAGS DE MIDI_DNA_UNIFIED
# ══════════════════════════════════════════════════════════════════════════════

def curves_to_flags(curves_dict, n_bars):
    """
    Convierte el diccionario de curvas a los flags --mt-* de midi_dna_unified.
    Retorna lista de strings de argumentos.
    """
    flags = []

    curve_to_flag = {
        'activity': '--mt-density',
        'register': '--mt-register',
        'harmony':  '--mt-harmony-complexity',
        'swing':    '--mt-swing',
    }

    for curve_name, flag in curve_to_flag.items():
        if curve_name in curves_dict:
            spec = curves_dict[curve_name]['mt_spec']
            flags.extend([flag, spec])

    # La curva de tensión no tiene un flag directo en midi_dna_unified
    # (se usa internamente), pero la exportamos como metadato
    return flags


def apply_tension_to_dna(tension_values, source_dna):
    """
    Inyecta la curva de tensión diseñada directamente en el DNA fuente.
    Retorna una copia modificada del DNA.
    """
    modified = copy.deepcopy(source_dna)
    n = len(tension_values)
    if n > 0:
        modified.tension_curve = list(tension_values)
        # Actualizar el arco emocional según la forma de la curva
        arr = np.array(tension_values)
        peak_pos = np.argmax(arr) / max(n - 1, 1)
        if peak_pos > 0.6:
            modified.emotional_arc_label = 'late_climax'
        elif arr[-1] > arr[0] + 0.2:
            modified.emotional_arc_label = 'crescendo'
        elif arr[0] > arr[-1] + 0.2:
            modified.emotional_arc_label = 'decrescendo'
        elif max(arr) - min(arr) < 0.15:
            modified.emotional_arc_label = 'plateau'
        else:
            modified.emotional_arc_label = 'arch'
    return modified


# ══════════════════════════════════════════════════════════════════════════════
#  MODO SIN GUI (batch)
# ══════════════════════════════════════════════════════════════════════════════

def generate_with_preset(preset_name, midi_paths, n_bars, mode,
                          output, mixer_script, verbose=False,
                          key=None, tempo=None):
    """Genera un MIDI directamente con un preset sin abrir la GUI."""
    if preset_name not in TENSION_PRESETS:
        print(f"ERROR: preset desconocido '{preset_name}'. Disponibles: {list(TENSION_PRESETS)}")
        sys.exit(1)

    tension_vals = TENSION_PRESETS[preset_name](n_bars)
    # Crear specs mt-*
    step = max(1, n_bars // 8)
    control_pts = [f"{i}:{tension_vals[i]:.3f}" for i in range(0, n_bars, step)]
    control_pts.append(f"{n_bars-1}:{tension_vals[-1]:.3f}")
    spec = ', '.join(control_pts)

    print(f"  Preset '{preset_name}' → {n_bars} compases")
    print(f"  Curva de tensión: {spec[:60]}…")

    cmd = [sys.executable, mixer_script] + (midi_paths or [])
    cmd += ['--output', output, '--bars', str(n_bars), '--mode', mode]
    if key:
        cmd += ['--key', key]
    if tempo:
        cmd += ['--tempo', str(tempo)]
    # Usar la tensión como guía de densidad y registro
    cmd += ['--mt-density', spec, '--mt-register', spec]

    if verbose:
        print(f"  CMD: {' '.join(cmd[:10])} …")

    result = subprocess.run(cmd, capture_output=not verbose, text=True)
    if result.returncode == 0:
        print(f"  ✓ MIDI generado: {output}")
    else:
        print(f"  ✗ Error en generación")
        if not verbose and result.stderr:
            print(result.stderr[-500:])
    return result.returncode == 0


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='TENSION DESIGNER — Diseñador visual de curvas de tensión',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('inputs', nargs='*', help='MIDIs fuente')
    parser.add_argument('--bars',    type=int, default=16, help='Número de compases (default: 16)')
    parser.add_argument('--load',    default=None, help='Cargar curvas desde JSON')
    parser.add_argument('--preset',  default=None,
                        choices=list(TENSION_PRESETS.keys()),
                        help='Preset de tensión inicial')
    parser.add_argument('--no-gui',  action='store_true',
                        help='Modo sin GUI: generar directamente con preset')
    parser.add_argument('--auto-generate', action='store_true',
                        help='Generar automáticamente al cerrar la GUI')
    parser.add_argument('--output',  default='tension_out.mid')
    parser.add_argument('--key',     default=None, help='Tonalidad (ej: C, D, Fm).')
    parser.add_argument('--tempo',   default=None, type=int, help='Tempo en BPM.')
    parser.add_argument('--mode',    default='auto',
                        choices=['auto','rhythm_melody','harmony_melody','full_blend',
                                 'custom','mosaic','energy','emotion'])
    parser.add_argument('--mixer',   default=None,
                        help='Ruta a midi_dna_unified.py')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Localizar mixer
    mixer_script = args.mixer
    if not mixer_script:
        for c in [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'midi_dna_unified.py'),
                  'midi_dna_unified.py']:
            if os.path.exists(c):
                mixer_script = c
                break
    if not mixer_script or not os.path.exists(mixer_script):
        print("ERROR: No se encontró midi_dna_unified.py. Usa --mixer para especificar la ruta.")
        sys.exit(1)

    midi_paths = [p for p in (args.inputs or []) if os.path.exists(p)]
    if not midi_paths and not args.no_gui:
        print("AVISO: No se especificaron MIDIs fuente. La generación no estará disponible.")

    print("═" * 65)
    print("  TENSION DESIGNER v1.0")
    print("═" * 65)
    print(f"  Compases : {args.bars}")
    if midi_paths:
        print(f"  Fuentes  : {', '.join(os.path.basename(p) for p in midi_paths)}")

    # ── Modo sin GUI ──────────────────────────────────────────────────────────
    if args.no_gui:
        if not args.preset:
            print("ERROR: --no-gui requiere --preset")
            sys.exit(1)
        # midi_paths es opcional: sin fuentes MIDI genera desde el preset standalone
        ok = generate_with_preset(
            args.preset, midi_paths, args.bars, args.mode,
            args.output, mixer_script, args.verbose,
            key=args.key, tempo=args.tempo,
        )
        sys.exit(0 if ok else 1)

    # ── Modo GUI ──────────────────────────────────────────────────────────────
    if not MATPLOTLIB_OK:
        print("ERROR: matplotlib no disponible. Instala con: pip install matplotlib")
        print("       Usa --no-gui para modo sin interfaz.")
        sys.exit(1)

    # Cargar curvas iniciales
    initial_curves = None
    if args.load and os.path.exists(args.load):
        with open(args.load) as f:
            raw = json.load(f)
        initial_curves = {k: v['values'] for k, v in raw.items() if 'values' in v}
        print(f"  ✓ Curvas cargadas desde {args.load}")

    designer = CurveDesigner(n_bars=args.bars, initial_curves=initial_curves)

    # Aplicar preset inicial si se especificó
    if args.preset:
        designer.curves['tension'] = [
            float(v) for v in TENSION_PRESETS[args.preset](args.bars)
        ]
        print(f"  ✓ Preset inicial '{args.preset}' aplicado")

    def generate_callback(curves_dict):
        """Callback llamado desde el botón G de la GUI."""
        if not midi_paths:
            raise RuntimeError("No hay MIDIs fuente especificados")

        mt_flags = curves_to_flags(curves_dict, args.bars)

        # Usar la tensión diseñada para guiar la densidad también
        tension_spec = curves_dict['tension']['mt_spec']

        cmd = [sys.executable, mixer_script] + midi_paths
        cmd += ['--output', args.output, '--bars', str(args.bars), '--mode', args.mode]
        cmd += mt_flags

        if args.verbose:
            print(f"\n  CMD: {' '.join(cmd[:12])} …")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-300:] if result.stderr else "Error desconocido")

        return args.output

    # Ejecutar GUI
    print("\n  Abriendo interfaz gráfica…")
    print("  Controles: T/A/R/H/S = curva  |  G = generar  |  Q = salir\n")
    final_curves = designer.run(generate_callback=generate_callback)

    # Auto-generar al cerrar si se pidió
    if args.auto_generate and midi_paths and final_curves:
        print("\n  Auto-generando con las curvas diseñadas…")
        mt_flags = curves_to_flags(final_curves, args.bars)
        cmd = [sys.executable, mixer_script] + midi_paths
        cmd += ['--output', args.output, '--bars', str(args.bars), '--mode', args.mode]
        cmd += mt_flags
        result = subprocess.run(cmd, capture_output=not args.verbose, text=True, timeout=120)
        if result.returncode == 0:
            print(f"  ✓ MIDI final: {args.output}")
        else:
            print("  ✗ Error en generación final")

    # Exportar siempre las curvas al salir
    curves_path = args.output.replace('.mid', '_curves.json')
    if not curves_path.endswith('.json'):
        curves_path += '_curves.json'
    with open(curves_path, 'w') as f:
        json.dump(final_curves, f, indent=2)
    print(f"  ✓ Curvas guardadas: {curves_path}")
    print("  Puedes recargarlas con: --load " + curves_path)


if __name__ == '__main__':
    main()
