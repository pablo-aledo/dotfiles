"""
renderers/html_report.py  v2
════════════════════════════
Renderer de informe estadístico interactivo (salida .html).
Genera HTML autocontenido con Chart.js para gráficas de curvas temporales,
tablas de cadencias, SSM coloreada, fases emocionales y métricas clave.

No contiene texto interpretativo — solo datos y visualizaciones.
"""

from __future__ import annotations
import math, json
from typing import Dict, List, Any

# ── Helpers ──────────────────────────────────────────────────────────────────

def _ts(sec):
    s = float(sec) if sec is not None else 0.0
    return f"{int(s//60)}:{int(s%60):02d}"

def _safe(val, default=0):
    return val if val is not None else default

def _pct_bar(value: float, max_val: float = 1.0, width: int = 90) -> str:
    pct = min(100, max(0, (value / max_val * 100) if max_val > 0 else 0))
    color = '#1D9E75' if pct > 65 else '#EF9F27' if pct > 35 else '#E24B4A'
    return (
        f'<div style="display:inline-flex;align-items:center;gap:6px">'
        f'<div style="background:#e5e7eb;border-radius:4px;height:7px;width:{width}px">'
        f'<div style="background:{color};border-radius:4px;height:7px;width:{pct*width/100:.0f}px"></div>'
        f'</div>'
        f'<span style="font-size:11px;color:#64748b;min-width:30px">{pct:.0f}%</span>'
        f'</div>'
    )

def _extract_curve(curve_data, key_x='time', key_y=None, max_pts=250):
    """
    Extrae (labels_list, data_list) de cualquier formato de curva del core:
      - Lista de TensionPoint (objetos con .time y .tension)
      - Lista de tuples/listas [t, v]
      - Lista de dicts {time: t, tension/valence/...: v}
      - Lista de valores escalares
    Devuelve dos listas Python listas para json.dumps().
    """
    if not curve_data:
        return [], []

    first = curve_data[0]

    # TensionPoint o cualquier objeto con atributos
    if hasattr(first, 'time') and not isinstance(first, (list, tuple, dict)):
        kv = key_y or 'tension'
        labels = [round(float(getattr(p, 'time', 0)), 2) for p in curve_data]
        data   = [round(float(getattr(p, kv, getattr(p, 'tension', 0))), 4) for p in curve_data]

    # tuple o list: [t, v] o (t, v)
    elif isinstance(first, (list, tuple)):
        labels = [round(float(p[0]), 2) for p in curve_data]
        data   = [round(float(p[1]), 4) for p in curve_data]

    # dict
    elif isinstance(first, dict):
        kv = key_y or 'tension'
        labels = [round(float(p.get(key_x, p.get('time', 0))), 2) for p in curve_data]
        data   = [round(float(p.get(kv, p.get('value', 0))), 4) for p in curve_data]

    # scalar
    else:
        labels = list(range(len(curve_data)))
        data   = [round(float(v), 4) for v in curve_data]

    # Downsample
    step = max(1, len(labels) // max_pts)
    return labels[::step], data[::step]


def _line_chart_html(canvas_id: str, label: str, color: str,
                     curve_data, key_y=None,
                     y_min=None, y_max=None) -> str:
    """
    Genera el bloque <script> que inicializa un Chart.js line chart.
    Evita cualquier colisión de nombre de variable usando prefijos únicos.
    """
    labels_list, data_list = _extract_curve(curve_data, key_y=key_y)
    if not labels_list:
        return f'<script>/* {canvas_id}: sin datos */</script>'

    var = canvas_id.replace('-', '_')
    y_min_js = 'undefined' if y_min is None else str(y_min)
    y_max_js = 'undefined' if y_max is None else str(y_max)

    return f"""<script>
var {var}_ctx = document.getElementById('{canvas_id}');
if ({var}_ctx) {{
  new Chart({var}_ctx.getContext('2d'), {{
    type: 'line',
    data: {{
      labels: {json.dumps(labels_list)},
      datasets: [{{
        label: '{label}',
        data: {json.dumps(data_list)},
        borderColor: '{color}',
        backgroundColor: '{color}22',
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.35
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ mode: 'index', intersect: false }}
      }},
      scales: {{
        x: {{ display: false }},
        y: {{
          min: {y_min_js},
          max: {y_max_js},
          grid: {{ color: '#f1f5f9' }},
          ticks: {{ font: {{ size: 10 }}, color: '#94a3b8', maxTicksLimit: 5 }}
        }}
      }}
    }}
  }});
}}
</script>"""


def _multi_line_chart_html(canvas_id: str, datasets_spec: list) -> str:
    """
    Genera script para múltiples series en el mismo canvas.
    datasets_spec: [{'label','color','curve_data','key_y'}, ...]
    """
    var = canvas_id.replace('-', '_')
    shared_labels = None
    ds_js = []

    for i, ds in enumerate(datasets_spec):
        lls, dls = _extract_curve(ds['curve_data'], key_y=ds.get('key_y'))
        if shared_labels is None:
            shared_labels = lls
        ds_js.append({
            'label': ds['label'],
            'data': dls,
            'borderColor': ds['color'],
            'borderWidth': 1.5,
            'pointRadius': 0,
            'fill': False,
            'tension': 0.35
        })

    if not shared_labels:
        return f'<script>/* {canvas_id}: sin datos */</script>'

    return f"""<script>
(function(){{
  var {var}_labels = {json.dumps(shared_labels)};
  var {var}_ds = {json.dumps(ds_js)};
  {var}_ds.forEach(function(d, i) {{
    var src = [{', '.join('[' + json.dumps(ds['data'] if 'data' in ds else []) + ']'
                         for ds in ds_js)}][i];
    d.data = src;
  }});
  var {var}_ctx = document.getElementById('{canvas_id}');
  if (!{var}_ctx) return;
  new Chart({var}_ctx.getContext('2d'), {{
    type: 'line',
    data: {{ labels: {var}_labels, datasets: {var}_ds }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {{ legend: {{ position: 'top', labels: {{ font: {{ size: 10 }} }} }} }},
      scales: {{
        x: {{ display: false }},
        y: {{ grid: {{ color: '#f1f5f9' }},
              ticks: {{ font: {{ size: 10 }}, color: '#94a3b8', maxTicksLimit: 5 }} }}
      }}
    }}
  }});
}})();
</script>"""

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f8fafc; color: #1e293b; padding: 24px; font-size: 14px; }
h1 { font-size: 22px; font-weight: 600; margin-bottom: 4px; }
h2 { font-size: 15px; font-weight: 600; color: #334155; margin: 20px 0 10px;
     padding-bottom: 6px; border-bottom: 1.5px solid #e2e8f0; }
h3 { font-size: 13px; font-weight: 600; color: #475569; margin: 12px 0 6px; }
.meta { font-size: 13px; color: #64748b; margin-bottom: 24px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.grid4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 14px; }
@media (max-width: 900px) { .grid2,.grid3,.grid4 { grid-template-columns: 1fr; } }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 16px; }
.card-full { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
             padding: 20px; margin-bottom: 20px; }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; padding: 6px 8px; background: #f1f5f9;
     border-bottom: 1px solid #e2e8f0; font-weight: 500; color: #475569; }
td { padding: 5px 8px; border-bottom: 1px solid #f8fafc; color: #334155; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px;
         font-size: 11px; font-weight: 500; background: #e0f2fe; color: #0369a1;
         white-space: nowrap; }
.badge-green { background: #dcfce7; color: #15803d; }
.badge-amber { background: #fef9c3; color: #92400e; }
.badge-red   { background: #fee2e2; color: #b91c1c; }
.badge-purple{ background: #f3e8ff; color: #6d28d9; }
.chart-wrap { position: relative; height: 170px; }
.lbl { font-size: 10px; font-weight: 600; color: #94a3b8; text-transform: uppercase;
       letter-spacing: 0.05em; margin-bottom: 8px; }
.metric { display: flex; justify-content: space-between; align-items: center;
          padding: 5px 0; border-bottom: 1px dashed #f1f5f9; font-size: 12px; }
.metric:last-child { border-bottom: none; }
.metric-key { color: #64748b; }
.metric-val { font-weight: 500; color: #0f172a; text-align: right; }
.section-divider { margin: 28px 0 4px; font-size: 11px; font-weight: 600;
                   color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em;
                   border-top: 1px solid #e2e8f0; padding-top: 12px; }
"""

_CHART_JS_CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"


# ── Section builders ──────────────────────────────────────────────────────────

def _s_identity(r: dict) -> str:
    gd = r.get('genre_detection') or {}
    se = r.get('semantic_enrichment') or {}
    notes = r.get('notes') or []
    vel = [n.velocity for n in notes] if notes else [64]
    dur = _safe(r.get('total_dur'))

    metrics = [
        ('Tonalidad',       r.get('key_name', '—')),
        ('Modo',            r.get('mode', '—')),
        ('Confianza KS',    f"{_safe(r.get('kconf')):.3f}"),
        ('Tempo principal', f"{_safe(r.get('main_bpm')):.1f} BPM"),
        ('Tempo medio',     f"{_safe(r.get('avg_bpm')):.1f} BPM"),
        ('Compás',          r.get('ts_str_val', '4/4')),
        ('Duración',        f"{int(dur//60)}m{int(dur%60):02d}s"),
        ('Total notas',     str(r.get('notes_count', len(notes)))),
        ('Velocidad',       f"pp={min(vel)}  ff={max(vel)}  μ={sum(vel)/len(vel):.0f}"),
    ]
    metric_rows = ''.join(
        f'<div class="metric"><span class="metric-key">{k}</span>'
        f'<span class="metric-val">{v}</span></div>'
        for k, v in metrics
    )

    # Genre top3
    top3 = gd.get('top3') or []
    genre_rows = ''
    for i, g in enumerate(top3[:3]):
        icon = '●' if i == 0 else '○'
        genre_rows += (
            f'<tr><td style="color:#475569">{icon} {g["label"]}</td>'
            f'<td>{_pct_bar(g["score"])}</td></tr>'
        )

    genre_block = f'''
<h3>Género detectado</h3>
<p style="margin-bottom:8px">
  <strong>{gd.get("genre_label","—")}</strong>
  {f'<span style="color:#94a3b8"> / {gd["subgenre"]}</span>' if gd.get("subgenre") else ""}
  &nbsp;
  <span class="badge {'badge-green' if gd.get('confidence',0)>0.65 else 'badge-amber'}">
    {gd.get("confidence",0):.0%}
  </span>
</p>
<table><tbody>{genre_rows}</tbody></table>
<p style="font-size:11px;color:#94a3b8;margin-top:6px">{gd.get("key_signature","")}</p>
''' if gd.get('genre_label') else '<p style="color:#94a3b8;font-size:12px">sin detección de género</p>'

    sem_block = f'''
<h3 style="margin-top:14px">Affektenlehre</h3>
<div class="metric"><span class="metric-key">Concepto</span>
  <span class="metric-val">{se["concept"]} <span class="badge badge-purple">{se.get("fit_score",0):.0%}</span></span></div>
<div class="metric"><span class="metric-key">Arco</span>
  <span class="metric-val">{se.get("arc_desc","")}</span></div>
<div class="metric"><span class="metric-key">Persona afín</span>
  <span class="metric-val">{se.get("persona_name","")}</span></div>
''' if se.get('concept') else ''

    return f'''
<div class="card-full">
  <div class="grid3">
    <div>
      <div class="lbl">Ficha técnica</div>
      {metric_rows}
    </div>
    <div>
      {genre_block}
    </div>
    <div>
      {sem_block}
    </div>
  </div>
</div>'''


def _s_curves(r: dict) -> str:
    tc  = r.get('tension_curve') or []
    dv  = r.get('dynamic_valence') or {}
    ep  = r.get('energy_profile') or {}
    rg  = r.get('roughness') or {}
    per = r.get('perceptual') or {}

    dv_curve = dv.get('curve') or []
    ep_curve = ep.get('curve') or []
    rg_curve = rg.get('curve_norm') or rg.get('curve') or []

    # Stats for captions
    tc_vals = []
    for p in tc:
        if hasattr(p, 'tension'):
            tc_vals.append(p.tension)
        elif isinstance(p, (list, tuple)):
            tc_vals.append(p[1])
    tc_mean = sum(tc_vals) / len(tc_vals) if tc_vals else 0

    scripts = (
        _line_chart_html('chart-tension',  'Tensión',   '#ef4444', tc,       key_y='tension',  y_min=0, y_max=1) +
        _line_chart_html('chart-valence',  'Valencia',  '#6366f1', dv_curve, y_min=-1, y_max=1) +
        _line_chart_html('chart-energy',   'Energía',   '#10b981', ep_curve) +
        _line_chart_html('chart-roughness','Roughness', '#f59e0b', rg_curve, y_min=0)
    )

    def cap(key, unit=''):
        val = _safe(rg.get(key) if 'roughness' in key else
                    dv.get(key) if 'valence' in key else
                    ep.get(key) if 'energy' in key else 0)
        return f'{val:.3f}{unit}'

    return f'''
<div class="card-full">
  <h2>Curvas temporales continuas</h2>
  <div class="grid4">

    <div class="card">
      <div class="lbl">Tensión armónica</div>
      <div class="chart-wrap"><canvas id="chart-tension"></canvas></div>
      <p style="font-size:11px;color:#94a3b8;margin-top:6px">
        μ = {tc_mean:.3f} · escala [0, 1]
      </p>
    </div>

    <div class="card">
      <div class="lbl">Valencia emocional</div>
      <div class="chart-wrap"><canvas id="chart-valence"></canvas></div>
      <p style="font-size:11px;color:#94a3b8;margin-top:6px">
        μ = {dv.get("mean_valence",0):+.3f} · arco: {dv.get("arc_shape","—")}
      </p>
    </div>

    <div class="card">
      <div class="lbl">Energía física</div>
      <div class="chart-wrap"><canvas id="chart-energy"></canvas></div>
      <p style="font-size:11px;color:#94a3b8;margin-top:6px">
        pico en {_ts(ep.get("peak_time",0))}
      </p>
    </div>

    <div class="card">
      <div class="lbl">Roughness (Plomp-Levelt)</div>
      <div class="chart-wrap"><canvas id="chart-roughness"></canvas></div>
      <p style="font-size:11px;color:#94a3b8;margin-top:6px">
        μ = {rg.get("mean_roughness",0):.3f} · arco: {rg.get("roughness_arc","—")}
      </p>
    </div>

  </div>
  {scripts}
</div>'''


def _s_emotional_map(r: dict) -> str:
    uem  = r.get('unified_emotional_map') or {}
    traj = r.get('emotional_trajectory') or {}
    cat  = r.get('catharsis') or {}
    pol  = r.get('polarity') or {}
    total_dur = _safe(r.get('total_dur'), 1)

    phases = uem.get('emotional_phases') or []
    COLORS = {
        'Euforia': '#f59e0b', 'Éxtasis quieto': '#a78bfa',
        'Angustia activa': '#ef4444', 'Terror helado': '#7c3aed',
        'Alegría plena': '#10b981', 'Serenidad': '#3b82f6',
        'Melancolía viva': '#6366f1', 'Vacío sereno': '#94a3b8', 'Ambiguo': '#e5e7eb',
    }

    # Timeline
    timeline = '<div style="display:flex;height:28px;border-radius:6px;overflow:hidden;margin:10px 0">'
    for ph in phases:
        pct  = ph['duration'] / total_dur * 100
        col  = COLORS.get(ph['label'], '#e5e7eb')
        tip  = f"{ph['label']} ({ph['start_str']}→{ph['end_str']}, {pct:.0f}%)"
        txt  = ph['label'][:5] if pct > 8 else ''
        timeline += (
            f'<div style="width:{pct:.2f}%;background:{col};display:flex;align-items:center;'
            f'justify-content:center;font-size:9px;color:#fff;font-weight:600;overflow:hidden" title="{tip}">{txt}</div>'
        )
    timeline += '</div>'

    # Legend
    legend = '<div style="display:flex;flex-wrap:wrap;gap:6px;font-size:11px;margin-bottom:14px">'
    seen = []
    for ph in phases:
        if ph['label'] not in seen:
            seen.append(ph['label'])
            col = COLORS.get(ph['label'], '#ccc')
            legend += (
                f'<span><span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:2px;background:{col};margin-right:3px"></span>{ph["label"]}</span>'
            )
    legend += '</div>'

    # Inflection table
    inflex = uem.get('inflection_points') or []
    inflex_rows = ''.join(
        f'<tr><td style="font-family:monospace">{ip["time_str"]}</td>'
        f'<td>{ip.get("from_label","")}</td>'
        f'<td style="color:#94a3b8">→</td>'
        f'<td>{ip.get("to_label","")}</td>'
        f'<td style="text-align:right;color:#ef4444">{ip.get("delta",0):.3f}</td></tr>'
        for ip in inflex[:6]
    ) or '<tr><td colspan="5" style="color:#94a3b8">sin giros bruscos</td></tr>'

    # VA quadrant
    qt = traj.get('quadrant_time') or {}
    qt_rows = ''.join(
        f'<tr><td>{q}</td><td>{_pct_bar(d / total_dur)}</td></tr>'
        for q, d in sorted(qt.items(), key=lambda x: -x[1])
    )

    cat_type = cat.get('catharsis_type', 'ausente')
    cat_cls  = 'badge-green' if cat_type not in ('ausente', 'negada') else 'badge-red' if cat_type == 'negada' else ''

    return f'''
<div class="card-full">
  <h2>Mapa emocional unificado</h2>
  {timeline}
  {legend}
  <div class="grid3">
    <div class="card">
      <div class="lbl">Giros emocionales ({len(inflex)})</div>
      <table>
        <thead><tr><th>t</th><th>desde</th><th></th><th>hacia</th><th>Δ</th></tr></thead>
        <tbody>{inflex_rows}</tbody>
      </table>
      <div class="metric" style="margin-top:10px">
        <span class="metric-key">Coherencia inter-dim.</span>
        <span class="metric-val">{uem.get("mean_coherence",0):.3f}</span>
      </div>
    </div>
    <div class="card">
      <div class="lbl">Trayectoria VA</div>
      <div class="metric"><span class="metric-key">Forma</span>
        <span class="metric-val">{traj.get("path_shape","—")}</span></div>
      <div class="metric"><span class="metric-key">Distancia total</span>
        <span class="metric-val">{traj.get("total_distance",0):.3f}</span></div>
      <div class="metric"><span class="metric-key">Eficiencia</span>
        <span class="metric-val">{traj.get("efficiency",0):.2f}</span></div>
      <div class="metric"><span class="metric-key">Centroide</span>
        <span class="metric-val">({(traj.get("centroid") or (0,0))[0]:+.2f}, {(traj.get("centroid") or (0,0))[1]:+.2f})</span></div>
      <div class="metric"><span class="metric-key">Cuadrante dom.</span>
        <span class="metric-val">{traj.get("dominant_quadrant","—")}</span></div>
      <h3>Tiempo por cuadrante</h3>
      <table><tbody>{qt_rows}</tbody></table>
    </div>
    <div class="card">
      <div class="lbl">Catarsis &amp; polaridad</div>
      <div class="metric"><span class="metric-key">Tipo catarsis</span>
        <span class="metric-val"><span class="badge {cat_cls}">{cat_type}</span></span></div>
      <div class="metric"><span class="metric-key">Fuerza</span>
        <span class="metric-val">{cat.get("release_strength",0):.2f}</span></div>
      <div class="metric"><span class="metric-key">Resolución</span>
        <span class="metric-val">{cat.get("resolution_quality",0):.2f}</span></div>
      {'<div class="metric"><span class="metric-key">Momento</span><span class="metric-val">' + _ts(cat.get("moment")) + '</span></div>' if cat.get("moment") else ""}
      <div class="metric" style="margin-top:8px"><span class="metric-key">Polaridad</span>
        <span class="metric-val">{pol.get("polarity_type","—")}</span></div>
    </div>
  </div>
</div>'''


def _s_ssm(r: dict) -> str:
    ssm = r.get('ssm') or {}
    matrix     = ssm.get('matrix') or []
    fl         = ssm.get('form_labels') or []
    n          = ssm.get('n') or 0
    total_dur  = _safe(r.get('total_dur'), 1)

    if not matrix or n == 0:
        matrix_html = '<p style="color:#94a3b8">SSM no disponible</p>'
    else:
        max_s = min(n, 28)
        step  = max(1, n // max_s)
        idxs  = list(range(0, n, step))[:max_s]

        def sim_col(v):
            v = min(1, max(0, float(v)))
            r2 = int(255 - v * 216)
            g2 = int(255 - v * 121)
            b2 = int(255 - v * 107)
            return f'#{r2:02x}{g2:02x}{b2:02x}'

        rows_html = ['<table style="border-collapse:collapse;font-size:10px">']
        rows_html.append('<tr><td></td>' +
                         ''.join(f'<td style="text-align:center;width:14px;color:#94a3b8">'
                                 f'{fl[i] if i < len(fl) else ""}</td>' for i in idxs) +
                         '</tr>')
        for ri in idxs:
            lbl = fl[ri] if ri < len(fl) else ''
            rows_html.append(f'<tr><td style="color:#94a3b8;padding-right:3px;font-size:10px">{lbl}</td>')
            for ci in idxs:
                try:
                    v = float(matrix[ri][ci])
                except (IndexError, TypeError):
                    v = 0
                rows_html.append(
                    f'<td style="background:{sim_col(v)};width:14px;height:14px" title="{v:.2f}"></td>'
                )
            rows_html.append('</tr>')
        rows_html.append('</table>')
        matrix_html = '\n'.join(rows_html)

    blocks = ssm.get('block_structure') or []
    block_rows = ''.join(
        f'<tr><td><strong>{b.get("label","?")}</strong></td>'
        f'<td>{_ts(b.get("start",0))}</td>'
        f'<td>{_ts(b.get("end",0))}</td>'
        f'<td>{(b.get("end",0)-b.get("start",0))/total_dur*100:.0f}%</td></tr>'
        for b in blocks
    )

    return f'''
<div class="card-full">
  <h2>Self-Similarity Matrix &amp; forma musical</h2>
  <div class="grid2">
    <div>
      <div class="lbl">Similitud entre segmentos</div>
      {matrix_html}
      <p style="font-size:11px;color:#94a3b8;margin-top:6px">
        Forma: <strong>{ssm.get("form_str","?")}</strong> ·
        {ssm.get("n_unique",0)} célula(s) ·
        simetría espejo: {ssm.get("symmetry",0):.2f}
      </p>
    </div>
    <div>
      <div class="card" style="margin-bottom:12px">
        <div class="lbl">Datos formales</div>
        <div class="metric"><span class="metric-key">Forma</span>
          <span class="metric-val">{ssm.get("form_canon","—")}</span></div>
        <div class="metric"><span class="metric-key">Patrón</span>
          <span class="metric-val" style="font-family:monospace">{ssm.get("form_str","—")}</span></div>
        <div class="metric"><span class="metric-key">Células únicas</span>
          <span class="metric-val">{ssm.get("n_unique",0)}</span></div>
        <div class="metric"><span class="metric-key">Repeticiones lit.</span>
          <span class="metric-val">{len(ssm.get("repetitions") or [])}</span></div>
        <div class="metric"><span class="metric-key">Simetría espejo</span>
          <span class="metric-val">{ssm.get("symmetry",0):.2f}</span></div>
      </div>
      <table>
        <thead><tr><th>Bloque</th><th>Inicio</th><th>Fin</th><th>%</th></tr></thead>
        <tbody>{block_rows}</tbody>
      </table>
    </div>
  </div>
</div>'''


def _s_harmony(r: dict) -> str:
    cad = r.get('cadences') or {}
    hg  = r.get('harmonic_graph') or {}
    tg  = r.get('tonal_gravity') or {}
    ta  = r.get('tonal_ambiguity') or {}
    fp  = r.get('fingerprint') or {}
    mm  = r.get('modal_borrow') or []
    cp  = r.get('canon_progs') or []
    ch  = r.get('chromaticism') or {}
    dr  = r.get('dissonance_res') or {}

    cad_c = cad.get('counts') or {}
    cad_items = (cad_c.most_common() if hasattr(cad_c, 'most_common')
                 else sorted(cad_c.items(), key=lambda x: -x[1]) if isinstance(cad_c, dict) else [])
    cad_rows = ''.join(
        f'<tr><td>{k}</td><td style="text-align:right">{v}</td></tr>'
        for k, v in cad_items
    ) or '<tr><td colspan="2" style="color:#94a3b8">sin cadencias</td></tr>'

    fp_data = fp.get('fingerprint') or {}
    fp_items = [
        ('Acordes mayores',  fp_data.get('maj_ratio', 0)),
        ('Acordes menores',  fp_data.get('min_ratio', 0)),
        ('Extensiones 7ª+',  fp_data.get('extension_ratio', 0)),
        ('Mov. por 5ª',      fp_data.get('fifths_preference', 0)),
        ('Mov. por 3ª',      fp_data.get('thirds_preference', 0)),
        ('Mov. semitonal',   fp_data.get('semitone_preference', 0)),
    ]
    fp_rows = ''.join(
        f'<div class="metric"><span class="metric-key">{k}</span>'
        f'<span class="metric-val">{_pct_bar(v)}</span></div>'
        for k, v in fp_items
    )

    return f'''
<div class="card-full">
  <h2>Análisis armónico</h2>
  <div class="grid3">
    <div class="card">
      <div class="lbl">Cadencias</div>
      <div class="metric"><span class="metric-key">Total</span>
        <span class="metric-val">{_safe(cad.get("total"))}</span></div>
      <div class="metric"><span class="metric-key">Tipo dominante</span>
        <span class="metric-val">{cad.get("dominant_type","—")}</span></div>
      <table style="margin-top:8px">
        <thead><tr><th>Tipo</th><th>N</th></tr></thead>
        <tbody>{cad_rows}</tbody>
      </table>
      <h3 style="margin-top:12px">Progresiones canónicas</h3>
      <ul style="font-size:12px;padding-left:16px;color:#334155;margin-top:4px">
        {''.join(f"<li>{p}</li>" for p in cp[:6]) or '<li style="color:#94a3b8">ninguna</li>'}
      </ul>
    </div>
    <div class="card">
      <div class="lbl">Grafo armónico</div>
      <div class="metric"><span class="metric-key">Entropía</span>
        <span class="metric-val"><strong>{hg.get("entropy",0):.3f}</strong></span></div>
      <div class="metric"><span class="metric-key">Centro gravitacional</span>
        <span class="metric-val">{hg.get("center","—")}</span></div>
      <div class="metric"><span class="metric-key">Modulaciones</span>
        <span class="metric-val">{tg.get("n_modulations",0)}</span></div>
      <div class="metric"><span class="metric-key">Distancia tonal total</span>
        <span class="metric-val">{tg.get("total_distance",0):.1f}</span></div>
      <div class="metric"><span class="metric-key">Ambigüedad tonal</span>
        <span class="metric-val">{ta.get("ambiguity_index",0):.3f}</span></div>
      <div class="metric"><span class="metric-key">Cromatismo</span>
        <span class="metric-val">{ch.get("chromaticism_index",0):.3f}</span></div>
      <div class="metric"><span class="metric-key">Res. disonancia μ</span>
        <span class="metric-val">{dr.get("resolution_time_mean",0):.2f}s</span></div>
      {'<h3 style="margin-top:12px">Préstamo modal</h3><ul style="font-size:12px;padding-left:16px;margin-top:4px">' + ''.join(f"<li>{m}</li>" for m in mm[:4]) + '</ul>' if mm else ''}
    </div>
    <div class="card">
      <div class="lbl">Fingerprint armónico</div>
      {fp_rows}
      {'<h3 style="margin-top:12px">Estilo</h3><ul style="font-size:12px;padding-left:16px;margin-top:4px">' + ''.join(f"<li>{t}</li>" for t in (fp.get('style_tags') or [])[:4]) + '</ul>' if fp.get('style_tags') else ''}
    </div>
  </div>
</div>'''


def _s_rhythm_melody(r: dict) -> str:
    rh = r.get('rhythm') or {}
    mt = r.get('micro_timing') or {}
    gr = r.get('groove') or {}
    mm_m = r.get('melodic_markov') or {}
    ia   = r.get('interval_anal') or {}
    exp  = r.get('expectation') or {}
    mh   = r.get('metric_hierarchy') or {}

    mm_style = mm_m.get('style_profile', '')
    mm_cls   = ('badge-green' if mm_style == 'predecible' else
                'badge-amber' if mm_style == 'equilibrado' else
                'badge-red')

    return f'''
<div class="card-full">
  <h2>Ritmo, melodía y micro-timing</h2>
  <div class="grid3">
    <div class="card">
      <div class="lbl">Ritmo</div>
      <div class="metric"><span class="metric-key">Síncopa</span>
        <span class="metric-val">{rh.get("syncopation",0):.3f} {_pct_bar(rh.get("syncopation",0))}</span></div>
      <div class="metric"><span class="metric-key">Variedad rítmica</span>
        <span class="metric-val">{rh.get("variety",0):.3f}</span></div>
      <div class="metric"><span class="metric-key">Swing ratio</span>
        <span class="metric-val">{rh.get("swing_ratio",0):.3f}</span></div>
      <div class="metric"><span class="metric-key">Groove strength</span>
        <span class="metric-val">{gr.get("groove_strength",0):.3f}</span></div>
    </div>
    <div class="card">
      <div class="lbl">Micro-timing</div>
      <div class="metric"><span class="metric-key">Humanización</span>
        <span class="metric-val">{mt.get("humanization",0):.3f}</span></div>
      <div class="metric"><span class="metric-key">Desv. media</span>
        <span class="metric-val">{mt.get("mean_deviation_ms",0):.1f} ms</span></div>
      <div class="metric"><span class="metric-key">Drag ratio</span>
        <span class="metric-val">{mt.get("drag_ratio",0):.3f}</span></div>
      <div class="metric"><span class="metric-key">Anticipación</span>
        <span class="metric-val">{mt.get("anticipation_ratio",0):.3f}</span></div>
      <div class="metric"><span class="metric-key">Estilo</span>
        <span class="metric-val" style="max-width:150px;text-align:right;font-size:11px">{mt.get("style","—")[:40]}</span></div>
    </div>
    <div class="card">
      <div class="lbl">Markov melódico (2º orden)</div>
      <div class="metric"><span class="metric-key">Perfil</span>
        <span class="metric-val"><span class="badge {mm_cls}">{mm_style or "—"}</span></span></div>
      <div class="metric"><span class="metric-key">P(transición) μ</span>
        <span class="metric-val">{mm_m.get("mean_probability",0):.3f}</span></div>
      <div class="metric"><span class="metric-key">Entropía Markov</span>
        <span class="metric-val">{mm_m.get("entropy_markov",0):.3f} bits</span></div>
      <div class="metric"><span class="metric-key">Estados únicos</span>
        <span class="metric-val">{mm_m.get("n_unique_states",0)}</span></div>
      <div class="metric"><span class="metric-key">Hapax melódicos</span>
        <span class="metric-val">{len(mm_m.get("singular_gestures") or [])}</span></div>
      <h3 style="margin-top:12px">Intervalos (melodía)</h3>
      <div class="metric"><span class="metric-key">Mov. conjunto</span>
        <span class="metric-val">{ia.get("conjunct",0):.0%}</span></div>
      <div class="metric"><span class="metric-key">Saltos</span>
        <span class="metric-val">{ia.get("leaps",0):.0%}</span></div>
      <div class="metric"><span class="metric-key">Ascendente</span>
        <span class="metric-val">{ia.get("ascending",0):.0%}</span></div>
      <div class="metric"><span class="metric-key">Sorpresa μ (IDyOM)</span>
        <span class="metric-val">{exp.get("mean_surprise",0):.3f}</span></div>
    </div>
  </div>
</div>'''


def _s_narrative(r: dict) -> str:
    ni  = r.get('narrative_intention') or {}
    ac  = r.get('anti_conventional') or {}
    mld = r.get('multilevel_density') or {}
    amb = r.get('emotional_ambivalence') or {}
    co  = r.get('coherence') or {}

    ni_scores = ni.get('scores') or {}
    score_rows = ''.join(
        f'<tr><td>{k}</td><td>{_pct_bar(v, max(ni_scores.values(), default=1))}</td></tr>'
        for k, v in sorted(ni_scores.items(), key=lambda x: -x[1])
    )

    devs = ac.get('deviations') or []
    dev_rows = ''.join(
        f'<tr><td>{d["type"]}</td><td>{_pct_bar(d.get("weight",0))}</td></tr>'
        for d in devs[:5]
    ) or '<tr><td colspan="2" style="color:#94a3b8">escritura convencional</td></tr>'

    nc = mld.get('novelty_curve') or []
    nc_rows = ''.join(
        f'<tr><td>Sec.{item["section"]}</td>'
        f'<td>{_pct_bar(item.get("novelty",0))}</td>'
        f'<td style="text-align:right;font-size:11px">{item.get("entropy",0):.2f} bits</td></tr>'
        for item in nc
    )

    return f'''
<div class="card-full">
  <h2>Análisis narrativo e intencional</h2>
  <div class="grid3">
    <div class="card">
      <div class="lbl">Arquetipo narrativo</div>
      <p style="margin-bottom:10px">
        <strong>{ni.get("archetype","—")}</strong>
        &nbsp;<span class="badge">{ni.get("confidence",0):.0%}</span>
      </p>
      <div class="metric"><span class="metric-key">Alternativo</span>
        <span class="metric-val">{ni.get("alternative","—")}</span></div>
      <h3 style="margin-top:10px">Scores por arquetipo</h3>
      <table><tbody>{score_rows or "<tr><td colspan=2 style='color:#94a3b8'>sin datos</td></tr>"}</tbody></table>
    </div>
    <div class="card">
      <div class="lbl">Voz compositiva</div>
      <div class="metric"><span class="metric-key">Originalidad</span>
        <span class="metric-val">{ac.get("deviation_score",0):.2f} {_pct_bar(ac.get("deviation_score",0))}</span></div>
      <div class="metric"><span class="metric-key">Coherencia emocional</span>
        <span class="metric-val">{co.get("coherence",1):.3f}</span></div>
      <div class="metric"><span class="metric-key">Ambivalencia μ</span>
        <span class="metric-val">{amb.get("mean_ambivalence",0):.3f}</span></div>
      <h3 style="margin-top:10px">Desviaciones anti-conv. ({len(devs)})</h3>
      <table><tbody>{dev_rows}</tbody></table>
    </div>
    <div class="card">
      <div class="lbl">Densidad informacional multinivel</div>
      <div class="metric"><span class="metric-key">Micro (nota)</span>
        <span class="metric-val">{mld.get("micro_entropy",0):.3f} bits</span></div>
      <div class="metric"><span class="metric-key">Meso (frase)</span>
        <span class="metric-val">{mld.get("meso_entropy",0):.3f} bits</span></div>
      <div class="metric"><span class="metric-key">Macro (sección)</span>
        <span class="metric-val">{mld.get("macro_entropy",0):.3f} bits</span></div>
      <div class="metric"><span class="metric-key">Economía comp.</span>
        <span class="metric-val">{mld.get("composer_economy",0):.3f} {_pct_bar(mld.get("composer_economy",0))}</span></div>
      <div class="metric"><span class="metric-key">Perfil</span>
        <span class="metric-val">{mld.get("density_profile","—")}</span></div>
      {f'<h3 style="margin-top:10px">Novedad por sección</h3><table><thead><tr><th>Sec.</th><th>Novedad</th><th>Entropía</th></tr></thead><tbody>{nc_rows}</tbody></table>' if nc_rows else ''}
    </div>
  </div>
</div>'''




# ── Radar 6D + Heatmap 5×5 (ported from mscz2vec) ────────────────────────────

_FUNC_LABELS = ['T', 'PD', 'D', 'Dsec', 'Other']

def _chord_to_function(chord, key_root: int, mode: str) -> str:
    if chord is None:
        return 'Other'
    root  = getattr(chord, 'root', None)
    ctype = getattr(chord, 'chord_type', '') or ''
    if root is None:
        return 'Other'
    interval = (int(root) - int(key_root)) % 12
    is_minor = mode in ('minor', 'dorian', 'phrygian', 'locrian')
    if interval == 0:
        return 'T'
    if interval == 9 and not is_minor:
        return 'T'
    if interval == 3 and is_minor:
        return 'T'
    if interval == 7:
        return 'D'
    if interval == 11:
        return 'D'
    if 'dom7' in ctype and interval not in (0, 7):
        return 'Dsec'
    if interval in (2, 5):
        return 'PD'
    return 'Other'


def _build_transition_matrix(chords: list, key_root: int, mode: str) -> list:
    n   = len(_FUNC_LABELS)
    idx = {f: i for i, f in enumerate(_FUNC_LABELS)}
    matrix = [[0] * n for _ in range(n)]
    funcs  = [_chord_to_function(c, key_root, mode) for c in (chords or [])]
    for i in range(len(funcs) - 1):
        r2 = idx.get(funcs[i], 4)
        c2 = idx.get(funcs[i+1], 4)
        matrix[r2][c2] += 1
    for row in matrix:
        total = sum(row)
        if total > 0:
            for j in range(n):
                row[j] = round(row[j] / total, 4)
    return matrix


def _compute_6d_vector(r: dict) -> dict:
    dv    = r.get('dynamic_valence') or {}
    ep    = r.get('energy_profile') or {}
    tc    = r.get('tension_curve') or []
    ta    = r.get('tonal_ambiguity') or {}
    rh    = r.get('rhythm') or {}
    notes = r.get('notes') or []

    val_raw  = _safe(dv.get('mean_valence'), 0.0)
    valence  = (float(val_raw) + 1.0) / 2.0

    ep_curve = ep.get('curve') or []
    ep_vals  = []
    for p in ep_curve:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            ep_vals.append(float(p[1]))
        elif isinstance(p, dict):
            ep_vals.append(float(p.get('energy', p.get('value', 0))))
    ep_max     = max(ep_vals) if ep_vals else 1
    activation = min(1.0, (sum(ep_vals)/len(ep_vals)/ep_max) if ep_vals and ep_max > 0 else 0.5)

    t_vals = []
    for p in tc:
        if hasattr(p, 'tension'):
            t_vals.append(float(p.tension))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            t_vals.append(float(p[1]))
        elif isinstance(p, dict):
            t_vals.append(float(p.get('tension', 0)))
    harm_tension = sum(t_vals)/len(t_vals) if t_vals else 0.3

    amb_idx        = _safe(ta.get('ambiguity_index'), 0.5)
    tonal_stability = max(0.0, 1.0 - float(amb_idx))

    sync   = _safe(rh.get('syncopation'), 0.0)
    variety = _safe(rh.get('variety'), 0.0)
    rhythm_density = min(1.0, float(sync) * 0.6 + float(variety) * 0.4)

    if notes:
        pitches = [getattr(n, 'pitch', 60) for n in notes]
        avg_p   = sum(pitches) / len(pitches)
        brightness = max(0.0, min(1.0, (avg_p - 36) / 48.0))
    else:
        brightness = 0.5

    return {
        'valence':         round(valence, 3),
        'activation':      round(activation, 3),
        'harm_tension':    round(harm_tension, 3),
        'tonal_stability': round(tonal_stability, 3),
        'rhythm_density':  round(rhythm_density, 3),
        'brightness':      round(brightness, 3),
    }


def _s_radar_6d(r: dict) -> str:
    vec    = _compute_6d_vector(r)
    labels = ['Valence', 'Activation', 'Harm. tension',
              'Tonal stability', 'Rhythm density', 'Brightness']
    values = [vec['valence'], vec['activation'], vec['harm_tension'],
              vec['tonal_stability'], vec['rhythm_density'], vec['brightness']]

    script = f"""<script>
(function(){{
  var radar_ctx = document.getElementById('radar_6d');
  if (!radar_ctx) return;
  new Chart(radar_ctx.getContext('2d'), {{
    type: 'radar',
    data: {{
      labels: {json.dumps(labels)},
      datasets: [{{
        label: 'Percepción unificada',
        data: {json.dumps(values)},
        backgroundColor: 'rgba(99,102,241,0.15)',
        borderColor: '#6366f1',
        borderWidth: 2,
        pointBackgroundColor: '#6366f1',
        pointRadius: 4,
      }}]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        r: {{
          min: 0, max: 1,
          ticks: {{ stepSize: 0.25, font: {{ size: 10 }}, color: '#94a3b8', backdropColor: 'transparent' }},
          pointLabels: {{ font: {{ size: 11 }}, color: '#475569' }},
          grid: {{ color: '#f1f5f9' }},
          angleLines: {{ color: '#e2e8f0' }}
        }}
      }}
    }}
  }});
}})();
</script>"""

    dim_meta = [
        ('Valence',         'Valencia emocional media',             '#6366f1'),
        ('Activation',      'Energía / arousal medio',              '#10b981'),
        ('Harm. tension',   'Tensión armónica media',               '#ef4444'),
        ('Tonal stability', '1 − ambigüedad tonal',                 '#3b82f6'),
        ('Rhythm density',  'Síncopa + variedad rítmica',           '#f59e0b'),
        ('Brightness',      'Luminosidad tímbrica (registro midi)', '#a855f7'),
    ]
    dim_rows = ''
    for (lbl, desc, col), val in zip(dim_meta, values):
        bw = int(val * 80)
        dim_rows += (
            f'<tr><td><span style="display:inline-block;width:8px;height:8px;'
            f'border-radius:50%;background:{col};margin-right:6px"></span>{lbl}</td>'
            f'<td style="color:#64748b;font-size:11px">{desc}</td>'
            f'<td><div style="display:inline-flex;align-items:center;gap:4px">'
            f'<div style="background:#e5e7eb;border-radius:3px;height:6px;width:80px">'
            f'<div style="background:{col};border-radius:3px;height:6px;width:{bw}px"></div></div>'
            f'<span style="font-size:11px;font-weight:500;color:#0f172a">{val:.3f}</span>'
            f'</div></td></tr>'
        )

    return f'''
<div class="card-full">
  <h2>Vector de percepción unificada 6D
    <span style="font-size:12px;font-weight:400;color:#94a3b8">(adaptado de mscz2vec)</span>
  </h2>
  <div class="grid2">
    <div style="height:280px;position:relative">
      <canvas id="radar_6d"></canvas>
    </div>
    <div>
      <p style="font-size:12px;color:#64748b;margin-bottom:12px">
        Firma perceptual compacta de la obra — permite comparar identidad musical entre piezas.
        Cada dimensión normalizada [0, 1].
      </p>
      <table>
        <thead><tr><th>Dimensión</th><th>Descripción</th><th>Valor</th></tr></thead>
        <tbody>{dim_rows}</tbody>
      </table>
    </div>
  </div>
  {script}
</div>'''


def _s_harmonic_heatmap(r: dict) -> str:
    chords   = r.get('chords') or []
    key_root = int(_safe(r.get('key_root'), 0))
    mode     = r.get('mode', 'major')
    matrix   = _build_transition_matrix(chords, key_root, mode)
    funcs    = _FUNC_LABELS

    cell = 52
    pad  = 62
    size = pad + cell * 5 + 24

    def cell_color(v: float) -> str:
        v = min(1.0, max(0.0, v))
        r2 = int(255 - v * 172)
        g2 = int(255 - v * 181)
        b2 = int(255 - v * 72)
        return f'rgb({r2},{g2},{b2})'

    func_full = {
        'T':'T (tónica)', 'PD':'PD (predom.)',
        'D':'D (dom.)', 'Dsec':'Dsec (sec.)', 'Other':'Other'
    }

    cells_svg = ''
    for ci, f in enumerate(funcs):
        x = pad + ci * cell + cell // 2
        cells_svg += (
            f'<text x="{x}" y="{pad-10}" text-anchor="middle" '
            f'font-size="10" fill="#94a3b8" font-family="sans-serif">{f}</text>'
        )
    for ri, frow in enumerate(funcs):
        yc = pad + ri * cell + cell // 2
        cells_svg += (
            f'<text x="{pad-6}" y="{yc}" text-anchor="end" '
            f'dominant-baseline="central" font-size="10" fill="#475569" '
            f'font-family="sans-serif">{func_full[frow]}</text>'
        )
        for ci, _ in enumerate(funcs):
            v   = matrix[ri][ci]
            cx  = pad + ci * cell
            cy  = pad + ri * cell
            pct = f'{v*100:.0f}%' if v > 0.02 else ''
            tc2 = '#1e1b4b' if v > 0.4 else '#475569'
            cells_svg += (
                f'<rect x="{cx+1}" y="{cy+1}" width="{cell-2}" height="{cell-2}" '
                f'rx="4" fill="{cell_color(v)}" stroke="#f1f5f9" stroke-width="0.5"/>'
                f'<text x="{cx+cell//2}" y="{cy+cell//2}" text-anchor="middle" '
                f'dominant-baseline="central" font-size="11" fill="{tc2}" '
                f'font-weight="500" font-family="sans-serif">{pct}</text>'
            )

    heatmap_svg = (
        f'<svg width="100%" viewBox="0 0 {size} {size}" style="max-width:360px;display:block">'
        f'{cells_svg}</svg>'
    )

    # Patterns
    fi = {f: i for i, f in enumerate(funcs)}
    dt  = matrix[fi['D']][fi['T']]
    pdt = matrix[fi['PD']][fi['T']]
    tt  = matrix[fi['T']][fi['T']]
    dpd = matrix[fi['D']][fi['PD']]

    patterns = []
    if dt  > 0.4:  patterns.append(('Muy cadencial',  'D→T frecuente — resolución constante',         '#10b981'))
    if pdt > 0.3:  patterns.append(('Plagal',          'PD→T frecuente — cadencias modales/plagales',  '#3b82f6'))
    if tt  > 0.3:  patterns.append(('Estática',        'T→T — la tónica se prolonga sin moverse',      '#94a3b8'))
    if dpd > 0.25: patterns.append(('Evasiva',         'D→PD — dominante que evita resolver',          '#f59e0b'))
    if not patterns: patterns.append(('Equilibrada',   'Distribución uniforme de transiciones',         '#6366f1'))

    pat_html = ''.join(
        f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;'
        f'border-bottom:1px dashed #f1f5f9">'
        f'<span style="width:10px;height:10px;border-radius:50%;background:{col};'
        f'flex-shrink:0;display:inline-block"></span>'
        f'<div><div style="font-size:12px;font-weight:500;color:#0f172a">{lbl}</div>'
        f'<div style="font-size:11px;color:#64748b">{desc}</div></div></div>'
        for lbl, desc, col in patterns
    )

    # Top transitions
    pairs = sorted(
        [(matrix[ri][ci], funcs[ri], funcs[ci])
         for ri in range(5) for ci in range(5) if matrix[ri][ci] > 0],
        reverse=True
    )
    top_html = ''.join(
        f'<div class="metric">'
        f'<span class="metric-key" style="font-family:monospace">{fr} → {fc}</span>'
        f'<span class="metric-val">{_pct_bar(v)}</span></div>'
        for v, fr, fc in pairs[:5]
    )

    # Function distribution
    func_counts = {}
    for c in chords:
        f = _chord_to_function(c, key_root, mode)
        func_counts[f] = func_counts.get(f, 0) + 1
    total = max(len(chords), 1)
    dist_html = ''.join(
        f'<div class="metric"><span class="metric-key">{f}</span>'
        f'<span class="metric-val">{_pct_bar(func_counts.get(f,0), total)}</span></div>'
        for f in funcs
    )

    return f'''
<div class="card-full">
  <h2>Flujo de funciones armónicas — heatmap 5×5
    <span style="font-size:12px;font-weight:400;color:#94a3b8">(adaptado de mscz2vec)</span>
  </h2>
  <p style="font-size:12px;color:#64748b;margin-bottom:16px">
    Probabilidad de transición de cada función a la siguiente.
    Diagonal fuerte = estasis tonal. Columna T fuerte = música cadencial.
  </p>
  <div class="grid3">
    <div>
      <div class="lbl">Matriz de transición (normalizada por filas)</div>
      {heatmap_svg}
      <p style="font-size:10px;color:#94a3b8;margin-top:4px">
        T=Tónica · PD=Predominante · D=Dominante · Dsec=Dom. sec. · Other=Otro
      </p>
    </div>
    <div>
      <div class="lbl">Patrón detectado</div>
      {pat_html}
      <div style="margin-top:14px">
        <div class="lbl">Distribución ({len(chords)} acordes)</div>
        {dist_html}
      </div>
    </div>
    <div>
      <div class="lbl">Transiciones más frecuentes</div>
      {top_html}
    </div>
  </div>
</div>'''


# ── Main render ───────────────────────────────────────────────────────────────

def render_html(results: dict) -> str:
    """
    Genera el informe estadístico HTML interactivo.

    Args:
        results: dict devuelto por run_analysis()

    Returns:
        str: HTML autocontenido con Chart.js
    """
    import re

    r     = results
    title = r.get('title', 'sin título')
    dur   = _safe(r.get('total_dur'))
    dur_s = f"{int(dur//60)}m{int(dur%60):02d}s"
    kn    = r.get('key_name', f"{r.get('key_root','')} {r.get('mode','')}")

    body = (
        _s_identity(r) +
        _s_curves(r) +
        _s_radar_6d(r) +
        _s_emotional_map(r) +
        _s_ssm(r) +
        _s_harmony(r) +
        _s_harmonic_heatmap(r) +
        _s_rhythm_melody(r) +
        _s_narrative(r)
    )

    # ── Extract all inline <script>…</script> blocks from body ──
    # and collect them into a single deferred block that runs after
    # Chart.js is guaranteed to be loaded.
    script_contents = re.findall(r'<script>(.*?)</script>', body, re.DOTALL)
    body_clean = re.sub(r'<script>.*?</script>', '', body, flags=re.DOTALL)

    # Wrap all chart-init code in a single window.onload block
    combined_js = '\n'.join(script_contents)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Análisis MIDI — {title}</title>
<style>{_CSS}</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">{kn} · {r.get("avg_bpm",0):.0f} BPM · {dur_s} · midi_analyzer v12.0</p>
{body_clean}
<!-- Chart.js must load before chart initialization scripts -->
<script src="{_CHART_JS_CDN}"></script>
<script>
/* All chart initializations — runs after Chart.js is loaded */
window.addEventListener('load', function() {{
{combined_js}
}});
</script>
</body>
</html>"""
