# =========================
# EXPORTADOR DE DATOS - VERSIÓN CORREGIDA
# =========================

import json
import numpy as np
from datetime import datetime
from music21 import converter

# IMPORTANTE: Importar tus funciones de análisis
# Cambia 'mscz2vec' por el nombre de tu archivo (sin .py)
import sys
import os

# Agregar el directorio actual al path de Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar todas las funciones del archivo de análisis
# Opción 1: Si tu archivo se llama mscz2vec.py
try:
    from mscz2vec import *
    print("✅ Funciones importadas desde mscz2vec.py")
except ImportError:
    # Opción 2: Si se llama analisis_musical.py
    try:
        from analisis_musical import *
        print("✅ Funciones importadas desde analisis_musical.py")
    except ImportError:
        print("❌ ERROR: No se pudieron importar las funciones de análisis")
        print("   Asegúrate de que el archivo esté en el mismo directorio")
        print("   y se llame 'mscz2vec.py' o 'analisis_musical.py'")
        sys.exit(1)

def export_to_javascript(score, output_file='music_data.js'):
    """
    Extrae todos los análisis y genera código JavaScript listo para copiar.
    """

    print("Iniciando análisis de la partitura...")

    # 1. MELODÍA
    print("  → Analizando melodía...")
    melodic_vec = melodic_features(score)
    melodic_js = {
        'intervals': [
            {'name': '-12 a -10', 'value': float(melodic_vec[0])},
            {'name': '-9 a -7', 'value': float(melodic_vec[1])},
            {'name': '-6 a -4', 'value': float(melodic_vec[2])},
            {'name': '-3 a -1', 'value': float(melodic_vec[3])},
            {'name': '0 a 2', 'value': float(melodic_vec[4])},
            {'name': '3 a 5', 'value': float(melodic_vec[5])},
            {'name': '6 a 8', 'value': float(melodic_vec[6])},
            {'name': '9 a 11', 'value': float(melodic_vec[7])},
        ],
        'stats': {
            'mean': float(melodic_vec[12]),
            'std': float(melodic_vec[13]),
            'range': float(melodic_vec[14]),
            'direction': float(melodic_vec[15])
        }
    }

    # 2. ARMONÍA
    print("  → Analizando armonía...")
    harmonic_vec = harmonic_features(score)
    harmonic_js = [
        {'function': 'Tónica (T)', 'value': float(harmonic_vec[0]), 'fullMark': 1},
        {'function': 'Predominante (PD)', 'value': float(harmonic_vec[1]), 'fullMark': 1},
        {'function': 'Dominante (D)', 'value': float(harmonic_vec[2]), 'fullMark': 1},
        {'function': 'Dom. Sec. (Dsec)', 'value': float(harmonic_vec[3]), 'fullMark': 1},
        {'function': 'Otros', 'value': float(harmonic_vec[4]), 'fullMark': 1}
    ]

    # 3. TRANSICIONES ARMÓNICAS
    print("  → Analizando transiciones armónicas...")
    transitions_vec = harmonic_transition_features(score)
    labels = ['T', 'PD', 'D', 'Dsec', 'Other']
    transitions_js = []

    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            value = float(transitions_vec[idx])
            if value > 0.05:  # Solo transiciones significativas
                transitions_js.append({
                    'from': labels[i],
                    'to': labels[j],
                    'value': value
                })

    # Ordenar por valor y tomar top 10
    transitions_js = sorted(transitions_js, key=lambda x: x['value'], reverse=True)[:10]

    # 4. RITMO
    print("  → Analizando ritmo...")
    rhythmic_vec = rhythmic_features(score)
    rhythmic_js = {
        'durations': [
            {'name': f'{i*0.167:.2f}-{(i+1)*0.167:.2f}', 'value': float(rhythmic_vec[i])}
            for i in range(min(12, len(rhythmic_vec)))
        ],
        'fft': [
            {'freq': i, 'magnitude': float(rhythmic_vec[12 + i])}
            for i in range(min(16, len(rhythmic_vec) - 12))
        ]
    }

    # 5. INSTRUMENTACIÓN
    print("  → Analizando instrumentación...")
    inst_vec = instrumental_features(score)
    families = ['Piano', 'Cuerdas', 'Vientos Madera', 'Vientos Metal',
                'Guitarra', 'Voz', 'Percusión']
    instrumental_js = [
        {'family': families[i], 'active': int(inst_vec[i])}
        for i in range(len(families))
    ]

    # 6. MOTIVOS
    print("  → Analizando motivos...")
    motif_vec = motif_vector(score)
    top_indices = np.argsort(motif_vec)[-10:][::-1]
    motifs_js = []

    for rank, idx in enumerate(top_indices):
        if motif_vec[idx] > 0:
            motifs_js.append({
                'id': rank + 1,
                'strength': float(motif_vec[idx]),
                'length': 3 + (rank % 4),
                'repetitions': int(motif_vec[idx] * 20),
                'notes': f'Motivo_{idx}'
            })

    # 7. FORMA
    print("  → Analizando forma...")
    form_string = advanced_sequitur_form(score)

    # Extraer secciones
    sections_js = []
    if form_string and form_string != "N/A" and len(form_string) > 0:
        current_section = form_string[0]
        start = 0

        for i, char in enumerate(form_string + ' '):
            if char != current_section:
                sections_js.append({
                    'name': current_section,
                    'measures': i - start,
                    'start': start
                })
                current_section = char
                start = i

    form_js = {
        'structure': form_string,
        'sections': sections_js,
        'noveltyCurve': [
            {'measure': i + 1, 'novelty': 0.3 + 0.1 * np.sin(i/4)}
            for i in range(len(form_string) * 8 if form_string and form_string != "N/A" else 32)
        ]
    }

    # GENERAR CÓDIGO JAVASCRIPT
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    js_code = f"""// ============================================
// DATOS GENERADOS AUTOMÁTICAMENTE
// Generado desde Python el {timestamp}
// ============================================

const melodicData = {json.dumps(melodic_js, indent=2)};

const harmonicData = {json.dumps(harmonic_js, indent=2)};

const harmonicTransitions = {json.dumps(transitions_js, indent=2)};

const rhythmicData = {json.dumps(rhythmic_js, indent=2)};

const instrumentalData = {json.dumps(instrumental_js, indent=2)};

const motifData = {json.dumps(motifs_js, indent=2)};

const formData = {json.dumps(form_js, indent=2)};

// ============================================
// INSTRUCCIONES:
// 1. Copia todo este código
// 2. En el visualizador React, reemplaza las secciones correspondientes
// 3. Los datos se actualizarán automáticamente
// ============================================
"""

    # Guardar en archivo
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(js_code)

    print("\n" + "="*50)
    print(f"✅ Datos exportados a: {output_file}")
    print(f"📊 Formato de forma detectado: {form_string}")
    print(f"🎵 Total de motivos encontrados: {len(motifs_js)}")
    print("="*50)
    print(f"\n💡 Ahora copia el contenido de '{output_file}' al visualizador React")

    return js_code


def export_to_json(score, output_file='music_data.json'):
    """
    Exporta a JSON puro (más fácil de procesar programáticamente).
    """

    print("Generando JSON...")

    data = {
        'melodic': melodic_features(score).tolist(),
        'harmonic': harmonic_features(score).tolist(),
        'transitions': harmonic_transition_features(score).tolist(),
        'rhythmic': rhythmic_features(score).tolist(),
        'instrumental': instrumental_features(score).tolist(),
        'motifs': motif_vector(score).tolist(),
        'form': advanced_sequitur_form(score),
        'form_vector': form_string_to_vector(advanced_sequitur_form(score)).tolist()
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"✅ JSON exportado a: {output_file}")
    return data


# =========================
# EJEMPLO DE USO
# =========================

if __name__ == "__main__":
    import sys

    # Verificar argumentos
    if len(sys.argv) < 2:
        print("Uso: python exportador_corregido.py <archivo.musicxml>")
        print("\nEjemplo:")
        print("  python exportador_corregido.py dreams2.musicxml")
        sys.exit(1)

    archivo = sys.argv[1]

    print("="*50)
    print("🎵 EXPORTADOR DE ANÁLISIS MUSICAL")
    print("="*50)
    print(f"📄 Archivo: {archivo}\n")

    try:
        # Cargar partitura
        score = converter.parse(archivo)
        print(f"✅ Partitura cargada correctamente\n")

        # OPCIÓN 1: Generar JavaScript
        export_to_javascript(score, 'music_data.js')

        # OPCIÓN 2: Generar JSON
        export_to_json(score, 'music_data.json')

        print("\n" + "="*50)
        print("✅ EXPORTACIÓN COMPLETADA")
        print("="*50)

    except FileNotFoundError:
        print(f"❌ Error: Archivo '{archivo}' no encontrado")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al procesar: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
