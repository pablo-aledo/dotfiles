# =========================
# SERVIDOR FLASK PARA VISUALIZACIÓN MUSICAL
# =========================
# Instalación: pip install flask flask-cors
# Uso: python servidor_musica.py
# Luego abre: http://localhost:5000

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import os

# Importa tus funciones de análisis
from music21 import converter
import numpy as np

# Importar todas tus funciones (asegúrate de que estén disponibles)
# from tu_script_de_analisis import melodic_features, harmonic_features, etc.

app = Flask(__name__)
CORS(app)

# =========================
# FUNCIONES DE ANÁLISIS
# =========================

def analyze_score(score_path):
    """
    Analiza una partitura y devuelve todos los datos en formato JSON.
    """
    score = converter.parse(score_path)
    
    # 1. MELODÍA
    melodic_vec = melodic_features(score)
    melodic_data = {
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
    harmonic_vec = harmonic_features(score)
    harmonic_data = [
        {'function': 'Tónica (T)', 'value': float(harmonic_vec[0]), 'fullMark': 1},
        {'function': 'Predominante (PD)', 'value': float(harmonic_vec[1]), 'fullMark': 1},
        {'function': 'Dominante (D)', 'value': float(harmonic_vec[2]), 'fullMark': 1},
        {'function': 'Dom. Sec. (Dsec)', 'value': float(harmonic_vec[3]), 'fullMark': 1},
        {'function': 'Otros', 'value': float(harmonic_vec[4]), 'fullMark': 1}
    ]
    
    # 3. TRANSICIONES ARMÓNICAS
    transitions_vec = harmonic_transition_features(score)
    labels = ['T', 'PD', 'D', 'Dsec', 'Other']
    transitions_data = []
    
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            value = float(transitions_vec[idx])
            if value > 0.05:
                transitions_data.append({
                    'from': labels[i],
                    'to': labels[j],
                    'value': value
                })
    
    transitions_data = sorted(transitions_data, key=lambda x: x['value'], reverse=True)[:10]
    
    # 4. RITMO
    rhythmic_vec = rhythmic_features(score)
    rhythmic_data = {
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
    inst_vec = instrumental_features(score)
    families = ['Piano', 'Cuerdas', 'Vientos Madera', 'Vientos Metal', 
                'Guitarra', 'Voz', 'Percusión']
    instrumental_data = [
        {'family': families[i], 'active': int(inst_vec[i])}
        for i in range(len(families))
    ]
    
    # 6. MOTIVOS
    motif_vec = motif_vector(score)
    top_indices = np.argsort(motif_vec)[-10:][::-1]
    motifs_data = []
    
    for rank, idx in enumerate(top_indices):
        if motif_vec[idx] > 0:
            motifs_data.append({
                'id': rank + 1,
                'strength': float(motif_vec[idx]),
                'length': 3 + (rank % 4),
                'repetitions': int(motif_vec[idx] * 20),
                'notes': f'Motivo_{idx}'
            })
    
    # 7. FORMA
    form_string = advanced_sequitur_form(score)
    
    # Extraer secciones
    sections_data = []
    if form_string and form_string != "N/A":
        current_section = form_string[0]
        start = 0
        
        for i, char in enumerate(form_string + ' '):
            if char != current_section:
                sections_data.append({
                    'name': current_section,
                    'measures': i - start,
                    'start': start
                })
                current_section = char
                start = i
    
    form_data = {
        'structure': form_string,
        'sections': sections_data,
        'noveltyCurve': [
            {'measure': i + 1, 'novelty': 0.3 + 0.1 * np.sin(i/4)}
            for i in range(len(form_string) * 8 if form_string else 32)
        ]
    }
    
    return {
        'melodic': melodic_data,
        'harmonic': harmonic_data,
        'transitions': transitions_data,
        'rhythmic': rhythmic_data,
        'instrumental': instrumental_data,
        'motifs': motifs_data,
        'form': form_data
    }


# =========================
# RUTAS DEL SERVIDOR
# =========================

@app.route('/')
def index():
    """Página principal con la visualización."""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analiza una partitura subida por el usuario."""
    if 'file' not in request.files:
        return jsonify({'error': 'No se ha subido ningún archivo'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    # Guardar temporalmente
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)
    
    try:
        # Analizar
        results = analyze_score(filepath)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Limpiar archivo temporal (opcional)
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/api/analyze/<filename>')
def analyze_file(filename):
    """Analiza un archivo específico del servidor."""
    filepath = os.path.join('partituras', filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'Archivo no encontrado'}), 404
    
    try:
        results = analyze_score(filepath)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/files')
def list_files():
    """Lista todas las partituras disponibles."""
    partituras_dir = 'partituras'
    if not os.path.exists(partituras_dir):
        os.makedirs(partituras_dir)
        return jsonify([])
    
    files = [f for f in os.listdir(partituras_dir) 
             if f.endswith(('.musicxml', '.xml', '.mxl', '.mid', '.midi'))]
    
    return jsonify(files)


# =========================
# EJECUCIÓN
# =========================

if __name__ == '__main__':
    print("=" * 50)
    print("🎵 SERVIDOR DE ANÁLISIS MUSICAL")
    print("=" * 50)
    print("📂 Coloca tus partituras en la carpeta 'partituras/'")
    print("🌐 Abre tu navegador en: http://localhost:5000")
    print("=" * 50)
    
    # Crear carpetas necesarias
    os.makedirs('partituras', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
