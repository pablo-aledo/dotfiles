# 🎵 Sistema de Análisis Musical - Guía de Instalación

## 📋 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## 🚀 Instalación Paso a Paso

### 1. Crear la estructura de carpetas

```bash
mi_proyecto_musical/
├── servidor_musica.py          # Servidor Flask
├── analisis_musical.py         # Tus funciones de análisis
├── templates/
│   └── index.html             # Frontend
├── partituras/                # Tus archivos .musicxml aquí
├── uploads/                   # Carpeta temporal para uploads
└── requirements.txt           # Dependencias
```

### 2. Instalar dependencias

Crea un archivo `requirements.txt`:

```txt
flask==3.0.0
flask-cors==4.0.0
music21==9.1.0
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.3
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

### 3. Organizar los archivos

#### a) `analisis_musical.py`

Copia TODAS tus funciones de análisis en este archivo (el código Python que ya tienes).
Asegúrate de que incluye:
- `melodic_features()`
- `harmonic_features()`
- `harmonic_transition_features()`
- `rhythmic_features()`
- `instrumental_features()`
- `motif_vector()`
- `advanced_sequitur_form()`
- Y todas las funciones auxiliares

#### b) `servidor_musica.py`

Al inicio del archivo, agrega esta línea para importar tus funciones:

```python
from analisis_musical import *
```

#### c) `templates/index.html`

Copia el archivo HTML completo en esta carpeta.

### 4. Agregar tus partituras

Coloca tus archivos `.musicxml`, `.xml`, `.mxl`, `.mid` en la carpeta `partituras/`:

```bash
partituras/
├── dreams2.musicxml
├── mi_composicion.musicxml
└── bach_coral.xml
```

### 5. Ejecutar el servidor

```bash
python servidor_musica.py
```

Verás algo como:

```
==================================================
🎵 SERVIDOR DE ANÁLISIS MUSICAL
==================================================
📂 Coloca tus partituras en la carpeta 'partituras/'
🌐 Abre tu navegador en: http://localhost:5000
==================================================
 * Running on http://0.0.0.0:5000
```

### 6. Usar la aplicación

1. Abre tu navegador en `http://localhost:5000`
2. Selecciona una partitura del menú desplegable o sube una nueva
3. Haz clic en "Analizar Partitura"
4. Navega por las diferentes pestañas para ver los resultados

## 🔧 Solución de Problemas

### Error: "No module named 'flask'"

```bash
pip install flask flask-cors
```

### Error: "No module named 'music21'"

```bash
pip install music21
```

### El servidor no encuentra las funciones de análisis

Verifica que `analisis_musical.py` esté en la misma carpeta que `servidor_musica.py` y que la primera línea de `servidor_musica.py` sea:

```python
from analisis_musical import *
```

### No aparecen las partituras en el selector

Asegúrate de que:
1. La carpeta `partituras/` existe
2. Los archivos tienen extensión `.musicxml`, `.xml`, `.mxl`, `.mid` o `.midi`
3. El servidor se reinició después de agregar nuevos archivos

## 📊 Características

✅ **Análisis Melódico**: Intervalos, dirección, rango  
✅ **Análisis Armónico**: Funciones armónicas y transiciones  
✅ **Análisis Rítmico**: Duraciones y análisis espectral  
✅ **Instrumentación**: Familias instrumentales activas  
✅ **Motivos**: Detección de patrones recurrentes  
✅ **Forma**: Estructura musical (AABA, etc.) y curva de novedad

## 🌐 Acceso desde otros dispositivos

Para acceder desde otros dispositivos en tu red local:

1. Encuentra tu IP local:
   ```bash
   # En Windows
   ipconfig
   
   # En Mac/Linux
   ifconfig
   ```

2. Abre en el navegador de otro dispositivo:
   ```
   http://TU_IP_LOCAL:5000
   ```
   
   Ejemplo: `http://192.168.1.100:5000`

## 🔒 Producción (opcional)

Para uso en producción, usa un servidor WSGI como Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 servidor_musica:app
```

## 📝 Personalización

### Cambiar el puerto

En `servidor_musica.py`, cambia la última línea:

```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Usa el puerto 8080
```

### Agregar más análisis

1. Agrega tu función en `analisis_musical.py`
2. Llama la función en `analyze_score()` en `servidor_musica.py`
3. Agrega la visualización correspondiente en `index.html`

## 💡 Consejos

- **Rendimiento**: Para archivos grandes, el análisis puede tomar varios segundos
- **Cache**: Considera implementar cache para evitar reanálisis de archivos ya procesados
- **Logs**: El servidor imprime mensajes útiles en la consola
- **Debug**: Usa `DEBUG = True` en `analisis_musical.py` para ver información detallada

## 🆘 ¿Necesitas ayuda?

Si tienes problemas:
1. Verifica que todas las dependencias estén instaladas
2. Revisa la consola del servidor para errores
3. Asegúrate de que los archivos MusicXML sean válidos
4. Prueba con una partitura simple primero

---

¡Disfruta analizando música! 🎼
