#!/bin/bash

# =========================
# SCRIPT DE INSTALACIÓN AUTOMÁTICA
# Sistema de Análisis Musical
# =========================

echo "========================================"
echo "🎵 INSTALADOR - SISTEMA DE ANÁLISIS MUSICAL"
echo "========================================"
echo ""

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar Python
echo -e "${BLUE}[1/6]${NC} Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}⚠️  Python3 no encontrado. Por favor instálalo primero.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python $(python3 --version) encontrado"
echo ""

# Crear estructura de carpetas
echo -e "${BLUE}[2/6]${NC} Creando estructura de carpetas..."
mkdir -p templates
mkdir -p partituras
mkdir -p uploads
echo -e "${GREEN}✓${NC} Carpetas creadas"
echo ""

# Crear requirements.txt
echo -e "${BLUE}[3/6]${NC} Creando archivo de dependencias..."
cat > requirements.txt << EOF
flask==3.0.0
flask-cors==4.0.0
music21==9.1.0
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.3
EOF
echo -e "${GREEN}✓${NC} requirements.txt creado"
echo ""

# Instalar dependencias
echo -e "${BLUE}[4/6]${NC} Instalando dependencias (esto puede tardar unos minutos)..."
pip3 install -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Dependencias instaladas correctamente"
else
    echo -e "${YELLOW}⚠️  Hubo problemas instalando algunas dependencias${NC}"
fi
echo ""

# Crear archivo de ejemplo
echo -e "${BLUE}[5/6]${NC} Creando archivo de configuración de ejemplo..."
cat > config_ejemplo.py << EOF
# Configuración del servidor

DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Carpetas
PARTITURAS_DIR = 'partituras'
UPLOADS_DIR = 'uploads'
TEMPLATES_DIR = 'templates'

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'.musicxml', '.xml', '.mxl', '.mid', '.midi'}
EOF
echo -e "${GREEN}✓${NC} Archivo de configuración creado"
echo ""

# Instrucciones finales
echo -e "${BLUE}[6/6]${NC} Configuración final..."
echo ""
echo "========================================"
echo -e "${GREEN}✅ INSTALACIÓN COMPLETADA${NC}"
echo "========================================"
echo ""
echo "📋 SIGUIENTES PASOS:"
echo ""
echo "1. Copia tu archivo de análisis musical:"
echo "   ${BLUE}cp tu_script_analisis.py analisis_musical.py${NC}"
echo ""
echo "2. Copia el servidor Flask:"
echo "   ${BLUE}# Usa el archivo servidor_musica.py proporcionado${NC}"
echo ""
echo "3. Copia el archivo HTML a templates/:"
echo "   ${BLUE}cp index.html templates/${NC}"
echo ""
echo "4. Coloca tus partituras en la carpeta partituras/:"
echo "   ${BLUE}cp *.musicxml partituras/${NC}"
echo ""
echo "5. Ejecuta el servidor:"
echo "   ${BLUE}python3 servidor_musica.py${NC}"
echo ""
echo "6. Abre tu navegador en:"
echo "   ${GREEN}http://localhost:5000${NC}"
echo ""
echo "========================================"
echo "🎼 ¡Disfruta analizando música!"
echo "========================================"
