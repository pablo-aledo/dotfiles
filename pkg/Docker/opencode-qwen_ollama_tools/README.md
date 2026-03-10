# Qwen3 + opencode — Stack Docker local con Ollama

## Arquitectura

```
┌─────────────────────────────────────┐
│           docker-compose            │
│                                     │
│  ┌─────────────┐  ┌──────────────┐  │
│  │   Ollama    │  │  opencode    │  │
│  │ :11434/v1   │◄─│  (cliente)   │  │
│  │  Qwen3-4B   │  │              │  │
│  └─────────────┘  └──────────────┘  │
│        │                            │
│  Modelos guardados (volumen Docker) │
└─────────────────────────────────────┘
```

## Requisitos

| Requisito | Mínimo |
|-----------|--------|
| GPU NVIDIA | 4 GB VRAM (Qwen3-4B ~3 GB) |
| RAM sistema | 16 GB |
| Docker + NVIDIA Container Toolkit | instalado |
| Espacio disco | ~3 GB (pesos del modelo) |

> **Sin GPU:** elimina las secciones `runtime: nvidia` y `deploy.resources` del
> `docker-compose.yml`. Ollama detecta automáticamente si hay GPU disponible.

## Instalación rápida

```bash
# 1. Coloca los 3 ficheros en un directorio
mkdir qwen3-opencode && cd qwen3-opencode
# (copia aquí docker-compose.yml, Dockerfile.opencode, entrypoint.sh)

# 2. Levanta Ollama — descarga el modelo automáticamente (~3 GB, solo la primera vez)
docker compose up -d ollama ollama-pull

# 3. Sigue los logs hasta ver "[pull] Modelo listo"
docker compose logs -f ollama-pull

# 4. Arranca opencode en modo interactivo
docker compose run --rm opencode opencode
```

## Uso diario

```bash
# Levantar Ollama en segundo plano
docker compose up -d ollama

# Iniciar sesión de opencode
docker compose run --rm opencode opencode

# Ver logs de Ollama
docker compose logs -f ollama

# Parar todo
docker compose down
```

## Verificar que Ollama responde

```bash
curl http://localhost:11434/api/tags
# Debe devolver la lista de modelos descargados

# Compatibilidad OpenAI (para Antigravity)
curl http://localhost:11434/v1/models
```

## Conectar Antigravity al modelo local

En Antigravity: **Settings → Models → Add Custom Provider**

```
Base URL:  http://localhost:11434/v1
API Key:   ollama
Model ID:  qwen3:4b
```

## Cambiar de modelo

Edita `docker-compose.yml` y cambia la variable `OLLAMA_MODEL`:

```yaml
# Modelos disponibles de Qwen3 en Ollama
OLLAMA_MODEL=qwen3:0.6b   # Muy rápido, CPU ok, ~500 MB
OLLAMA_MODEL=qwen3:1.7b   # Ligero, ~1.5 GB VRAM
OLLAMA_MODEL=qwen3:4b     # Recomendado, ~3 GB VRAM  ← por defecto
OLLAMA_MODEL=qwen3:8b     # Mejor calidad, ~6 GB VRAM
```

## Parámetros configurables

| Variable | Descripción |
|----------|-------------|
| `OLLAMA_KEEP_ALIVE` | Tiempo que el modelo permanece en VRAM (defecto: 24h) |
| `OLLAMA_NUM_PARALLEL` | Peticiones simultáneas (defecto: 2) |
| `OLLAMA_MAX_LOADED_MODELS` | Modelos en memoria a la vez (defecto: 1) |

## Diferencias respecto a vLLM

| | Ollama | vLLM |
|---|---|---|
| Instalación | Muy sencilla | Compleja |
| Rendimiento GPU | Alto | Muy alto |
| Gestión de modelos | Automática (`pull`) | Manual |
| Compatibilidad OpenAI | ✅ `/v1` | ✅ `/v1` |
| Ideal para | Uso local / desarrollo | Producción |
