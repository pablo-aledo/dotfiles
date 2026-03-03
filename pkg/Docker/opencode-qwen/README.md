# Qwen3-Coder + opencode — Stack Docker local

## Arquitectura

```
┌─────────────────────────────────────┐
│           docker-compose            │
│                                     │
│  ┌─────────────┐  ┌──────────────┐  │
│  │    vLLM     │  │  opencode    │  │
│  │ :8000/v1    │◄─│  (cliente)   │  │
│  │ Qwen3-30B   │  │              │  │
│  └─────────────┘  └──────────────┘  │
│        │                            │
│  HuggingFace cache (volumen Docker) │
└─────────────────────────────────────┘
```

## Requisitos

| Requisito | Mínimo |
|-----------|--------|
| GPU NVIDIA | 24 GB VRAM (modelo AWQ ~15 GB) |
| RAM sistema | 32 GB |
| Docker + NVIDIA Container Toolkit | instalado |
| Espacio disco | ~20 GB (pesos del modelo) |

> **Sin GPU:** elimina las secciones `runtime: nvidia` y `deploy.resources` del
> `docker-compose.yml` y añade `--device cpu` al comando de vLLM. La inferencia
> será muy lenta pero funcional.

## Instalación rápida

```bash
# 1. Clona / copia los ficheros en un directorio
mkdir qwen3-opencode && cd qwen3-opencode
# (copia aquí docker-compose.yml, Dockerfile.opencode, entrypoint.sh)

# 2. (Opcional) Token de HuggingFace si el modelo es privado/gated
export HF_TOKEN=hf_xxxxxxxxxxxx

# 3. Levanta el stack — la primera vez descarga el modelo (~15 GB)
docker compose up -d vllm

# 4. Espera a que vLLM esté listo (puede tardar 2-5 min la primera vez)
docker compose logs -f vllm   # espera a ver "Application startup complete"

# 5. Arranca opencode en modo interactivo
docker compose run --rm opencode opencode
```

## Uso diario

```bash
# Levantar vLLM en segundo plano
docker compose up -d vllm

# Iniciar sesión de opencode
docker compose run --rm opencode opencode

# Ver logs de vLLM
docker compose logs -f vllm

# Parar todo
docker compose down
```

## Verificar que vLLM responde

```bash
curl http://localhost:8000/v1/models
# Debe devolver: {"data":[{"id":"qwen3-coder",...}]}
```

## Parámetros configurables

Edita `docker-compose.yml` para ajustar:

| Parámetro vLLM | Descripción |
|----------------|-------------|
| `--max-model-len` | Contexto máximo (tokens). Reduce si tienes poca VRAM. |
| `--gpu-memory-utilization` | Fracción de VRAM usada (0.0-1.0) |
| `--tensor-parallel-size N` | Usa N GPUs en paralelo |

## Sin GPU (CPU only)

Elimina en `docker-compose.yml`:
- `runtime: nvidia`
- `deploy.resources`

Y cambia el command de vLLM a:
```yaml
command: >
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ
  --quantization awq
  --device cpu
  --dtype float32
  --max-model-len 4096
  ...
```
