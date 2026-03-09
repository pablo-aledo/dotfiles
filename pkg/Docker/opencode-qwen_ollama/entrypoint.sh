#!/bin/bash
# entrypoint.sh
# Escribe la configuración de opencode para usar Ollama local.
# Se ejecuta en cada arranque, por lo que siempre tiene la URL correcta.

CONFIG_DIR="/home/opencode/.config/opencode"
CONFIG_FILE="${CONFIG_DIR}/opencode.json"

# Validación básica
if [ -z "${OPENCODE_API_URL}" ]; then
  echo "[entrypoint] ERROR: OPENCODE_API_URL no definida" >&2
  exit 1
fi

mkdir -p "${CONFIG_DIR}"

# Ollama expone una API compatible con OpenAI en /v1
# Se usa @ai-sdk/openai-compatible igual que con vLLM
cat > "${CONFIG_FILE}" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local Ollama",
      "options": {
        "baseURL": "${OPENCODE_API_URL:-http://ollama:11434/v1}",
        "apiKey": "${OPENCODE_API_KEY:-ollama}"
      },
      "models": {
        "${OPENCODE_MODEL:-qwen3:4b}": {
          "name": "${OPENCODE_MODEL:-qwen3:4b}",
          "limit": {
            "context": 8192,
            "output": 2048
          }
        }
      }
    }
  },
  "model": "ollama/${OPENCODE_MODEL:-qwen3:4b}"
}
EOF

echo "[entrypoint] Config generada en ${CONFIG_FILE}"
echo "[entrypoint] Modelo  : ${OPENCODE_MODEL}"
echo "[entrypoint] API URL : ${OPENCODE_API_URL}"

exec "$@"
