#!/bin/bash
# entrypoint.sh
# Escribe la configuración de opencode para usar el servidor vLLM local.
# Se ejecuta en cada arranque, por lo que siempre tiene la URL correcta.

CONFIG_DIR="/home/opencode/.config/opencode"
CONFIG_FILE="${CONFIG_DIR}/opencode.json"

mkdir -p "${CONFIG_DIR}"

# opencode usa un fichero config.json con proveedores personalizados.
# Ajusta "baseURL" si cambias el puerto o el nombre del servicio en compose.
cat > "${CONFIG_FILE}" <<EOF
{
  "\$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local vLLM",
      "options": {
        "baseURL": "${OPENCODE_API_URL:-http://vllm:8000/v1}",
        "apiKey": "${OPENCODE_API_KEY:-local}"
      },
      "models": {
        "${OPENCODE_MODEL:-qwen3-coder}": {
          "name": "${OPENCODE_MODEL:-qwen3-coder}",
          "limit": {
            "context": 4784,
            "output": 2048
          }
        }
      }
    }
  },
  "model": "vllm/${OPENCODE_MODEL:-qwen3-coder}"
}
EOF

echo "[entrypoint] Config generada en ${CONFIG_FILE}"
echo "[entrypoint] Modelo  : ${OPENCODE_MODEL}"
echo "[entrypoint] API URL : ${OPENCODE_API_URL}"

exec "$@"
