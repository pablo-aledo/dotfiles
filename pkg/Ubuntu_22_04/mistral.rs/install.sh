MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Coder-7B-Instruct}"
MR_PORT="${MR_PORT:-8080}"
MR_HOST="${MR_HOST:-127.0.0.1}"
MR_SERVICE_NAME="mistralrs"
PI_MODELS_FILE="${HOME}/.pi/agent/models.json"

sudo apt-get update -y
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    git \
    curl \
    ca-certificates \
    unzip \
    jq

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh

mistralrs tune plain -m "${MODEL_ID}" --profile balanced

jq -n \
--arg base_url "http://${MR_HOST}:${MR_PORT}/v1" \
--arg model_id "${MODEL_ID}" \
'{
  providers: {
    mistralrs: {
      baseUrl: $base_url,
      api: "openai-completions",
      apiKey: "local",
      compat: {
        supportsDeveloperRole: false,
        supportsReasoningEffort: false
      },
      models: [
        {
          id: $model_id,
          name: ("mistral.rs local: " + $model_id),
          reasoning: false,
          input: ["text"],
          contextWindow: 32768,
          maxTokens: 8192,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }
        }
      ]
    }
  }
}' > "${PI_MODELS_FILE}"

# mistralrs run   -m Qwen/Qwen2.5-Coder-7B-Instruct --quant q4k --enable-shell --enable-code-execution # --agent --quant auto
# mistralrs serve -m Qwen/Qwen2.5-Coder-7B-Instruct --quant q4k --enable-shell --enable-code-execution --port 8080
#
# mistralrs run -m Qwen/Qwen2.5-Coder-7B-Instruct --quant q4k -i "Explica en una frase qué es un causal dilated conv"
# mistralrs run -m google/gemma-4-E4B-it --quant auto -i "Describe qué instrumentos aparecen en esta partitura" --image ~/partituras/entre_tango.png
# mistralrs run -m mistralai/Voxtral-Mini-3B-2507 --quant auto -i "Transcribe este audio" --audio ~/grabaciones/idea_piano.wav
# mistralrs serve -m Qwen/Qwen2.5-Coder-7B-Instruct --quant auto --port 8080 # http://localhost:8080/ui
# mistralrs serve -m Qwen/Qwen2.5-Coder-7B-Instruct --quant auto --enable-shell --enable-code-execution --port 8080 --no-ui
# mistralrs run --agent -m Qwen/Qwen2.5-Coder-7B-Instruct --quant q4k
# mistralrs doctor --json | jq '.system'
# mistralrs tune plain -m Qwen/Qwen2.5-Coder-7B-Instruct --profile quality
# mistralrs bench -m Qwen/Qwen2.5-Coder-7B-Instruct --quant q4k --prompt-len 512,2048 --gen-len 256
# mistralrs serve --format gguf -m TheBloke/Qwen2.5-Coder-7B-GGUF -f qwen2.5-coder-7b-q4_k_m.gguf --port 8080

# git clone https://github.com/EricLBuehler/mistral.rs
# cd mistral.rs
# cargo build --release -p mistralrs-cli
# ./target/release/mistralrs run -m Qwen/Qwen3-4B
