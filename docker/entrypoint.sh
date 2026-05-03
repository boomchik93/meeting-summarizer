#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
# Точка входа: читает config.yml, качает недостающие модели, запускает сервер
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

CONFIG="/app/config.yml"
MODELS_DIR="/models"
PYTHON="/opt/venv/bin/python"

log() { echo "[entrypoint] $*"; }

# парсим config.yml → экспортируем shell переменные
eval "$("$PYTHON" - "$CONFIG" <<'PYEOF'
import sys, yaml

cfg = yaml.safe_load(open(sys.argv[1]))

out = {
    "CFG_PORT":             str(cfg["server"]["port"]),
    "CFG_WORKERS":          str(cfg["server"]["workers"]),
    "CFG_LOG_LEVEL":        cfg["server"]["log_level"],
    "CFG_WHISPER_HF_MODEL": cfg["whisper"]["hf_model"],
    "CFG_USE_HF_MODEL":     "1" if cfg["whisper"]["use_hf_model"] else "0",
    "CFG_WHISPER_MODEL":    cfg["whisper"]["fallback_model"],
}

llm = cfg.get("llm", {})
out.update({
    "CFG_LLM_ENABLED":      "1" if llm.get("enabled", False) else "0",
    "CFG_LLM_REPO":         llm.get("model", {}).get("repo", ""),
    "CFG_LLM_FILENAME":     llm.get("model", {}).get("filename", ""),
    "CFG_LLM_LOCAL":        llm.get("model", {}).get("local", ""),
    "CFG_LLM_N_CTX":        str(llm.get("n_ctx", 8192)),
    "CFG_LLM_N_THREADS":    str(llm.get("n_threads", 4)),
    "CFG_LLM_N_GPU_LAYERS": str(llm.get("n_gpu_layers", -1)),
})

for k, v in out.items():
    print(f"export {k}={v!r}")
PYEOF
)"

log "Конфиг загружен"
log "  Whisper : $CFG_WHISPER_HF_MODEL"
log "  LLM     : $CFG_LLM_FILENAME (gpu_layers=$CFG_LLM_N_GPU_LAYERS)"

# создаем папки моделей (volume пустой при первом запуске)
mkdir -p "$MODELS_DIR/qwen" "$MODELS_DIR/resemblyzer" "$MODELS_DIR/whisper"

# качаем GGUF если нет
if [ "$CFG_LLM_ENABLED" = "1" ] && [ -n "$CFG_LLM_LOCAL" ] && [ ! -f "$CFG_LLM_LOCAL" ]; then
    log "Качаем $CFG_LLM_FILENAME из $CFG_LLM_REPO …"
    "$PYTHON" - <<PYEOF
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id  = os.environ["CFG_LLM_REPO"],
    filename = os.environ["CFG_LLM_FILENAME"],
    local_dir= os.path.dirname(os.environ["CFG_LLM_LOCAL"]),
    token    = os.environ.get("HF_TOKEN") or None,
)
print(f"[entrypoint] сохранено → {path}")
PYEOF
else
    [ -f "$CFG_LLM_LOCAL" ] && log "LLM уже есть: $CFG_LLM_LOCAL"
fi

# передаем значения конфига в приложение через env
export WHISPER_HF_MODEL="$CFG_WHISPER_HF_MODEL"
export WHISPER_MODEL="$CFG_WHISPER_MODEL"
export USE_HF_MODEL="$CFG_USE_HF_MODEL"
export LLM_MODEL_PATH="$CFG_LLM_LOCAL"
export LLM_N_CTX="$CFG_LLM_N_CTX"
export LLM_N_THREADS="$CFG_LLM_N_THREADS"
export LLM_N_GPU_LAYERS="$CFG_LLM_N_GPU_LAYERS"
export PORT="$CFG_PORT"

log "Запускаем сервер на порту $CFG_PORT …"

exec /opt/venv/bin/uvicorn app:app \
    --host 0.0.0.0 \
    --port "$CFG_PORT" \
    --workers "$CFG_WORKERS" \
    --log-level "$CFG_LOG_LEVEL"
