import os
import json
import re
from pathlib import Path

# llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("[llm] llama-cpp-python не стоит — суммаризация выкл")

# настройки
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    "/models/qwen/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
)
LLM_N_CTX        = int(os.getenv("LLM_N_CTX",        "32768"))
LLM_N_THREADS    = int(os.getenv("LLM_N_THREADS",    str(os.cpu_count() or 8)))
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))  # -1 = все слои на GPU

_llm: "Llama | None" = None


def load_llm() -> bool:
    global _llm
    if not LLAMA_AVAILABLE:
        return False

    model_path = Path(LLM_MODEL_PATH)
    if not model_path.exists():
        print(f"[llm] модель не найдена в {model_path} — суммаризация выкл")
        print(f"[llm] скачать: huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir models/qwen/")
        return False

    print(f"[llm] грузим Qwen2.5-7B из {model_path} …")
    print(f"[llm] n_gpu_layers={LLM_N_GPU_LAYERS} ({'все слои на GPU' if LLM_N_GPU_LAYERS == -1 else 'только CPU' if LLM_N_GPU_LAYERS == 0 else f'{LLM_N_GPU_LAYERS} слоев на GPU'})")
    _llm = Llama(
        model_path=str(model_path),
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        verbose=False,
    )
    print("[llm] Qwen2.5-7B готов")
    return True


def is_ready() -> bool:
    return _llm is not None

# конвертим сегменты в текст
def _build_transcript_text(segments: list[dict]) -> str:
    lines = []
    prev_speaker = None
    for seg in segments:
        speaker = seg.get("speaker", "Спикер")
        text    = seg.get("text", "").strip()
        if not text:
            continue
        if speaker != prev_speaker:
            lines.append(f"\n{speaker}: {text}")
            prev_speaker = speaker
        else:
            lines.append(text)
    return " ".join(lines).strip()


_SYSTEM_PROMPT = """Ты — экспертный аналитик деловых коммуникаций и системный архитектор. Твоя задача — провести глубокую деконструкцию предоставленной транскрибации переговоров.

Твоя цель: Извлечь 100% фактологической информации без потери контекста. Ты должен зафиксировать каждое техническое решение, каждое сомнение, каждую упомянутую систему и организационную деталь.

ФОРМАТ ОТВЕТА (СТРОГО JSON):
{
  "summary": "Краткое резюме всего разговора в 2-3 предложениях",
  "topics": [
    {
      "title": "Название темы (максимально конкретное)",
      "category": "technical|business|organizational|decision|problem",
      "points": [
        "Факт, решение или проблема с конкретикой",
        "Техническая деталь или условие"
      ]
    }
  ],
  "decisions": [
    {
      "decision": "Принятое решение",
      "context": "Контекст и обоснование",
      "responsible": "Ответственный (если упомянут)"
    }
  ],
  "action_items": [
    {
      "action": "Что нужно сделать",
      "responsible": "Кто отвечает (если упомянут)",
      "deadline": "Срок (если упомянут)"
    }
  ],
  "risks": [
    {
      "risk": "Описание риска или проблемы",
      "impact": "Возможные последствия"
    }
  ],
  "key_points": [
    "Важный факт или инсайт из разговора",
    "Критическая информация"
  ]
}

ПРАВИЛА АНАЛИЗА:
1. Выделяй от 3 до 8 ключевых тем (topics). Каждая тема должна иметь категорию.
2. В каждой теме должно быть от 2 до 5 конкретных пунктов (points).
3. Решения (decisions) — это конкретные принятые решения, а не обсуждения.
4. Action items — это конкретные задачи, которые нужно выполнить.
5. Риски (risks) — упомянутые проблемы, ограничения, потенциальные сложности.
6. Key points — самые важные факты, которые нельзя упустить.

ОГРАНИЧЕНИЯ:
- Только JSON. Никакого вводного текста, пояснений или markdown-разметки (```json).
- Язык ответа — русский.
- Не обобщай: используй конкретные термины и названия из разговора.
- Если какая-то секция пустая (например, нет решений), оставь пустой массив []."""

# суммаризация через qwen, просим json
def summarize(segments: list[dict], speakers: dict | None = None) -> dict:
    if _llm is None:
        return {
            "error": "LLM not loaded",
            "summary": "",
            "topics": [],
            "decisions": [],
            "action_items": [],
            "risks": [],
            "key_points": []
        }

    transcript = _build_transcript_text(segments)
    if len(transcript) < 50:
        return {
            "error": "Transcript too short",
            "summary": "",
            "topics": [],
            "decisions": [],
            "action_items": [],
            "risks": [],
            "key_points": []
        }

    # обрезаем чтоб не переполнить контекст (~6000 символов ≈ ~1500 токенов)
    if len(transcript) > 6000:
        transcript = transcript[:6000] + "\n[...текст обрезан...]"

    user_msg = f"Составь структурированный пересказ следующего разговора:\n\n{transcript}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    # строим промпт через Qwen2.5 шаблон
    prompt = _format_qwen_chat(messages)

    try:
        output = _llm(
            prompt,
            max_tokens=2048,
            temperature=0.1,
            top_p=0.9,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
        )
        raw = output["choices"][0]["text"].strip()
        return _parse_json_response(raw)
    except Exception as e:
        return {
            "error": str(e),
            "summary": "",
            "topics": [],
            "decisions": [],
            "action_items": [],
            "risks": [],
            "key_points": []
        }

# форматируем сообщения для qwen
def _format_qwen_chat(messages: list[dict]) -> str:
    prompt = ""
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

# извлекаем и парсим json от llm
def _parse_json_response(raw: str) -> dict:
    # убираем markdown блоки если есть
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
        # проверяем что есть хотя бы одно из ожидаемых полей
        if any(key in data for key in ["summary", "topics", "decisions", "action_items", "risks", "key_points"]):
            # добавляем пустые массивы для отсутствующих полей
            data.setdefault("summary", "")
            data.setdefault("topics", [])
            data.setdefault("decisions", [])
            data.setdefault("action_items", [])
            data.setdefault("risks", [])
            data.setdefault("key_points", [])
            return data
        return {"error": "Unexpected JSON structure", "summary": "", "topics": [], "decisions": [], "action_items": [], "risks": [], "key_points": [], "raw": raw}
    except json.JSONDecodeError:
        # пытаемся найти JSON объект в тексте
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if any(key in data for key in ["summary", "topics", "decisions", "action_items", "risks", "key_points"]):
                    data.setdefault("summary", "")
                    data.setdefault("topics", [])
                    data.setdefault("decisions", [])
                    data.setdefault("action_items", [])
                    data.setdefault("risks", [])
                    data.setdefault("key_points", [])
                    return data
            except Exception:
                pass
        return {"error": "Failed to parse JSON", "summary": "", "topics": [], "decisions": [], "action_items": [], "risks": [], "key_points": [], "raw": raw[:500]}