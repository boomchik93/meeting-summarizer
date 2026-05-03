import os
import sys
import gc
import types
import tempfile
import warnings
import numpy as np
from pathlib import Path
from collections import Counter
from contextlib import asynccontextmanager

import llm as llm_module

_stub = types.ModuleType('webrtcvad')
_stub.Vad = None
sys.modules.setdefault('webrtcvad', _stub)

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial.distance import cosine

warnings.filterwarnings("ignore")

# настройки
# юзаем bond005/whisper-podlodka-turbo — заточен под русскую речь с шумами
# https://huggingface.co/bond005/whisper-podlodka-turbo
HF_MODEL_ID  = os.getenv("WHISPER_HF_MODEL", "bond005/whisper-podlodka-turbo")
MODEL_SIZE   = os.getenv("WHISPER_MODEL", "large-v3")   # бэкап если HF сдохнет
SAMPLE_RATE  = 16_000
USE_HF_MODEL = os.getenv("USE_HF_MODEL", "1") != "0"   # 0 = форсим faster-whisper

# автодетект CUDA
if torch.cuda.is_available():
    DEVICE       = "cuda"
    COMPUTE_TYPE = "float16"
    TORCH_DTYPE  = torch.float16
    print(f"[устройство] CUDA найдена: {torch.cuda.get_device_name(0)}")
else:
    DEVICE       = "cpu"
    COMPUTE_TYPE = "int8"
    TORCH_DTYPE  = torch.float32
    print("[устройство] CUDA нет, юзаем CPU")

# настройки диаризации
DIAR_THRESHOLD      = 0.55   # порог для нового спикера
DIAR_MIN_SEG_SEC    = 0.8    # склеиваем короткие куски
DIAR_CONTEXT_SEC    = 2.0    # макс контекст
DIAR_SMOOTH_WINDOW  = 3      # сглаживание

whisper_model = None   # бэкап faster-whisper
hf_pipeline   = None   # основной transformers
encoder_model = None


# грузим модели
def load_models():
    global whisper_model, hf_pipeline, encoder_model

    if USE_HF_MODEL:
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            print(f"[whisper] грузим HF модель '{HF_MODEL_ID}' …")
            print(f"[whisper] откуда: https://huggingface.co/{HF_MODEL_ID}")
            hf_processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
            hf_model_obj = AutoModelForSpeechSeq2Seq.from_pretrained(
                HF_MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device_map="auto",
            )
            hf_model_obj.eval()
            # убираем forced_decoder_ids чтобы не было конфликта с task=transcribe
            hf_model_obj.generation_config.forced_decoder_ids = None
            hf_pipeline = (hf_model_obj, hf_processor)
            print(f"[whisper] HF готов")
        except Exception as e:
            print(f"[whisper] HF сдох ({e}), откат на faster-whisper")
            hf_pipeline = None

    if hf_pipeline is None:
        from faster_whisper import WhisperModel
        print(f"[whisper] грузим faster-whisper '{MODEL_SIZE}' …")
        whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        print("[whisper] faster готов")

    print("[resemblyzer] грузим энкодер …")
    from resemblyzer import VoiceEncoder
    encoder_model = VoiceEncoder(device=DEVICE)
    print("[resemblyzer] готов")


# хелперы аудио

def load_audio_mono(path: str) -> np.ndarray:
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze().numpy().astype(np.float32)


# возвращает (waveform_CHxT, num_channels)
def load_audio_channels(path: str) -> tuple[np.ndarray, int]:
    waveform, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.numpy().astype(np.float32), waveform.shape[0]


# транскрибим один моно канал, убираем дубли
def _transcribe_channel(audio: np.ndarray) -> list[dict]:
    segs_gen, _ = whisper_model.transcribe(
        audio,
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
        },
        language="ru",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-1.0,
        word_timestamps=False,
    )
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segs_gen
        if s.text.strip()
    ]
    # убираем дубли подряд
    deduped = []
    for seg in segments:
        if deduped and seg["text"] == deduped[-1]["text"]:
            deduped[-1]["end"] = seg["end"]
        else:
            deduped.append(seg)
    return deduped

# стерео диаризация: каждый канал отдельно
def diarize_stereo(ch0: np.ndarray, ch1: np.ndarray) -> list[dict]:
    print("[диар] стерео — канал 0 …")
    segs0 = _transcribe_channel(ch0)
    print(f"[диар] канал 0: {len(segs0)} сегментов")

    print("[диар] стерео — канал 1 …")
    segs1 = _transcribe_channel(ch1)
    print(f"[диар] канал 1: {len(segs1)} сегментов")

    for s in segs0:
        s["speaker"] = "SPEAKER_00"
    for s in segs1:
        s["speaker"] = "SPEAKER_01"

    merged = sorted(segs0 + segs1, key=lambda x: x["start"])
    return merged

# нормализуем float32 в float64 для resemblyzer
def norm_audio(audio: np.ndarray) -> np.ndarray:
    wav = audio.astype(np.float64)
    peak = np.abs(wav).max()
    if peak > 1e-6:
        wav = wav / peak * 0.95
    return wav


# диаризация
def _embed_window(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray | None:
    dur = end_s - start_s
    if dur < 0.1:
        return None

    # расширяем короткие сегменты контекстом
    if dur < DIAR_MIN_SEG_SEC:
        pad = (DIAR_MIN_SEG_SEC - dur) / 2
        start_s = max(0.0, start_s - pad)
        end_s   = min(len(audio) / SAMPLE_RATE, end_s + pad)

    # ограничиваем макс контекстом
    if (end_s - start_s) > DIAR_CONTEXT_SEC:
        mid = (start_s + end_s) / 2
        start_s = mid - DIAR_CONTEXT_SEC / 2
        end_s   = mid + DIAR_CONTEXT_SEC / 2

    s = int(start_s * SAMPLE_RATE)
    e = int(end_s   * SAMPLE_RATE)
    chunk = audio[s:e]

    if len(chunk) < int(SAMPLE_RATE * 0.3):
        return None

    try:
        return encoder_model.embed_utterance(norm_audio(chunk))
    except Exception:
        return None


def _smooth_labels(labels: list[int], window: int) -> list[int]:
    if window < 2 or len(labels) < window:
        return labels
    out = list(labels)
    half = window // 2
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        window_vals = labels[lo:hi]
        # мажоритарное голосование
        out[i] = max(set(window_vals), key=window_vals.count)
    return out


def diarize(audio: np.ndarray, segments: list[dict]) -> list[dict]:
    embeddings: list[np.ndarray | None] = []
    for seg in segments:
        emb = _embed_window(audio, seg["start"], seg["end"])
        embeddings.append(emb)

    # жадная кластеризация по центроидам
    # каждый центроид хранит список эмбеддингов для стабильного усреднения
    centroid_embs: list[list[np.ndarray]] = []
    centroids:     list[np.ndarray]       = []
    raw_labels:    list[int]              = []

    for emb in embeddings:
        if emb is None:
            raw_labels.append(raw_labels[-1] if raw_labels else 0)
            continue

        if not centroids:
            centroid_embs.append([emb.copy()])
            centroids.append(emb.copy())
            raw_labels.append(0)
            continue

        dists = [cosine(emb, c) for c in centroids]
        best  = int(np.argmin(dists))

        if dists[best] < DIAR_THRESHOLD:
            # обновляем центроид как среднее всех эмбеддингов (без дрифта)
            centroid_embs[best].append(emb.copy())
            centroids[best] = np.mean(centroid_embs[best], axis=0)
            raw_labels.append(best)
        else:
            # перед созданием нового кластера проверяем близость к существующим
            # (предотвращает фрагментацию от зашумленных сегментов)
            second_pass = [cosine(emb, c) for c in centroids]
            if min(second_pass) < DIAR_THRESHOLD * 1.3:
                best2 = int(np.argmin(second_pass))
                centroid_embs[best2].append(emb.copy())
                centroids[best2] = np.mean(centroid_embs[best2], axis=0)
                raw_labels.append(best2)
            else:
                centroid_embs.append([emb.copy()])
                centroids.append(emb.copy())
                raw_labels.append(len(centroids) - 1)

    smooth = _smooth_labels(raw_labels, DIAR_SMOOTH_WINDOW)

    label_counts = Counter(smooth)
    # ищем кластеры с малым количеством сегментов
    tiny = {lbl for lbl, cnt in label_counts.items() if cnt < 3}
    if tiny and len(label_counts) > 1:
        for i, lbl in enumerate(smooth):
            if lbl in tiny:
                # ищем ближайший центроид который НЕ мелкий
                valid = [j for j in range(len(centroids)) if j not in tiny]
                if valid and embeddings[i] is not None:
                    dists = [cosine(embeddings[i], centroids[j]) for j in valid]
                    smooth[i] = valid[int(np.argmin(dists))]
                elif valid:
                    smooth[i] = valid[0]

    # переиндексируем спикеров 0,1,2… по порядку появления
    remap: dict[int, int] = {}
    next_id = 0
    final_labels: list[int] = []
    for lbl in smooth:
        if lbl not in remap:
            remap[lbl] = next_id
            next_id += 1
        final_labels.append(remap[lbl])

    # навешиваем лейблы
    result = []
    for seg, label in zip(segments, final_labels):
        s = dict(seg)
        s["speaker"] = f"SPEAKER_{label:02d}"
        result.append(s)

    return result


def _run_whisper(audio: np.ndarray) -> tuple[list[dict], str]:
    if hf_pipeline is not None:
        return _run_whisper_hf(audio)
    return _run_whisper_faster(audio)


def _run_whisper_hf(audio: np.ndarray) -> tuple[list[dict], str]:
    from transformers import pipeline as hf_pipe

    hf_model_obj, hf_processor = hf_pipeline

    pipe = hf_pipe(
        "automatic-speech-recognition",
        model=hf_model_obj,
        tokenizer=hf_processor.tokenizer,
        feature_extractor=hf_processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
    )

    result = pipe(
        audio.copy(),
        generate_kwargs={
            "task": "transcribe",
            "language": "russian",
            "num_beams": 5,
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "return_legacy_cache": False,
        },
        return_timestamps=True,
    )

    chunks = result.get("chunks", [])
    segments = []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        ts    = chunk.get("timestamp", (0.0, 0.0))
        start = round(float(ts[0] or 0.0), 2)
        end   = round(float(ts[1] or start + 1.0), 2)
        segments.append({"start": start, "end": end, "text": text})

    # убираем дубли подряд идентичных сегментов
    deduped: list[dict] = []
    for seg in segments:
        if deduped and seg["text"] == deduped[-1]["text"]:
            deduped[-1]["end"] = seg["end"]
        else:
            deduped.append(seg)

    return deduped, "ru"


def _run_whisper_faster(audio: np.ndarray) -> tuple[list[dict], str]:
    segs_gen, info = whisper_model.transcribe(
        audio,
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
        },
        language="ru",
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.0,
        log_prob_threshold=-1.0,
        word_timestamps=False,
    )
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segs_gen
        if s.text.strip()
    ]
    deduped = []
    for seg in segments:
        if deduped and seg["text"] == deduped[-1]["text"]:
            deduped[-1]["end"] = seg["end"]
        else:
            deduped.append(seg)
    return deduped, info.language


def transcribe(audio_path: str) -> dict:
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    num_channels = waveform.shape[0]

    if num_channels >= 2:
        ch0 = waveform[0].numpy().astype(np.float32)
        ch1 = waveform[1].numpy().astype(np.float32)
        e0 = float(np.abs(ch0).mean())
        e1 = float(np.abs(ch1).mean())
        print(f"[аудио] стерео — ch0={e0:.5f}, ch1={e1:.5f}")

        if e0 > 1e-4 and e1 > 1e-4:
            print("[диар] стерео: каждый канал = спикер")
            segs0, lang = _run_whisper(ch0)
            segs1, _    = _run_whisper(ch1)
            for s in segs0: s["speaker"] = "SPEAKER_00"
            for s in segs1: s["speaker"] = "SPEAKER_01"
            segments = sorted(segs0 + segs1, key=lambda x: x["start"])
            language = lang
            print(f"[диар] стерео готово: {len(segments)} сегментов, 2 спикера")
        else:
            # один канал тихий — моно режим
            print("[аудио] один канал тихий, моно режим")
            audio = waveform.mean(dim=0).numpy().astype(np.float32)
            segments, language = _transcribe_mono(audio)

    else:
        print("[аудио] моно — юзаем resemblyzer")
        audio = waveform[0].numpy().astype(np.float32)
        segments, language = _transcribe_mono(audio)

    # Саммари по спикерам
    speakers: dict[str, str] = {}
    for s in segments:
        sp = s["speaker"]
        speakers[sp] = (speakers.get(sp, "") + " " + s["text"]).strip()

    gc.collect()

    return {
        "language": language,
        "text":     " ".join(s["text"] for s in segments),
        "segments": segments,
        "speakers": speakers,
    }


def _transcribe_mono(audio: np.ndarray) -> tuple[list[dict], str]:
    print("[whisper] транскрибим моно …")
    segments, language = _run_whisper(audio)
    print(f"[whisper] {len(segments)} сегментов, язык={language}")

    if encoder_model is not None and segments:
        print("[диар] назначаем спикеров …")
        segments = diarize(audio, segments)
        n = len({s["speaker"] for s in segments})
        print(f"[диар] найдено {n} спикеров")
    else:
        for s in segments:
            s["speaker"] = "SPEAKER_00"

    return segments, language


# FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    llm_module.load_llm()
    yield

app = FastAPI(title="Transcriber", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {
        "status":      "ok",
        "whisper":     whisper_model is not None or hf_pipeline is not None,
        "diarization": encoder_model is not None,
        "model":       HF_MODEL_ID if hf_pipeline is not None else MODEL_SIZE,
        "model_type":  "huggingface" if hf_pipeline is not None else "faster-whisper",
        "device":      DEVICE,
        "llm":         llm_module.is_ready(),
        "llm_model":   "Qwen2.5-7B" if llm_module.is_ready() else "not loaded",
    }


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXT)}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        data = transcribe(tmp_path)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

    # Автоматическая генерация саммари после транскрибации
    summary = None
    if llm_module.is_ready():
        try:
            print("[api] генерируем саммари через LLM ...")
            summary = llm_module.summarize(data["segments"], data["speakers"])
            print(f"[api] саммари готово: {len(summary.get('topics', []))} топиков")
        except Exception as e:
            print(f"[api] ошибка генерации саммари: {e}")
            summary = {"error": str(e), "topics": []}
    else:
        print("[api] LLM не загружен, саммари пропущено")

    return JSONResponse({
        "status":              "success",
        "filename":            file.filename,
        "language":            data["language"],
        "transcription":       data["text"],
        "segments":            data["segments"],
        "speakers":            data["speakers"],
        "summary":             summary,  # добавляем саммари в ответ
        "diarization_enabled": encoder_model is not None,
        "stereo_diarization":  len(data["speakers"]) == 2 and
                               "SPEAKER_00" in data["speakers"] and
                               "SPEAKER_01" in data["speakers"],
        "model":               HF_MODEL_ID if hf_pipeline is not None else MODEL_SIZE,
        "model_type":          "huggingface" if hf_pipeline is not None else "faster-whisper",
    })


@app.post("/api/summarize")
async def api_summarize(request: Request):
    if not llm_module.is_ready():
        raise HTTPException(503, "LLM not loaded. Check LLM_MODEL_PATH and ensure model file exists.")

    body = await request.json()
    segments = body.get("segments", [])
    speakers = body.get("speakers", {})

    if not segments:
        raise HTTPException(400, "No segments provided")

    result = llm_module.summarize(segments, speakers)
    return JSONResponse(result)


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
