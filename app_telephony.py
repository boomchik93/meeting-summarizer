"""
Оптимизированная версия для телефонного трафика 8kHz
- Предобработка narrowband audio (8kHz → 16kHz с фильтрацией)
- Поддержка локальных моделей без HuggingFace
- Улучшенная диаризация для телефонии
- Универсальная поддержка CUDA/MPS/CPU
"""
import os
import sys
import gc
import types
import tempfile
import warnings
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager

import llm as llm_module

# патч: resemblyzer → webrtcvad → pkg_resources
_stub = types.ModuleType('webrtcvad')
_stub.Vad = None
sys.modules.setdefault('webrtcvad', _stub)

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from scipy.spatial.distance import cosine
from scipy import signal

warnings.filterwarnings("ignore")

# настройки
MODEL_SIZE   = os.getenv("WHISPER_MODEL", "large-v3")
SAMPLE_RATE  = 16_000
TELEPHONY_SR = 8_000  # узкополосная телефония

# оптимизация: все ядра для предобработки
NUM_THREADS  = os.cpu_count() or 8
torch.set_num_threads(NUM_THREADS)
torch.set_num_interop_threads(NUM_THREADS)

# автодетект устройства
if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
    DEVICE       = "cuda"
    COMPUTE_TYPE = "float16"
    TORCH_DTYPE  = torch.float16
    print(f"[устройство] CUDA найдена: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    TORCH_DEVICE = "mps"  # для PyTorch операций (предобработка)
    DEVICE       = "cpu"  # для faster-whisper (ctranslate2 не поддерживает MPS)
    COMPUTE_TYPE = "int8"
    TORCH_DTYPE  = torch.float32
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    print(f"[устройство] MPS найден — MPS для предобработки, CPU для инференса, {NUM_THREADS} потоков")
else:
    TORCH_DEVICE = "cpu"
    DEVICE       = "cpu"
    COMPUTE_TYPE = "int8"
    TORCH_DTYPE  = torch.float32
    print(f"[устройство] CPU — {NUM_THREADS} потоков")

# настройки диаризации для телефонии
DIAR_THRESHOLD      = 0.50   # ниже для телефона (голоса похожи)
DIAR_MIN_SEG_SEC    = 1.0    # длиннее сегменты для телефона
DIAR_CONTEXT_SEC    = 3.0    # больше контекста для шумного аудио
DIAR_SMOOTH_WINDOW  = 5      # сильнее сглаживание

whisper_model = None
encoder_model = None


# предобработка телефонного аудио

def enhance_telephony_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    # полосовой фильтр (300-3400 Hz — стандартный телефонный диапазон)
    nyquist = sr / 2
    low = 300 / nyquist
    high = min(3400 / nyquist, 0.95)
    
    # секции второго порядка для численной стабильности
    sos = signal.butter(4, [low, high], btype='band', output='sos')
    filtered = signal.sosfilt(sos, audio)
    
    # векторизованный noise gate
    rms = np.sqrt(np.mean(filtered ** 2))
    gate_threshold = rms * 0.1
    # плавный gate через tanh для лучшего качества
    gate_factor = np.tanh((np.abs(filtered) - gate_threshold) / (gate_threshold + 1e-8))
    gate_factor = np.maximum(gate_factor, 0)
    gated = filtered * gate_factor
    
    # нормализация
    peak = np.abs(gated).max()
    if peak > 1e-6:
        gated = gated / peak * 0.95
    
    return gated.astype(np.float32)


def upsample_8k_to_16k(audio: np.ndarray) -> np.ndarray:
    # улучшаем перед апсемплингом
    enhanced = enhance_telephony_audio(audio, TELEPHONY_SR)
    
    # полифазный ресемплинг с окном Kaiser (лучшее качество)
    # современные процессоры эффективно обрабатывают эти операции
    upsampled = signal.resample_poly(
        enhanced, 
        2, 1,  # 8k * 2 = 16k
        window=('kaiser', 5.0),  # лучше качество чем по умолчанию
        padtype='line'
    )
    
    return upsampled.astype(np.float32)


def load_audio_smart(path: str) -> tuple[np.ndarray, int, bool]:
    waveform, sr = torchaudio.load(path)
    is_telephony = sr == TELEPHONY_SR
    
    print(f"[аудио] загружено: {sr}Hz, {waveform.shape[0]} каналов")
    
    # Обработка каждого канала отдельно
    if sr == TELEPHONY_SR:
        print("[аудио] телефонный режим: улучшаем 8kHz аудио")
        channels = []
        for ch in range(waveform.shape[0]):
            ch_audio = waveform[ch].numpy()
            upsampled = upsample_8k_to_16k(ch_audio)
            channels.append(upsampled)
        waveform = np.stack(channels)
    else:
        # стандартный ресемплинг для не-телефонии
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.numpy().astype(np.float32)
    
    return waveform, waveform.shape[0], is_telephony


# загрузка моделей

def load_models():
    global whisper_model, encoder_model
    
    from faster_whisper import WhisperModel
    print(f"[whisper] грузим faster-whisper '{MODEL_SIZE}' …")
    
    # оптимизация: все потоки CPU для инференса
    # ctranslate2 оптимизирован для разных архитектур и будет использовать доступные инструкции
    whisper_model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=NUM_THREADS,    # все доступные ядра
        num_workers=1,              # один воркер (параллельная обработка в ctranslate2)
        download_root=None,
        local_files_only=False,
    )
    print(f"[whisper] готов — {COMPUTE_TYPE} квантизация, {NUM_THREADS} потоков")
    
    print("[resemblyzer] грузим энкодер …")
    from resemblyzer import VoiceEncoder
    # Resemblyzer на CPU (MPS не поддерживается)
    encoder_model = VoiceEncoder(device="cpu")
    print("[resemblyzer] готов")


# транскрибация

def _transcribe_channel(audio: np.ndarray, is_telephony: bool = False) -> list[dict]:
    # параметры VAD для телефонии (агрессивнее)
    vad_params = {
        "min_silence_duration_ms": 800 if is_telephony else 500,
        "speech_pad_ms": 600 if is_telephony else 400,
        "threshold": 0.4 if is_telephony else 0.5,
    }
    
    segs_gen, _ = whisper_model.transcribe(
        audio,
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6] if is_telephony else [0.0, 0.2, 0.4],
        vad_filter=True,
        vad_parameters=vad_params,
        language="ru",
        condition_on_previous_text=False,
        no_speech_threshold=0.65 if is_telephony else 0.6,
        compression_ratio_threshold=2.4 if is_telephony else 2.0,
        log_prob_threshold=-1.0,
        word_timestamps=False,
    )
    
    segments = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segs_gen
        if s.text.strip()
    ]
    
    # убираем дубли
    deduped = []
    for seg in segments:
        if deduped and seg["text"] == deduped[-1]["text"]:
            deduped[-1]["end"] = seg["end"]
        else:
            deduped.append(seg)
    
    return deduped


def norm_audio(audio: np.ndarray) -> np.ndarray:
    wav = audio.astype(np.float64)
    peak = np.abs(wav).max()
    if peak > 1e-6:
        wav = wav / peak * 0.95
    return wav


# диаризация (улучшенная для телефонии)

def _embed_window(audio: np.ndarray, start_s: float, end_s: float) -> np.ndarray | None:
    dur = end_s - start_s
    if dur < 0.1:
        return None
    
    # расширяем короткие сегменты
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
    
    if len(chunk) < int(SAMPLE_RATE * 0.4):
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
        out[i] = max(set(window_vals), key=window_vals.count)
    return out


def diarize(audio: np.ndarray, segments: list[dict]) -> list[dict]:
    # вычисляем эмбеддинги
    embeddings: list[np.ndarray | None] = []
    for seg in segments:
        emb = _embed_window(audio, seg["start"], seg["end"])
        embeddings.append(emb)
    
    # жадная кластеризация
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
            centroid_embs[best].append(emb.copy())
            centroids[best] = np.mean(centroid_embs[best], axis=0)
            raw_labels.append(best)
        else:
            centroid_embs.append([emb.copy()])
            centroids.append(emb.copy())
            raw_labels.append(len(centroids) - 1)
    
    smooth = _smooth_labels(raw_labels, DIAR_SMOOTH_WINDOW)
    
    from collections import Counter
    label_counts = Counter(smooth)
    tiny = {lbl for lbl, cnt in label_counts.items() if cnt < 3}
    if tiny and len(label_counts) > 1:
        for i, lbl in enumerate(smooth):
            if lbl in tiny:
                valid = [j for j in range(len(centroids)) if j not in tiny]
                if valid and embeddings[i] is not None:
                    dists = [cosine(embeddings[i], centroids[j]) for j in valid]
                    smooth[i] = valid[int(np.argmin(dists))]
                elif valid:
                    smooth[i] = valid[0]
    
    remap: dict[int, int] = {}
    next_id = 0
    final_labels: list[int] = []
    for lbl in smooth:
        if lbl not in remap:
            remap[lbl] = next_id
            next_id += 1
        final_labels.append(remap[lbl])
    
    result = []
    for seg, label in zip(segments, final_labels):
        s = dict(seg)
        s["speaker"] = f"SPEAKER_{label:02d}"
        result.append(s)
    
    return result


def transcribe(audio_path: str) -> dict:
    waveform, num_channels, is_telephony = load_audio_smart(audio_path)
    
    # Stereo path (идеально для телефонии: 2 канала = 2 спикера)
    if num_channels >= 2:
        ch0 = waveform[0]
        ch1 = waveform[1]
        e0 = float(np.abs(ch0).mean())
        e1 = float(np.abs(ch1).mean())
        print(f"[аудио] стерео — ch0={e0:.5f}, ch1={e1:.5f}")
        
        if e0 > 1e-4 and e1 > 1e-4:
            print("[диар] стерео режим: каждый канал = один спикер")
            segs0 = _transcribe_channel(ch0, is_telephony)
            segs1 = _transcribe_channel(ch1, is_telephony)
            
            for s in segs0: s["speaker"] = "SPEAKER_00"
            for s in segs1: s["speaker"] = "SPEAKER_01"
            
            segments = sorted(segs0 + segs1, key=lambda x: x["start"])
            print(f"[диар] стерео готово: {len(segments)} сегментов")
        else:
            print("[аудио] один канал тихий, откат на моно")
            audio = waveform.mean(axis=0)
            segments = _transcribe_mono(audio, is_telephony)
    
    # Mono path
    else:
        print("[аудио] моно — юзаем resemblyzer диаризацию")
        audio = waveform[0]
        segments = _transcribe_mono(audio, is_telephony)
    
    # Speakers summary
    speakers: dict[str, str] = {}
    for s in segments:
        sp = s["speaker"]
        speakers[sp] = (speakers.get(sp, "") + " " + s["text"]).strip()
    
    gc.collect()
    
    return {
        "language": "ru",
        "text":     " ".join(s["text"] for s in segments),
        "segments": segments,
        "speakers": speakers,
        "is_telephony": is_telephony,
    }


def _transcribe_mono(audio: np.ndarray, is_telephony: bool) -> list[dict]:
    print("[whisper] транскрибим моно …")
    segments = _transcribe_channel(audio, is_telephony)
    print(f"[whisper] {len(segments)} сегментов")
    
    if encoder_model is not None and segments:
        print("[диар] назначаем спикеров …")
        segments = diarize(audio, segments)
        n = len({s["speaker"] for s in segments})
        print(f"[диар] найдено {n} спикеров")
    else:
        for s in segments:
            s["speaker"] = "SPEAKER_00"
    
    return segments


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    llm_module.load_llm()
    yield

app = FastAPI(title="Telephony Transcriber", lifespan=lifespan)

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
        "whisper":     whisper_model is not None,
        "diarization": encoder_model is not None,
        "model":       MODEL_SIZE,
        "device":      DEVICE,
        "compute_type": COMPUTE_TYPE,
        "cpu_threads": NUM_THREADS,
        "telephony_support": True,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "llm":         llm_module.is_ready(),
        "llm_model":   "Qwen2.5-7B" if llm_module.is_ready() else "not loaded",
    }


@app.post("/api/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported: {ext}")
    
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
    try:
        import llm as llm_module
        if llm_module.is_ready():
            print("[api] генерируем саммари через LLM ...")
            summary = llm_module.summarize(data["segments"], data["speakers"])
            print(f"[api] саммари готово: {len(summary.get('topics', []))} топиков")
        else:
            print("[api] LLM не загружен, саммари пропущено")
    except Exception as e:
        print(f"[api] ошибка генерации саммари: {e}")
        summary = {"error": str(e), "topics": []}
    
    return JSONResponse({
        "status":              "success",
        "filename":            file.filename,
        "language":            data["language"],
        "transcription":       data["text"],
        "segments":            data["segments"],
        "speakers":            data["speakers"],
        "summary":             summary,  # добавляем саммари в ответ
        "is_telephony":        data["is_telephony"],
        "diarization_enabled": encoder_model is not None,
        "model":               MODEL_SIZE,
        "device":              DEVICE,
    })


app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)