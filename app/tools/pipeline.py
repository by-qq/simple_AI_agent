from __future__ import annotations

import json, math, os, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

import webrtcvad
from faster_whisper import WhisperModel
from langchain_core.documents import Document

from app.db import mysql_audio
from app.deps import get_audio_vs

ProgressFn = Callable[[int, str], None]

TARGET_SR = int(os.getenv("AUDIO_SR", "16000"))
TARGET_CH = int(os.getenv("AUDIO_CH", "1"))

VAD_MODE = int(os.getenv("VAD_MODE", "2"))
VAD_FRAME_MS = int(os.getenv("VAD_FRAME_MS", "30"))
VAD_PADDING_MS = int(os.getenv("VAD_PADDING_MS", "300"))
VAD_MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "500"))
VAD_MERGE_GAP_MS = int(os.getenv("VAD_MERGE_GAP_MS", "250"))

ASR_MODEL = os.getenv("ASR_MODEL", "base")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8")

MAX_CHUNK_MS = int(os.getenv("AUDIO_MAX_CHUNK_MS", "25000"))
MIN_CHUNK_MS = int(os.getenv("AUDIO_MIN_CHUNK_MS", "6000"))
MAX_CHARS_PER_CHUNK = int(os.getenv("AUDIO_MAX_CHARS_PER_CHUNK", "900"))

PUNCT_END = set("。.!?！？；;")

MAX_SPEECH_SEGMENTS = int(os.getenv("AUDIO_MAX_SPEECH_SEGMENTS", "2000"))


@dataclass
class SpeechSeg:
    start_ms: int
    end_ms: int


@dataclass
class AsrSeg:
    start_ms: int
    end_ms: int
    text: str

# 该函数是做进度处理的辅助函数
def _prog(cb: Optional[ProgressFn], p: int, m: str) -> None:
    if cb:
        cb(int(p), str(m))

# 运行一条系统命令
def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr[:4000]}")


def transcode_to_wav_16k_mono(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(dst)
    print(dst.parent)
    _run([
        "/home/by/下载/ffmpeg-master-latest-linux64-gpl.tar/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg",
        "-y",                   # 覆盖输出文件
        "-i", str(src),
        "-ac", str(TARGET_CH),
        "-ar", str(TARGET_SR),
        "-f", "wav",
        str(dst),
    ])


def ffprobe_duration_ms(path: Path) -> int:
    p = subprocess.run(
        ["/home/by/下载/ffmpeg-master-latest-linux64-gpl.tar/ffmpeg-master-latest-linux64-gpl/bin/ffprobe",
         "-v",     "error",
         "-show_entries",       "format=duration",
         "-of", "json", str(path)],
        stdout=subprocess.PIPE,     # 捕获标准输出
        stderr=subprocess.PIPE,     # 捕获错误输出
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr[:2000]}")
    data = json.loads(p.stdout or "{}")
    dur = float((data.get("format") or {}).get("duration") or 0.0)
    return int(dur * 1000)

def _read_wav_mono_16k(path: Path) -> np.ndarray:       # todo 小重点 返回numpy数组是为什么
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != TARGET_SR:
        raise RuntimeError(f"wav sample rate not {TARGET_SR}: {sr}")
    if isinstance(x, np.ndarray) and x.ndim == 2:
        x = x.mean(axis=1)
    return np.asarray(x, dtype=np.float32)


def _float_to_pcm16_bytes(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    return pcm.tobytes()


def detect_speech_segments(wav_path: Path) -> List[SpeechSeg]:
    # WebRTC VAD对输入音频有严格的要求：单声道，16位，采样率为8k,16k,32k,48k
    # 选择16kHz，平衡了计算效率和语音质量
    x = _read_wav_mono_16k(wav_path)        # 读取并重采样为16kHz的单声道
    pcm_bytes = _float_to_pcm16_bytes(x)    # 转换为16位PCM字节格式

    vad = webrtcvad.Vad(VAD_MODE)           # 创建VAD对象，使用预设的敏感度模式

    frame_len = int(TARGET_SR * (VAD_FRAME_MS / 1000.0))  # 每帧的样本数量
    frame_bytes = frame_len * 2                         # int16，每帧的字节数
    total_frames = len(pcm_bytes) // frame_bytes        # 总帧数

    def is_speech(i: int) -> bool:      # 语音帧检测，以固定帧长处理音频
        start = i * frame_bytes
        chunk = pcm_bytes[start:start + frame_bytes]
        if len(chunk) < frame_bytes:
            return False
        return vad.is_speech(chunk, sample_rate=TARGET_SR)

    # 遍历所有帧，识别语音段的开始和结束
    speech_frames: List[Tuple[int, int]] = []
    in_speech = False
    seg_start = 0
    for i in range(total_frames):
        sp = is_speech(i)
        if sp and not in_speech:
            in_speech = True
            seg_start = i
        elif (not sp) and in_speech:
            in_speech = False
            speech_frames.append((seg_start, i))

    if in_speech:
        speech_frames.append((seg_start, total_frames))
    # 边界扩展，在检测到语音段前后添加静音缓冲，确保不遗漏语音的起止部分
    pad_frames = int(math.ceil(VAD_PADDING_MS / VAD_FRAME_MS))
    out: List[SpeechSeg] = []
    for a, b in speech_frames:
        a2 = max(0, a - pad_frames)
        b2 = min(total_frames, b + pad_frames)
        start_ms = int(a2 * VAD_FRAME_MS)
        end_ms = int(b2 * VAD_FRAME_MS)
        if (end_ms - start_ms) >= VAD_MIN_SPEECH_MS:    # 过滤果断的语音段
            out.append(SpeechSeg(start_ms=start_ms, end_ms=end_ms))

    if not out:
        return []
    # 合并相似片段，若两个语音段之间的间隙小于阈值，则合并成一个片段
    merged: List[SpeechSeg] = [out[0]]
    for s in out[1:]:
        prev = merged[-1]
        if s.start_ms - prev.end_ms <= VAD_MERGE_GAP_MS:
            prev.end_ms = max(prev.end_ms, s.end_ms)
        else:
            merged.append(s)
    # 限制输出数量，确保返回的片段数量不超过上线
    if len(merged) > MAX_SPEECH_SEGMENTS:
        merged = merged[:MAX_SPEECH_SEGMENTS]

    return merged


def _load_asr_model() -> WhisperModel:
    return WhisperModel(
        ASR_MODEL,
        device=ASR_DEVICE,
        compute_type=ASR_COMPUTE_TYPE,
    )


def transcribe_segments(
    wav_path: Path,
    speech: List[SpeechSeg],
    *,
    language: Optional[str],
    on_progress: Optional[ProgressFn],
) -> List[AsrSeg]:
    if not speech:
        return []
    # 读取并预处理音频
    x = _read_wav_mono_16k(wav_path)
    # 加载ASR模型
    model = _load_asr_model()

    # 遍历语音片段，
    out: List[AsrSeg] = []
    for idx, seg in enumerate(speech):
        # 计算采样点位置（基于16kHz采样率）
        s0 = int(seg.start_ms * TARGET_SR / 1000)   # 开始采样点
        s1 = int(seg.end_ms * TARGET_SR / 1000)     # 结束采样点
        # 边界检查
        s0 = max(0, min(len(x), s0))
        s1 = max(0, min(len(x), s1))
        if s1 <= s0:   continue     # 条过无效片段

        clip = x[s0:s1]     # 提取音频片段
        segments, info = model.transcribe(  # 执行语音识别
            clip,
            language=language,
            vad_filter=False,   # 禁用VAD过滤
            beam_size=1,        # 使用贪心搜索
            condition_on_previous_text=False,   # 不依赖前文
        )
        # 计算进度，在20%-80%范围内报告进度
        pct = 20 + int(60 * (idx + 1) / max(1, len(speech)))
        _prog(on_progress, pct, f"asr {idx+1}/{len(speech)}")
        # 处理识别结果，将相对时间转换成绝对时间
        for s in segments:
            start_ms = seg.start_ms + int(float(s.start) * 1000)
            end_ms = seg.start_ms + int(float(s.end) * 1000)
            text = (s.text or "").strip()
            if not text:
                continue
            out.append(AsrSeg(start_ms=start_ms, end_ms=max(end_ms, start_ms + 1), text=text))
    # 返回按时间的排序结果
    out.sort(key=lambda t: (t.start_ms, t.end_ms))
    return out

def _ends_with_punct(t: str) -> bool: # 音频是否以某个标点作为结尾
    t = (t or "").strip()
    if not t:
        return False
    return t[-1] in PUNCT_END

# 每一个小的ASR（经过VAD）得到小的段，这个段落中包含该段的开始/结束时间，还有这一段的文字
# 该函数就是将小块合并成大块，这段可能会被问（怎么用VAD进行切割的，切割出来有什么，怎么控制在25秒之内，内容怎么控制在900个字中）
def merge_asr_to_chunks(asr: List[AsrSeg]) -> List[AsrSeg]:
    if not asr:
        return []

    chunks: List[AsrSeg] = []
    cur_start = asr[0].start_ms # 第一个分段开始
    cur_end = asr[0].end_ms     # 第一个分段结束
    buf: List[str] = [asr[0].text]  # 缓冲文本，不断的拼接

    def flush(force: bool = False) -> None:     # 负责将缓冲区内容输出为一个AsrSeg块
        nonlocal cur_start, cur_end, buf
        # 清理文本（去空格、拼接）
        txt = " ".join([b.strip() for b in buf if b.strip()]).strip()
        if not txt:
            buf = []
            return
        # 截断超常文本
        if len(txt) > MAX_CHARS_PER_CHUNK:
            txt = txt[:MAX_CHARS_PER_CHUNK]
        # 创建新的AsrSeg，并加入chunks
        chunks.append(AsrSeg(start_ms=cur_start, end_ms=cur_end, text=txt))
        buf = []    # 清空缓存区
    # 主逻辑循环，遍历ASR，
    for s in asr[1:]:
        next_end = max(cur_end, s.end_ms)
        next_txt = (buf[-1] if buf else "")
        span = next_end - cur_start

        buf.append(s.text)
        cur_end = next_end

        span = cur_end - cur_start
        # 时长超限强制切割
        if span >= MAX_CHUNK_MS:
            flush(force=True)   # 强制输出当前块
            # 重置为当前分段
            cur_start = s.start_ms
            cur_end = s.end_ms
            buf = [s.text]
            continue
        # 满足最小时长且以标点结尾
        if span >= MIN_CHUNK_MS and _ends_with_punct(s.text):
            flush() # 正常输出
            # 重置为当前分段
            cur_start = s.start_ms
            cur_end = s.end_ms
            buf = [s.text]

    if buf: # 处理生于缓冲区内容
        flush(force=True)

    chunks.sort(key=lambda t: (t.start_ms, t.end_ms))   # 按时间排序
    return chunks

def _vs_add(vs: Any, docs: List[Document], ids: List[str]) -> None:
    if hasattr(vs, "add_documents"):
        vs.add_documents(docs, ids=ids)
        return
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    if hasattr(vs, "add_texts"):
        vs.add_texts(texts, metadatas=metas, ids=ids)
        return
    raise RuntimeError("Vectorstore does not support add_documents/add_texts")


def _db_replace_segments(audio_id: str, rows: List[Dict[str, Any]]) -> None:
    if hasattr(mysql_audio, "replace_audio_segments"):
        mysql_audio.replace_audio_segments(audio_id, rows)
        return

    if hasattr(mysql_audio, "delete_audio_segments") and hasattr(mysql_audio, "insert_audio_segments_bulk"):
        mysql_audio.delete_audio_segments(audio_id)
        mysql_audio.insert_audio_segments_bulk(audio_id,rows)
        return

    raise AttributeError("mysql_audio.replace_audio_segments not found (and no fallback delete/insert found)")

def run_audio_ingest_pipeline(
    *,
    audio_id: str,
    raw_path: Path,
    original_filename: str,
    visibility: str,
    language: Optional[str],
    wav_dir: Path,
    on_progress: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    if not raw_path.exists():
        raise FileNotFoundError(str(raw_path))

    _prog(on_progress, 1, "start")

    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path = wav_dir / f"{audio_id}.wav"

    _prog(on_progress, 5, "transcoding")
    transcode_to_wav_16k_mono(raw_path, wav_path)
    duration_ms = ffprobe_duration_ms(wav_path)

    # 从音频中检测出包含语音的片段
    _prog(on_progress, 10, "vad")
    speech = detect_speech_segments(wav_path)

    # 语音撰写ASR，将音频转化成文本
    _prog(on_progress, 15, f"vad segments={len(speech)}")
    asr = transcribe_segments(wav_path, speech, language=language, on_progress=on_progress)

    # 将ASR分段合并为更大块的函数
    _prog(on_progress, 85, f"asr segments={len(asr)}")
    chunks = merge_asr_to_chunks(asr)

    _prog(on_progress, 88, f"chunks={len(chunks)}")

    rows: List[Dict[str, Any]] = []
    docs: List[Document] = []
    ids: List[str] = []

    for i, c in enumerate(chunks):
        seg_id = f"{audio_id}:{i}"
        start_ms = int(c.start_ms)
        end_ms = int(c.end_ms)
        text = (c.text or "").strip()

        rows.append({
            "audio_id": audio_id,
            "segment_idx": i,
            "segment_id": seg_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text,
            "visibility": visibility,
        })

        meta = {
            "audio_id": audio_id,
            "segment_id": seg_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "visibility": visibility,
            "original_filename": original_filename,
        }
        docs.append(Document(page_content=text, metadata=meta))
        ids.append(seg_id)

    _prog(on_progress, 90, "write db segments")
    _db_replace_segments(audio_id, rows)

    _prog(on_progress, 93, "write vectors")
    vs = get_audio_vs()
    _vs_add(vs, docs, ids)

    _prog(on_progress, 100, "done")

    return {
        "audio_id": audio_id,
        "duration_ms": int(duration_ms),
        "segments": int(len(chunks)),
        "language": language,
        "wav_path": str(wav_path),
    }

