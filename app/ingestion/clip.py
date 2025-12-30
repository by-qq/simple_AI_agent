from __future__ import annotations

import subprocess
from pathlib import Path

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def clip_audio_to_mp3(
    src_path: Path,
    dst_path: Path,
    start_ms: int,
    end_ms: int,
    sample_rate: int = 16000,
    channels: int = 1,
    bitrate: str = "96k",
) -> None:
    ensure_dir(dst_path.parent)

    start_s = max(0.0, float(start_ms) / 1000.0)
    end_s = max(0.0, float(end_ms) / 1000.0)
    if end_s <= start_s:
        raise ValueError("end_ms must be greater than start_ms")

    cmd = [
        "/home/by/下载/ffmpeg-master-latest-linux64-gpl.tar/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg",
        "-y",
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-i", str(src_path),
        "-vn",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        str(dst_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)