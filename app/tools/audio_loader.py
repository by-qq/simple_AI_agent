from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_dir(p: Path) -> None:  # 确保路径存在
    p.mkdir(parents=True, exist_ok=True)


def ffprobe_duration_ms(src: Path) -> int:  # 这段音频的时长，以毫秒为单位计数
    cmd = [
        "/home/by/下载/ffmpeg-master-latest-linux64-gpl.tar/ffmpeg-master-latest-linux64-gpl/bin/ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(src),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
    if not out:
        return 0
    sec = float(out)
    return int(sec * 1000)  # 这段音频的时长，以毫秒为单位计数


def transcode_to_wav_16k_mono(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    cmd = [
        "/home/by/下载/ffmpeg-master-latest-linux64-gpl.tar/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg",
        "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        str(dst),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
