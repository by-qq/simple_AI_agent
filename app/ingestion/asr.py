# faster-whisper转写且带时间戳
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from faster_whisper import WhisperModel


@dataclass		# 数据类
class ASRSegment:
    start_s: float  	# 开始时间
    end_s: float	# 结束时间
    text: str		# 转成文字后的文本


class ASR:
    def __init__(self, model_name: str = "base", device: str = "cpu", compute_type: str = "int8"):
        # 我是m2的芯片，可以考虑device="cpu", compute_type="int8"这样比较省资源，有显卡可以device="cuda"，compute_type="float16"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: str, language: Optional[str] = None) -> tuple[List[ASRSegment], Optional[str]]:
        segments_iter, info = self.model.transcribe(
            wav_path,
            language=language,
            vad_filter=False,  # v0先关，后续v1加VAD
        )
        segs: List[ASRSegment] = []
        for s in segments_iter:
            txt = (s.text or "").strip()
            if not txt:
                continue
            segs.append(ASRSegment(start_s=float(s.start), end_s=float(s.end), text=txt))
        lang = getattr(info, "language", None)
        return segs, lang

if __name__ == "__main__":
    print(ASR(model_name="base", device="cpu", compute_type="int8").transcribe("/home/by/文档/xwechat_files/wxid_nezttd0pobqm22_87f1/msg/file/2025-12/wyinpin.m4a"))
