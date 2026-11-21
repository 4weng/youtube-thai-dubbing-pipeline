"""Stage 2: Context-aware EN->TH translation, length-fitting, and deterministic Thai TTS normalization."""

from __future__ import annotations

import html
import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Protocol

from src.state import (
    ProviderError,
    SegmentTHDict,
    StageError,
    WorkspacePaths,
    atomic_write_json,
    is_valid_json_file,
    read_json,
)

logger = logging.getLogger(__name__)


class Translator(Protocol):
    def translate_segment(
        self,
        text_en: str,
        prev_context: list[str],
        next_context: list[str],
        slot_duration: float,
        max_chars: int,
    ) -> str:
        ...


class OpenAICompatibleTranslator:
    """Context-aware EN->TH dubbing translator using an OpenAI-compatible HTTP API (stdlib urllib)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        if not self.api_key:
            raise ProviderError(
                "OPENAI_API_KEY is required for OpenAICompatibleTranslator."
            )

    def translate_segment(
        self,
        text_en: str,
        prev_context: list[str],
        next_context: list[str],
        slot_duration: float,
        max_chars: int,
    ) -> str:
        prev_str = " | ".join(prev_context) if prev_context else "(none)"
        next_str = " | ".join(next_context) if next_context else "(none)"
        system_prompt = (
            "You are a professional English-to-Thai video dubbing translator. "
            "Translate the target English line into natural, spoken Thai. "
            "Keep the translation concise so it fits within the speaking time slot "
            "and character budget. Output ONLY the Thai translation with no quotes or notes."
        )
        user_prompt = (
            f"Previous context: {prev_str}\n"
            f"Target line ({slot_duration:.2f}s, target <= {max_chars} Thai chars): {text_en}\n"
            f"Next context: {next_str}\n"
            "Thai translation:"
        )
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"OpenAICompatibleTranslator request failed: {exc}") from exc

        try:
            text_th = body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"Unexpected OpenAI API response format: {body}") from exc

        if not text_th:
            raise ProviderError("OpenAICompatibleTranslator returned empty translation.")
        return text_th


class DeepTranslatorProvider:
    """Fallback EN->TH translator via deep-translator (GoogleTranslator, no API key required)."""

    def __init__(self, source: str = "en", target: str = "th") -> None:
        self.source = source
        self.target = target

    def translate_segment(
        self,
        text_en: str,
        prev_context: list[str],
        next_context: list[str],
        slot_duration: float,
        max_chars: int,
    ) -> str:
        try:
            from deep_translator import GoogleTranslator
        except ImportError as exc:
            raise ProviderError("deep-translator is not installed.") from exc

        try:
            translator = GoogleTranslator(source=self.source, target=self.target)
            result = translator.translate(text_en)
        except Exception as exc:
            raise ProviderError(f"DeepTranslatorProvider failed: {exc}") from exc

        if not result or not str(result).strip():
            raise ProviderError(f"DeepTranslatorProvider returned empty translation for {text_en!r}")
        return str(result).strip()


def create_translator(name: str = "auto", **kwargs: Any) -> Translator:
    """Select translator backend: 'auto' uses OpenAI if OPENAI_API_KEY is set, else DeepTranslator."""
    normalized = (name or "auto").strip().lower()
    if normalized == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAICompatibleTranslator(**kwargs)
        return DeepTranslatorProvider(**kwargs)
    if normalized in ("openai", "openai_compatible"):
        return OpenAICompatibleTranslator(**kwargs)
    if normalized in ("deep_translator", "deep-translator", "google"):
        return DeepTranslatorProvider(**kwargs)
    raise ProviderError(f"Unsupported translator provider: {name!r}")


def estimate_max_thai_chars(
    slot_duration: float,
    chars_per_sec: float = 14.0,
    min_chars: int = 12,
) -> int:
    """Compute target Thai character budget proportional to `slot_duration = end - start`."""
    budget = int(round(max(0.0, float(slot_duration)) * float(chars_per_sec)))
    return max(int(min_chars), budget)


_THAI_DIGITS = ("ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า")
_THAI_POSITIONS = ("", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน")


def _int_under_million_to_thai(n: int) -> str:
    if n == 0:
        return ""
    digits = [int(ch) for ch in str(n)]
    length = len(digits)
    parts: list[str] = []
    for idx, d in enumerate(digits):
        pos = length - idx - 1  # 0=ones, 1=tens, ..., 5=hundred-thousands
        if d == 0:
            continue
        if pos == 0:
            if d == 1 and length > 1:
                parts.append("เอ็ด")
            else:
                parts.append(_THAI_DIGITS[d])
        elif pos == 1:
            if d == 1:
                parts.append("สิบ")
            elif d == 2:
                parts.append("ยี่สิบ")
            else:
                parts.append(_THAI_DIGITS[d] + "สิบ")
        else:
            parts.append(_THAI_DIGITS[d] + _THAI_POSITIONS[pos])
    return "".join(parts)


def int_to_thai_words(n: int) -> str:
    """Pure-Python deterministic Thai cardinal number converter (handles negative, 0, สิบ/ยี่สิบ/เอ็ด, ล้าน+)."""
    if n < 0:
        return "ลบ" + int_to_thai_words(-n)
    if n == 0:
        return "ศูนย์"

    chunks: list[int] = []
    rem = n
    while rem > 0:
        chunks.append(rem % 1_000_000)
        rem //= 1_000_000

    parts: list[str] = []
    for idx in range(len(chunks) - 1, -1, -1):
        val = chunks[idx]
        if val > 0:
            parts.append(_int_under_million_to_thai(val))
        if idx > 0:
            parts.append("ล้าน")
    return "".join(parts)


def number_to_thai_words(num_str: str) -> str:
    """Convert a comma-grouped integer or decimal string into Thai spoken words."""
    s = num_str.strip().replace(",", "")
    if not s:
        return ""

    neg = False
    if s.startswith("-"):
        neg = True
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]

    if "." in s:
        int_part, frac_part = s.split(".", 1)
        int_words = int_to_thai_words(int(int_part)) if int_part else "ศูนย์"
        frac_words = "".join(_THAI_DIGITS[int(ch)] for ch in frac_part if ch.isdigit())
        result = f"{int_words}จุด{frac_words}" if frac_words else int_words
    else:
        result = int_to_thai_words(int(s))

    return ("ลบ" + result) if neg else result


_CURRENCY_SCALE = {
    "k": "พัน",
    "m": "ล้าน",
    "b": "พันล้าน",
    "t": "ล้านล้าน",
}

_CURRENCY_NAMES = {
    "$": "ดอลลาร์",
    "฿": "บาท",
    "€": "ยูโร",
    "£": "ปอนด์",
    "¥": "เยน",
}

_UNIT_NAMES = {
    "khz": "กิโลเฮิรตซ์",
    "mhz": "เมกะเฮิรตซ์",
    "ghz": "กิกะเฮิรตซ์",
    "hz": "เฮิรตซ์",
    "tb": "เทราไบต์",
    "gb": "กิกะไบต์",
    "mb": "เมกะไบต์",
    "kb": "กิโลไบต์",
    "fps": "เฟรมต่อวินาที",
    "ms": "มิลลิวินาที",
    "km": "กิโลเมตร",
    "cm": "เซนติเมตร",
    "mm": "มิลลิเมตร",
    "kg": "กิโลกรัม",
    "mg": "มิลลิกรัม",
    "kbps": "กิโลบิตต่อวินาที",
    "mbps": "เมกะบิตต่อวินาที",
    "gbps": "กิกะบิตต่อวินาที",
    "x": "เท่า",
    "k": "เค",
}

_ACRONYM_MAP = {
    "AI": "เอไอ",
    "API": "เอพีไอ",
    "ASR": "เอเอสอาร์",
    "CLI": "ซีแอลไอ",
    "CPU": "ซีพียู",
    "CSS": "ซีเอสเอส",
    "GPU": "จีพียู",
    "HTML": "เอชทีเอ็มแอล",
    "HTTP": "เอชทีทีพี",
    "HTTPS": "เอชทีทีพีเอส",
    "JSON": "เจสัน",
    "LLM": "แอลแอลเอ็ม",
    "MP3": "เอ็มพีสาม",
    "MP4": "เอ็มพีสี่",
    "ONNX": "ออนนิกซ์",
    "OS": "โอเอส",
    "PCM": "พีซีเอ็ม",
    "RAM": "แรม",
    "SDK": "เอสดีเค",
    "SQL": "เอสคิวแอล",
    "SRT": "เอสอาร์ที",
    "SSD": "เอสเอสดี",
    "TTS": "ทีทีเอส",
    "UI": "ยูไอ",
    "URL": "ยูอาร์แอล",
    "USB": "ยูเอสบี",
    "UX": "ยูเอ็กซ์",
    "VTT": "วีทีที",
    "WAV": "เวฟ",
    "YouTube": "ยูทูบ",
    "Python": "ไพธอน",
    "FFmpeg": "เอฟเอฟเอ็มเปก",
    "GitHub": "กิตฮับ",
    "Whisper": "วิสเปอร์",
    "Wi-Fi": "ไวไฟ",
}

_LETTER_PHONETIC = {
    "A": "เอ",
    "B": "บี",
    "C": "ซี",
    "D": "ดี",
    "E": "อี",
    "F": "เอฟ",
    "G": "จี",
    "H": "เอช",
    "I": "ไอ",
    "J": "เจ",
    "K": "เค",
    "L": "แอล",
    "M": "เอ็ม",
    "N": "เอ็น",
    "O": "โอ",
    "P": "พี",
    "Q": "คิว",
    "R": "อาร์",
    "S": "เอส",
    "T": "ที",
    "U": "ยู",
    "V": "วี",
    "W": "ดับเบิลยู",
    "X": "เอ็กซ์",
    "Y": "วาย",
    "Z": "แซด",
}

_NOISE_BRACKET_RE = re.compile(
    r"\[(?:music|applause|laughter|silence|inaudible|sfx|sound|เพลง|เสียงปรบมือ|เสียงหัวเราะ)[^\]]*\]"
    r"|\((?:music|applause|laughter|silence|inaudible|sfx)\)",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MARKDOWN_RE = re.compile(r"(\*\*|__|\*|`|~~)")
_CURRENCY_RE = re.compile(
    r"(?P<sym>[$฿€£¥])\s*(?P<num>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<scale>[KkMmBbTt])?(?![A-Za-z0-9]|[.,]\d)"
)
_PERCENT_RE = re.compile(r"(?P<num>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*%")
_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<num>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>kbps|mbps|gbps|kHz|MHz|GHz|Hz|TB|GB|MB|KB|fps|ms|km|cm|mm|kg|mg|x|K)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<num>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)(?![A-Za-z0-9]|[.,]\d)"
)
_KNOWN_TERMS_RE = re.compile(
    r"(?<![A-Za-z0-9])("
    + "|".join(re.escape(k) for k in sorted(_ACRONYM_MAP.keys(), key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)
_UNKNOWN_CAPS_RE = re.compile(r"(?<![A-Za-z])([A-Z]{2,5})(?![A-Za-z])")
_MULTI_PUNCT_RE = re.compile(r"([!?.,])\1+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_thai_for_tts(text: str) -> str:
    """Deterministic Thai TTS pre-processing converting numbers, currencies, units, acronyms, and symbols."""
    if not text:
        return ""

    s = html.unescape(text)
    s = _HTML_TAG_RE.sub(" ", s)
    s = _NOISE_BRACKET_RE.sub(" ", s)
    s = s.replace("♪", " ").replace("♫", " ")
    s = _MARKDOWN_RE.sub("", s)

    # 1. Known tech terms/acronyms containing digits (e.g., MP4, MP3) before number replacement
    def _replace_known_term(m: re.Match[str]) -> str:
        return _ACRONYM_MAP[m.group(1)]

    s = _KNOWN_TERMS_RE.sub(_replace_known_term, s)

    # 2. Currencies ($100, $1.5M, ฿500, €20, £10, ¥1000)
    def _replace_currency(m: re.Match[str]) -> str:
        sym = m.group("sym")
        num_words = number_to_thai_words(m.group("num"))
        scale = m.group("scale")
        scale_word = _CURRENCY_SCALE[scale.lower()] if scale else ""
        return f"{num_words}{scale_word}{_CURRENCY_NAMES[sym]}"

    s = _CURRENCY_RE.sub(_replace_currency, s)

    # 3. Percentages (25%, 3.5%)
    def _replace_percent(m: re.Match[str]) -> str:
        return f"{number_to_thai_words(m.group('num'))}เปอร์เซ็นต์"

    s = _PERCENT_RE.sub(_replace_percent, s)

    # 4. Units & rates (16kHz, 100GB, 60fps, 50ms, 10km, 1.25x, 4K)
    def _replace_unit(m: re.Match[str]) -> str:
        unit_raw = m.group("unit")
        # Require uppercase K for 4K/8K, allow lowercase x for 1.25x
        if unit_raw == "k":
            return m.group(0)
        unit_word = _UNIT_NAMES[unit_raw.lower()]
        return f"{number_to_thai_words(m.group('num'))}{unit_word}"

    s = _UNIT_RE.sub(_replace_unit, s)

    # 5. Remaining decimals & integers (including comma-formatted and negative)
    def _replace_number(m: re.Match[str]) -> str:
        return number_to_thai_words(m.group("num"))

    s = _NUMBER_RE.sub(_replace_number, s)

    # 6. Unknown 2-5 char ALL-CAPS English acronyms -> Thai phonetic letter-by-letter
    def _replace_unknown_caps(m: re.Match[str]) -> str:
        acronym = m.group(1)
        if acronym in _ACRONYM_MAP:
            return _ACRONYM_MAP[acronym]
        return "".join(_LETTER_PHONETIC[ch] for ch in acronym)

    s = _UNKNOWN_CAPS_RE.sub(_replace_unknown_caps, s)

    # 7. Symbols
    s = s.replace("&", " และ ")
    s = s.replace("+", " บวก ")
    s = s.replace("=", " เท่ากับ ")
    s = s.replace("@", " แอด ")
    s = s.replace("#", " แฮชแท็ก ")
    s = s.replace("%", " เปอร์เซ็นต์ ")

    # 8. Repeated punctuation & whitespace cleanup
    s = _MULTI_PUNCT_RE.sub(r"\1", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def run_translate(
    paths: WorkspacePaths,
    translator: Translator,
    *,
    context_window: int = 2,
    chars_per_sec: float = 14.0,
) -> list[SegmentTHDict]:
    """Execute Stage 2: translate segments_en.json -> normalized segments_th.json atomically."""
    if not is_valid_json_file(paths.segments_en, expected_type=list, allow_empty=False):
        raise StageError("translate", f"Missing or invalid segments_en.json: {paths.segments_en}")

    segments_en = read_json(paths.segments_en)
    texts_en = [str(seg["text_en"]).strip() for seg in segments_en]
    segments_th: list[SegmentTHDict] = []

    try:
        for idx, seg in enumerate(segments_en):
            text_en = texts_en[idx]
            start = float(seg["start"])
            end = float(seg["end"])
            slot_dur = max(0.1, end - start)
            max_chars = estimate_max_thai_chars(slot_dur, chars_per_sec=chars_per_sec)

            prev_start = max(0, idx - context_window)
            prev_ctx = texts_en[prev_start:idx]
            next_ctx = texts_en[idx + 1 : idx + 1 + context_window]

            raw_th = translator.translate_segment(
                text_en=text_en,
                prev_context=prev_ctx,
                next_context=next_ctx,
                slot_duration=slot_dur,
                max_chars=max_chars,
            )
            norm_th = normalize_thai_for_tts(raw_th)
            if not norm_th:
                raise ProviderError(
                    f"Segment {seg['id']} produced empty Thai text after normalization."
                )

            segments_th.append(
                {
                    "id": int(seg["id"]),
                    "start": start,
                    "end": end,
                    "text_en": text_en,
                    "text_th": norm_th,
                    "tts": {
                        "status": "pending",
                        "file": None,
                        "duration": None,
                        "tempo_used": None,
                    },
                }
            )

        atomic_write_json(paths.segments_th, segments_th)
        return segments_th
    except StageError:
        raise
    except Exception as exc:
        raise StageError("translate", str(exc), cause=exc) from exc
