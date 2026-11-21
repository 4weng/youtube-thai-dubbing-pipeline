"""Unit tests for Stage 2 src/translate.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.state import atomic_write_json, get_workspace_paths, read_json
from src.translate import (
    DeepTranslatorProvider,
    OpenAICompatibleTranslator,
    create_translator,
    estimate_max_thai_chars,
    int_to_thai_words,
    normalize_thai_for_tts,
    number_to_thai_words,
    run_translate,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "ศูนย์"),
        (1, "หนึ่ง"),
        (10, "สิบ"),
        (11, "สิบเอ็ด"),
        (20, "ยี่สิบ"),
        (21, "ยี่สิบเอ็ด"),
        (101, "หนึ่งร้อยเอ็ด"),
        (125, "หนึ่งร้อยยี่สิบห้า"),
        (2026, "สองพันยี่สิบหก"),
        (1_000_000, "หนึ่งล้าน"),
        (-42, "ลบสี่สิบสอง"),
    ],
)
def test_int_to_thai_words(value: int, expected: str) -> None:
    assert int_to_thai_words(value) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,250", "หนึ่งพันสองร้อยห้าสิบ"),
        ("3.14", "สามจุดหนึ่งสี่"),
        ("-0.5", "ลบศูนย์จุดห้า"),
        ("18", "สิบแปด"),
    ],
)
def test_number_to_thai_words(raw: str, expected: str) -> None:
    assert number_to_thai_words(raw) == expected


def test_normalize_thai_for_tts_comprehensive() -> None:
    assert "หนึ่งร้อยดอลลาร์" in normalize_thai_for_tts("ราคา $100")
    assert "หนึ่งจุดห้าล้านดอลลาร์" in normalize_thai_for_tts("ทุน $1.5M")
    assert "ห้าร้อยบาท" in normalize_thai_for_tts("จ่าย ฿500")
    assert "สิบแปดเปอร์เซ็นต์" in normalize_thai_for_tts("ลดเสียง 18%")
    assert "สามจุดห้าเปอร์เซ็นต์" in normalize_thai_for_tts("โต 3.5%")
    assert "สิบหกกิโลเฮิรตซ์" in normalize_thai_for_tts("เสียง 16kHz")
    assert "หกสิบเฟรมต่อวินาที" in normalize_thai_for_tts("วิดีโอ 60fps")
    assert "หนึ่งจุดสองห้าเท่า" in normalize_thai_for_tts("เร็ว 1.25x")
    assert "สี่เค" in normalize_thai_for_tts("ความละเอียด 4K")
    assert "เอไอ" in normalize_thai_for_tts("ใช้ AI สร้างเสียง")
    assert "เอพีไอ" in normalize_thai_for_tts("เรียก API")
    assert "เอ็มพีสี่" in normalize_thai_for_tts("ไฟล์ MP4")
    # Unknown ALL-CAPS acronym phonetic fallback (e.g. XYZ -> เอ็กซ์วายแซด)
    assert "เอ็กซ์วายแซด" in normalize_thai_for_tts("ระบบ XYZ")
    # Symbols and noise stripping
    cleaned = normalize_thai_for_tts("[Music] **A & B** = 10 + 5!!!")
    assert "[Music]" not in cleaned
    assert "*" not in cleaned
    assert "และ" in cleaned
    assert "เท่ากับ" in cleaned
    assert "บวก" in cleaned
    assert "สิบ" in cleaned and "ห้า" in cleaned


def test_estimate_max_thai_chars() -> None:
    assert estimate_max_thai_chars(4.0, chars_per_sec=14.0) == 56
    assert estimate_max_thai_chars(0.2, chars_per_sec=14.0, min_chars=12) == 12


def test_create_translator_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tr = create_translator("auto")
    assert isinstance(tr, DeepTranslatorProvider)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy-key")
    tr2 = create_translator("auto")
    assert isinstance(tr2, OpenAICompatibleTranslator)


def test_run_translate_passes_context_and_normalizes(tmp_path: Path) -> None:
    paths = get_workspace_paths(tmp_path, "vid00000010")
    paths.ensure_dirs()
    atomic_write_json(
        paths.segments_en,
        [
            {"id": 1, "start": 0.0, "end": 3.0, "text_en": "First sentence."},
            {"id": 2, "start": 3.0, "end": 6.5, "text_en": "Second sentence about AI 100%."},
            {"id": 3, "start": 6.5, "end": 10.0, "text_en": "Third sentence."},
        ],
    )

    recorded_calls: list[dict] = []

    class RecordingTranslator:
        def translate_segment(
            self,
            text_en: str,
            prev_context: list[str],
            next_context: list[str],
            slot_duration: float,
            max_chars: int,
        ) -> str:
            recorded_calls.append(
                {
                    "text_en": text_en,
                    "prev": list(prev_context),
                    "next": list(next_context),
                    "slot": slot_duration,
                    "max_chars": max_chars,
                }
            )
            if "AI" in text_en:
                return "ประโยคที่ 2 เกี่ยวกับ AI 100%"
            return f"แปล: {text_en}"

    segs_th = run_translate(paths, RecordingTranslator(), context_window=1)
    assert len(segs_th) == 3
    assert recorded_calls[0]["prev"] == []
    assert recorded_calls[0]["next"] == ["Second sentence about AI 100%."]
    assert recorded_calls[1]["prev"] == ["First sentence."]
    assert recorded_calls[1]["next"] == ["Third sentence."]
    assert recorded_calls[1]["slot"] == pytest.approx(3.5)
    assert recorded_calls[1]["max_chars"] == estimate_max_thai_chars(3.5)

    # Normalized Thai text in segment 2
    assert "สอง" in segs_th[1]["text_th"]
    assert "เอไอ" in segs_th[1]["text_th"]
    assert "หนึ่งร้อยเปอร์เซ็นต์" in segs_th[1]["text_th"]
    assert segs_th[1]["tts"]["status"] == "pending"
    assert read_json(paths.segments_th) == segs_th


def test_normalize_thai_for_tts_adjacent_thai_currency_and_units() -> None:
    assert normalize_thai_for_tts("ราคา$1.5Mครับ") == "ราคาหนึ่งจุดห้าล้านดอลลาร์ครับ"
    assert normalize_thai_for_tts("ราคา฿2,500ครับ") == "ราคาสองพันห้าร้อยบาทครับ"
    assert normalize_thai_for_tts("ความละเอียด4Kครับ") == "ความละเอียดสี่เคครับ"
    assert normalize_thai_for_tts("ความเร็ว1.25xครับ") == "ความเร็วหนึ่งจุดสองห้าเท่าครับ"
    assert normalize_thai_for_tts("เสียง16kHzครับ") == "เสียงสิบหกกิโลเฮิรตซ์ครับ"
    assert normalize_thai_for_tts("$100ล้าน") == "หนึ่งร้อยดอลลาร์ล้าน"

