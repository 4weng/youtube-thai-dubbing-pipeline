# Project Specification: YouTube English-to-Thai Dubbing Pipeline

## 1. Project Overview
ระบบแปลงและพากย์เสียงภาษาไทยทับวิดีโอ YouTube ภาษาอังกฤษแบบอัตโนมัติ (Offline Batch Pipeline) ตั้งแต่รับ URL วิดีโอ ดึงข้อความ/เวลา แปลภาษา สังเคราะห์เสียงพูดภาษาไทย (Thai TTS) ปรับจังหวะให้สอดคล้องกับช่วงเวลาเดิม และรวมแทร็กเสียงเข้ากับวิดีโอด้วย `ffmpeg` พร้อมระบบจัดเก็บสถานะ (Checkpoint & Resume) เพื่อรองรับคลิปความยาวมาก

### Core Goals
- รับลิงก์ YouTube ภาษาอังกฤษ และสร้างไฟล์วิดีโอผลลัพธ์ที่มีเสียงพากย์ภาษาไทย (`output_th.mp4`)
- รองรับทั้งโหมด **Audio Ducking** (ลดเสียงต้นฉบับและเปิดเสียงไทยทับ) และ **Replace Audio** (แทนที่เสียงต้นฉบับทั้งหมด)
- ป้องกันเสียงพากย์ล้นหรือซ้อนทับประโยคถัดไป ด้วยการคุมความยาวบทแปล การตัดความเงียบ และการปรับความเร็วเสียงด้วย `atempo`
- รองรับ **Checkpoint & Resume** สามารถรันต่อจากท่อนที่ค้างอยู่ได้ทันทีหากเกิด Error กลางทาง

---

## 2. System Architecture (4-Stage Pipeline)

```text
[YouTube URL (EN)]
       │
       ▼
┌──────────────────────────────────────────┐
│ Stage 1: Ingestion & ASR (EN)            │
│ • Download video/audio (yt-dlp)          │
│ • Extract EN subtitles or run Whisper    │
│ • Output: source.mp4, segments_en.json   │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│ Stage 2: Translation & Length Fitting    │
│ • Context-aware EN -> TH translation     │
│ • Text normalization (numbers/acronyms)  │
│ • Output: segments_th.json               │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│ Stage 3: Thai TTS & Timing Alignment     │
│ • Generate per-segment .wav (Thai TTS)   │
│ • Trim leading/trailing silence          │
│ • Measure duration vs slot (end - start) │
│ • Apply bounded atempo (<= 1.25x)        │
│ • Output: tts_segments/seg_XXXX.wav      │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│ Stage 4: Timeline Mix & Mux (ffmpeg)     │
│ • Place segments on timeline by start_ms │
│ • Render full track: thai_dub_full.wav   │
│ • Mix/Duck original audio + video copy   │
│ • Output: output_th.mp4                   │
└──────────────────────────────────────────┘
```

---

## 3. Directory & Workspace Structure

แยกโฟลเดอร์ทำงานตาม `video_id` เพื่อป้องกันไฟล์ชนกันและจัดการพื้นที่ง่าย:

```text
project_root/
├── pipeline.py              # CLI Entrypoint & Stage Orchestrator
├── src/
│   ├── __init__.py
│   ├── state.py             # Atomic file writes, manifest & integrity checks
│   ├── ingest.py            # Stage 1: yt-dlp + Whisper / subtitle parser
│   ├── translate.py         # Stage 2: EN -> TH translation & text normalization
│   ├── tts_align.py         # Stage 3: Thai TTS generation, trimming & atempo sync
│   └── mux.py               # Stage 4: Audio timeline assembly & ffmpeg muxing
├── requirements.txt
└── work/
    └── <video_id>/
        ├── manifest.json
        ├── source.mp4
        ├── audio_en.wav
        ├── segments_en.json
        ├── segments_th.json
        ├── tts_segments/
        │   ├── seg_0001.wav
        │   └── ...
        ├── thai_dub_full.wav
        └── output_th.mp4
```

---

## 4. Data Contracts (Intermediate Files)

### 4.1 `manifest.json` (Pipeline State)
```json
{
  "video_id": "5orlc92154E",
  "source_url": "https://youtu.be/5orlc92154E",
  "config": {
    "mix_mode": "duck",
    "original_volume": 0.18,
    "max_atempo": 1.25,
    "tts_provider": "pythaitts"
  },
  "stages": {
    "ingest": { "status": "done" },
    "translate": { "status": "done" },
    "tts_align": { "status": "in_progress" },
    "mux": { "status": "pending" }
  },
  "last_error": null
}
```

### 4.2 `segments_en.json`
```json
[
  {
    "id": 1,
    "start": 1.20,
    "end": 4.85,
    "text_en": "Today we build a local voice pipeline."
  }
]
```

### 4.3 `segments_th.json`
```json
[
  {
    "id": 1,
    "start": 1.20,
    "end": 4.85,
    "text_en": "Today we build a local voice pipeline.",
    "text_th": "วันนี้เราจะมาสร้างไปป์ไลน์เสียงแบบโลคัลกัน",
    "tts": {
      "status": "done",
      "file": "tts_segments/seg_0001.wav",
      "duration": 3.42,
      "tempo_used": 1.0
    }
  }
]
```

---

## 5. Technical Specifications by Stage

### Stage 1: Ingestion & ASR (`src/ingest.py`)
- ใช้ `yt-dlp` ดาวน์โหลดวิดีโอที่ดีที่สุดเป็น `source.mp4` และแยกเสียงเป็น `audio_en.wav` (16kHz mono สำหรับ ASR)
- แบ่งช่วงพูด (Segments) เป็นท่อนสั้นๆ ระดับประโยค (แนะนำ 3–8 วินาทีต่อท่อน) พร้อม `start` และ `end` (หน่วยวินาที)
- หากคลิปไม่มีซับอังกฤษที่แม่นยำ ให้ใช้โมเดลสาย Whisper ถอดข้อความพร้อม Timestamp

### Stage 2: Translation & Text Normalization (`src/translate.py`)
- แปล `text_en` เป็น `text_th` โดยส่งบริบทรอบข้าง (Context window) เพื่อให้ภาษาเป็นธรรมชาติ
- **Length Constraint:** กำหนดให้คำแปลภาษาไทยกระชับ สอดคล้องกับเวลาพูด `slot_dur = end - start`
- **Thai TTS Pre-processing:** แปลงตัวเลข สัญลักษณ์ และคำย่อให้เป็นคำอ่านภาษาไทยก่อนส่งเข้า TTS เพื่อป้องกันการอ่านข้ามหรือเพี้ยน

### Stage 3: Thai TTS & Timing Alignment (`src/tts_align.py`)
- **Provider Abstraction:** ออกแบบ Interface ให้สลับตัวสร้างเสียงได้ง่าย (เช่น `PyThaiTTS`, โมเดลบน Hugging Face หรือ Cloud TTS API) โดยล็อก Voice/Speaker ID เดิมทั้งคลิป
- **Alignment Logic per Segment:**
  1. สร้างเสียงดิบและตัดความเงียบหัว-ท้าย (Trim silence)
  2. คำนวณ `slot_dur = end - start` และ `actual_dur`
  3. หาก `actual_dur <= slot_dur`: ใช้งานไฟล์เสียงที่ความเร็ว `1.0`
  4. หาก `actual_dur > slot_dur`: คำนวณ `needed_tempo = actual_dur / slot_dur`
     - จำกัดเพดาน `tempo = min(needed_tempo, max_atempo)` (เช่น ไม่เกิน `1.25` เพื่อไม่ให้เสียงเพี้ยน)
     - รัน `ffmpeg -i raw.wav -filter:a "atempo={tempo}" aligned.wav`
  5. บันทึกสถานะของท่อนนั้นลง `segments_th.json` ทันที (Incremental Save)

### Stage 4: Timeline Mix & Mux (`src/mux.py`)
1. **Timeline Assembly:** นำ `seg_XXXX.wav` ทุกท่อนมาวางตามเวลา `start` (แทรกความเงียบตามช่วงเวลาที่เว้นว่าง) รวมเป็นแทร็กเดียว `thai_dub_full.wav`
2. **Video Muxing Commands:**
   - **โหมด Ducking (ลดเสียงต้นฉบับเหลือ 18% และผสมเสียงพากย์ไทย):**
     ```bash
     ffmpeg -y -i source.mp4 -i thai_dub_full.wav \
       -filter_complex "[0:a]volume=0.18[orig];[1:a]volume=1.0[th];[orig][th]amix=inputs=2:duration=first:dropout_transition=2[aout]" \
       -map 0:v -map "[aout]" -c:v copy -c:a aac -b:a 192k output_th.mp4
     ```
   - **โหมด Replace (แทนที่เสียงต้นฉบับทั้งหมด):**
     ```bash
     ffmpeg -y -i source.mp4 -i thai_dub_full.wav \
       -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k output_th.mp4
     ```

---

## 6. Reliability & Checkpoint/Resume Rules (`src/state.py`)

1. **Atomic File Writes:** เขียนไฟล์ JSON และไฟล์เสียง/วิดีโอลงไฟล์ชั่วคราว (`.tmp`) ก่อนเสมอ เมื่อเขียนเสร็จและปิดไฟล์แล้วจึงเรียก `os.replace()` เพื่อป้องกันไฟล์เสียกลางทาง
2. **Verify Before Skip:** ก่อนข้ามขั้นตอนหรือท่อนใดๆ ต้องตรวจสอบว่า:
   - ไฟล์มีอยู่จริงและขนาด `> 0` bytes
   - สำหรับไฟล์เสียง/วิดีโอ ต้องตรวจด้วย `ffprobe` หรือ `soundfile` ว่าอ่าน Metadata ได้และ `duration > 0`
3. **CLI Flags:**
   - `--url <youtube_url>`: รันไปป์ไลน์ทั้งหมด (ข้ามขั้นตอน/ท่อนที่สมบูรณ์แล้วอัตโนมัติ)
   - `--mix-mode <duck|replace>`: เลือกรูปแบบการผสมเสียง
   - `--force-stage <ingest|translate|tts_align|mux>`: บังคับรันใหม่เฉพาะ Stage ที่ระบุ (เช่น หลังแก้ไขบทแปลใน `segments_th.json` ด้วยมือ)

---

## 7. Implementation Checklist for AI Coding Agent

1. สร้างโครงสร้างโปรเจกต์ตาม Section 3 และไฟล์ `requirements.txt` (`yt-dlp`, `soundfile`, `pydub` หรือ `ffmpeg-python`, และไลบรารี ASR/TTS ที่เลือก)
2. เขียน `src/state.py` สำหรับจัดการ `manifest.json`, `atomic_write_json()`, และ `is_valid_audio_file()`
3. เขียน `src/ingest.py` ดึงวิดีโอ/เสียง และสร้าง `segments_en.json`
4. เขียน `src/translate.py` แปลบทพร้อมคุมความยาวและจัดรูปแบบคำอ่านภาษาไทย
5. เขียน `src/tts_align.py` สร้างเสียงรายท่อน ตัดความเงียบ ปรับ `atempo` และบันทึกสถานะทีละท่อน
6. เขียน `src/mux.py` ประกอบไทม์ไลน์เสียงไทยและรวมกับ `source.mp4` ด้วย `ffmpeg`
7. เขียน `pipeline.py` เชื่อมทั้ง 4 Stage พร้อม Argument Parser และ Error Handling