# SkyClip

Mac desktop app for drone footage analysis and highlight extraction.

## Quick Context

**What it does:** Ingests DJI drone footage, automatically finds the best moments using telemetry + CV, exports as clips or auto-generated highlight reels with content-aware editing.

**Target drones:** Mavic 3 series, Air 3, Mini 4 Pro, Mini 3 Pro (all generate SRT telemetry files)

**Tech stack:** Tauri 2.0 (Rust) + React/TypeScript frontend + Python sidecar for CV

---

## Current Status

### Completed
- **Phase 1 (Foundation):** Ingest, SRT parsing, proxies, thumbnails, SQLite
- **Phase 2 (Analysis):** Telemetry signals, segment detection, profiles, scoring
- **Basic UI:** Folder picker, profile tabs, thumbnail grid, video preview, segment export

### Current Phase: Phase 3 - Visual Analysis & Content-Aware Editing

---

## Phase 3: Visual Analysis & Smart Editing

### Overview
Add Python-based visual analysis to improve segment detection and enable intelligent auto-editing when generating highlight reels.

### Python Sidecar Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  PYTHON SIDECAR                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  analyze_clip(proxy_path) → VisualAnalysis               │
│    ├── motion_vectors: [(frame, magnitude, direction)]  │
│    ├── scene_changes: [frame_numbers]                   │
│    ├── dominant_colors: [(frame, [r,g,b])]              │
│    ├── objects: [(frame, [detections])]                 │
│    └── embedding: [768-dim CLIP vector]                 │
│                                                          │
│  suggest_edit_points(clip_a, clip_b) → EditSuggestion   │
│    ├── transition_type: "cut" | "dissolve" | "dip"      │
│    ├── a_out_frame: optimal exit frame in clip A        │
│    ├── b_in_frame: optimal entry frame in clip B        │
│    └── confidence: 0.0-1.0                              │
│                                                          │
│  generate_edit_sequence(clips, style) → EditDecisionList│
│    - Takes ordered clips + style profile                │
│    - Returns frame-accurate edit decisions              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Visual Analysis Features

**Motion Analysis (OpenCV optical flow):**
- Frame-to-frame motion magnitude and direction
- Distinguish camera movement vs. subject movement
- Detect "action peaks" - moments of maximum movement

**Scene Content:**
- Scene change detection (hard cuts, dissolves in source)
- Dominant colors per segment (for color-matched transitions)
- Horizon detection (is shot level/tilted)
- Sky vs. ground ratio (useful for reveal shots)

**Object Detection (YOLO):**
- People, vehicles, boats, buildings
- Track subjects entering/exiting frame
- Detect "subject centered" moments vs. empty landscape

**Semantic Understanding (CLIP embeddings):**
- "Beach sunset", "mountain flyover", "urban downtown"
- Group similar scenes for cohesive edits
- Enable text search: "find all waterfall shots"

### Content-Aware Editing Rules

When generating highlight reels, apply intelligent editing:

**1. Cut on Motion**
- If clip A ends with rightward pan, cut (don't dissolve) to clip B with similar motion
- Match motion direction/speed at cut points
- Cut during movement, not static moments

**2. Scene Similarity Transitions**
- Similar colors/content → hard cut (feels continuous)
- Very different scenes → crossfade or dip-to-black
- Same location, different angle → match cut

**3. Pacing Rules**
- Fast action clips → shorter, quick cuts
- Cinematic/slow clips → longer holds, slower transitions
- Build energy: start slow, peak in middle, resolve at end

**4. Subject Continuity**
- If subject exits frame right in clip A, find clip B where something enters from left
- Avoid jump cuts on static subjects

**5. Clip Reordering (Advanced)**
- AI can suggest reordering clips for better visual flow
- Group by location/color/content type
- Create narrative arc (establishing → action → resolution)

### User Control Model

"Let me tweak" approach:
1. AI analyzes footage and generates initial edit
2. User sees proposed timeline with edit decisions
3. User can:
   - Accept individual suggestions
   - Override transition types
   - Reorder clips manually
   - Adjust in/out points
   - Regenerate with different style
4. Final render uses user's approved decisions

### Database Schema Additions

```sql
-- Visual analysis results (computed by Python sidecar)
ALTER TABLE segments ADD COLUMN visual_motion_avg REAL;
ALTER TABLE segments ADD COLUMN visual_motion_direction REAL;  -- degrees
ALTER TABLE segments ADD COLUMN dominant_color_r INTEGER;
ALTER TABLE segments ADD COLUMN dominant_color_g INTEGER;
ALTER TABLE segments ADD COLUMN dominant_color_b INTEGER;
ALTER TABLE segments ADD COLUMN has_subject BOOLEAN;
ALTER TABLE segments ADD COLUMN subject_type TEXT;  -- "person", "vehicle", etc.
ALTER TABLE segments ADD COLUMN clip_embedding BLOB;  -- CLIP vector for search

-- Edit decisions for highlight reels
CREATE TABLE edit_sequences (
    id TEXT PRIMARY KEY,
    flight_id TEXT REFERENCES flights(id),
    name TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    style TEXT  -- "cinematic", "action", "social"
);

CREATE TABLE edit_decisions (
    id TEXT PRIMARY KEY,
    sequence_id TEXT REFERENCES edit_sequences(id),
    segment_id TEXT REFERENCES segments(id),
    sequence_order INTEGER,

    -- Timing adjustments
    adjusted_start_ms INTEGER,
    adjusted_end_ms INTEGER,

    -- Transition to next clip
    transition_type TEXT,  -- "cut", "dissolve", "dip_black", "wipe"
    transition_duration_ms INTEGER,

    -- AI confidence (user can override)
    ai_suggested BOOLEAN DEFAULT TRUE,
    user_approved BOOLEAN DEFAULT FALSE
);
```

### Phase 3 Tasks

1. [ ] Set up Python sidecar with Tauri
2. [ ] Implement OpenCV motion analysis
3. [ ] Add scene change detection
4. [ ] Extract dominant colors per segment
5. [ ] Integrate YOLO for object detection
6. [ ] Add CLIP embeddings for semantic search
7. [ ] Build edit suggestion engine
8. [ ] Create highlight reel generator UI
9. [ ] Implement timeline preview with edit decisions
10. [ ] Add user override controls
11. [ ] FFmpeg concat with transitions

### Future: Music Integration
- User provides audio track
- Detect beats/tempo
- Align cuts to strong beats
- Match motion peaks to drops

---

## Critical Architecture Decisions

### 1. Rust/Python Split
**Rust handles (fast, lightweight):**
- File system operations
- SRT text file parsing
- SQLite database read/write
- Profile score calculations
- FFmpeg subprocess orchestration

**Python sidecar handles (heavy, on-demand):**
- OpenCV motion analysis
- Scene change detection
- CLIP embeddings
- YOLO object detection
- Edit suggestion algorithm

### 2. LRF Proxy Shortcut
DJI drones generate `.LRF` files (720p H.264 proxies) alongside 4K source files.

**Use LRF when available:**
- Rename `.LRF` → `.mp4` and use as proxy
- Saves ~15 minutes per hour of footage
- ~90% of target drones support this

**Fallback to FFmpeg when:**
- LRF doesn't exist (Mini 3 non-Pro, Air 3S)
- LRF is corrupted or duration mismatch
- Required for ~10% of users

**LRF Validation:**
```
1. Check if matching .LRF exists in LRF/ folder
2. Validate: file exists, non-zero, playable (FFprobe)
3. Check duration matches source (±1 frame tolerance)
4. If any check fails → generate via FFmpeg
```

**LRF Caveats:**
- Capped at 30fps regardless of source (60fps/120fps → 30fps LRF)
- Fine for preview/scrubbing, NOT for NLE proxy workflows
- DJI Air 3 has 50/50 chance LRF is 1 frame shorter

### 3. FFmpeg Hardware Acceleration
**MANDATORY for all FFmpeg operations:**
```bash
-hwaccel videotoolbox
```

Without this, 4K HEVC decoding melts the CPU.

**Proxy generation command:**
```bash
ffmpeg -hwaccel videotoolbox \
  -i source.mp4 \
  -vf "scale=1280:720" \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac -b:a 128k \
  output_proxy.mp4
```

### 4. FFmpeg Stream Copy for Export
For "Quick Clips" export, use `-c copy` for instant lossless cutting:

```bash
ffmpeg -ss 00:01:30 -to 00:02:00 -i input.mp4 \
  -c copy -avoid_negative_ts make_zero \
  output.mp4
```

**Tradeoff:** Cuts snap to keyframes (±1-2 seconds). Offer "Precise" mode with re-encode for frame-exact cuts.

### 5. Analyze in Place
- **Never copy** 4K source files to Mac
- Source stays on SD card/external drive
- Only proxies, thumbnails, metadata stored locally
- 100GB footage → ~2-3GB local cache

---

## DJI File Structure

```
/Volumes/DJI_MAVIC_3/
├── DCIM/
│   └── 100MEDIA/
│       ├── DJI_0001.MP4    # 4K source
│       ├── DJI_0001.SRT    # Telemetry
│       └── ...
└── LRF/                    # Low-res proxies (separate folder!)
    └── 100MEDIA/
        ├── DJI_0001.LRF
        └── ...
```

**Note:** LRF files are in a separate `LRF/` folder, not alongside MP4s.

---

## SRT Telemetry Format

```
1
00:00:00,000 --> 00:00:01,000
<font size="28">SrtCnt : 1, DiffTime : 1000ms
2025-12-23 14:32:15.123
[iso : 100] [shutter : 1/500.0] [fnum : 280] [ev : 0]
[ct : 5500] [color_md : default] [focal_len : 24.00]
[latitude : 40.7128] [longitude : -74.0060] [altitude: 150.0]
[gb_yaw : 45.2] [gb_pitch : -15.3] [gb_roll : 0.1]</font>
```

**Key fields for scoring:**
- `gb_pitch`, `gb_yaw` → gimbal movement (intentional camera moves)
- `latitude`, `longitude`, `altitude` → GPS movement/speed
- `iso` → quality indicator (high ISO = grainy)
- `shutter` → motion blur risk

**70%+ of scoring comes from SRT parsing alone** - no video decoding needed.

---

## SQLite Schema (Core Tables)

```sql
CREATE TABLE flights (
    id TEXT PRIMARY KEY,
    name TEXT,
    import_date DATETIME,
    source_path TEXT,
    location_name TEXT,
    gps_center_lat REAL,
    gps_center_lon REAL
);

CREATE TABLE source_clips (
    id TEXT PRIMARY KEY,
    flight_id TEXT REFERENCES flights(id),
    filename TEXT,
    source_path TEXT,
    proxy_path TEXT,
    srt_path TEXT,
    duration_sec REAL,
    resolution_width INTEGER,
    resolution_height INTEGER,
    framerate REAL
);

CREATE TABLE segments (
    id TEXT PRIMARY KEY,
    source_clip_id TEXT REFERENCES source_clips(id),
    start_time_ms INTEGER,
    end_time_ms INTEGER,
    thumbnail_path TEXT,

    -- Raw signals (computed once)
    motion_magnitude REAL,
    gimbal_pitch_delta_avg REAL,
    gimbal_yaw_delta_avg REAL,
    gimbal_smoothness REAL,
    altitude_delta REAL,
    gps_speed_avg REAL,
    iso_avg REAL,
    visual_quality REAL,

    -- User state
    is_selected BOOLEAN DEFAULT FALSE
);
```

---

## Project Structure

```
skyclip/
├── src-tauri/           # Rust backend
│   └── src/
│       ├── main.rs
│       ├── commands/    # Tauri IPC commands
│       ├── services/    # srt_parser, ffmpeg, database, scoring
│       └── models/      # flight, clip, segment, profile
├── src/                 # React frontend
│   ├── components/
│   ├── hooks/
│   └── stores/
├── python/              # Python sidecar
│   └── skyclip_analyzer/
│       ├── __init__.py
│       ├── motion.py    # OpenCV optical flow
│       ├── scene.py     # Scene change detection
│       ├── color.py     # Dominant color extraction
│       ├── objects.py   # YOLO detection
│       ├── semantic.py  # CLIP embeddings
│       └── editor.py    # Edit suggestion engine
├── profiles/            # JSON profile configs
└── docs/
    └── SPEC.md          # Full product spec
```

---

## Conventions

- **Rust:** Use `sqlx` for async SQLite, `serde` for JSON
- **Frontend:** React 18 + TypeScript + Tailwind + Zustand for state
- **Python:** Python 3.11+, opencv-python, ultralytics (YOLO), transformers (CLIP)
- **IPC:** Tauri commands return `Result<T, String>`
- **File paths:** Always use absolute paths in database
- **UUIDs:** Use `uuid` crate for all IDs
- **Errors:** Log with context, surface user-friendly messages

---

## Reference Docs

- **Full product spec:** `docs/SPEC.md`
- **Tauri docs:** https://tauri.app/v2/
- **sqlx:** https://github.com/launchbadge/sqlx
- **FFmpeg filters:** https://ffmpeg.org/ffmpeg-filters.html
- **OpenCV optical flow:** https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
- **Ultralytics YOLO:** https://docs.ultralytics.com/
- **CLIP:** https://github.com/openai/CLIP
