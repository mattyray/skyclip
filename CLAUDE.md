# SkyClip

Mac desktop app for drone footage analysis and highlight extraction.

## Quick Context

**What it does:** Ingests DJI drone footage, automatically finds the best moments using telemetry + CV, exports as clips/Premiere projects/auto-generated videos.

**Target drones:** Mavic 3 series, Air 3, Mini 4 Pro, Mini 3 Pro (all generate SRT telemetry files)

**Tech stack:** Tauri 2.0 (Rust) + React/TypeScript frontend + Python sidecar for CV

---

## Current Phase: Phase 1 - Foundation

### Phase 1 Goals
1. Tauri project scaffolded with React frontend
2. SRT telemetry parser working in Rust
3. SQLite schema implemented
4. FFmpeg wrapper with VideoToolbox acceleration
5. Proxy generation (with LRF shortcut)
6. Thumbnail extraction

### Phase 1 Deliverable
Import a folder of DJI footage → parse SRT files → generate proxies/thumbnails → store in SQLite

---

## Critical Architecture Decisions

### 1. Rust/Python Split
**Rust handles (fast, lightweight):**
- File system operations
- SRT text file parsing
- SQLite database read/write
- Profile score calculations
- FFmpeg subprocess orchestration
- Premiere XML generation

**Python sidecar handles (heavy, on-demand):**
- OpenCV motion analysis
- Scene change detection
- (Future) CLIP embeddings, YOLO

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
├── profiles/            # JSON profile configs
└── docs/
    └── SPEC.md          # Full product spec
```

---

## Conventions

- **Rust:** Use `sqlx` for async SQLite, `serde` for JSON
- **Frontend:** React 18 + TypeScript + Tailwind + Zustand for state
- **IPC:** Tauri commands return `Result<T, String>`
- **File paths:** Always use absolute paths in database
- **UUIDs:** Use `uuid` crate for all IDs
- **Errors:** Log with context, surface user-friendly messages

---

## Phase 1 Tasks

1. [ ] `cargo create-tauri-app skyclip` with React template
2. [ ] Implement SRT parser in `src-tauri/src/services/srt_parser.rs`
3. [ ] Create SQLite schema with migrations
4. [ ] FFmpeg wrapper with VideoToolbox flags
5. [ ] LRF detection and validation logic
6. [ ] Proxy generation (LRF copy or FFmpeg fallback)
7. [ ] Thumbnail extraction at 1fps
8. [ ] Basic ingest command that populates database

---

## Reference Docs

- **Full product spec:** `docs/SPEC.md`
- **Tauri docs:** https://tauri.app/v2/
- **sqlx:** https://github.com/launchbadge/sqlx
- **FFmpeg filters:** https://ffmpeg.org/ffmpeg-filters.html
