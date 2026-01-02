# SkyClip: Complete Product Specification

**Version:** 1.0  
**Date:** December 26, 2025  
**Status:** Ready for Development

---

## Table of Contents

1. [Product Definition](#1-product-definition)
2. [User Experience](#2-user-experience)
3. [Technical Architecture](#3-technical-architecture)
4. [Scoring & Profiles](#4-scoring--profiles)
5. [Development Phases](#5-development-phases)
6. [Research Findings](#6-research-findings)
7. [Appendices](#7-appendices)

---

# 1. Product Definition

## 1.1 What We're Building

### One-Line Description


A Mac desktop application that automatically analyzes drone footage, identifies the best moments using telemetry data and computer vision, and exports highlights as individual clips, Premiere Pro projects, or auto-generated videos.

### Core Value Proposition
> "Plug in your drone. See your best shots through different creative lenses. Export exactly what you want."

### Tagline
> "Premiere is for editors. SkyClip is for pilots who want to post."

---

## 1.2 The Problem

### Pain Points (Validated Through Research)

| Pain Point | Evidence |
|------------|----------|
| Hours of footage, minutes of gold | Forum users: "logging footage is unfortunately something we can't avoid" |
| No memory of what's where | Files named DJI_0847.MP4 with no context |
| Manual scrubbing is tedious | Tutorials assume "starting with individual clips, not long shots" |
| Storage chaos | Users managing 4TB+ drives, still can't find old footage |
| LightCut limitations | Mobile-only, freezes on large files, ~1 minute max output |
| AI tools ignore visual content | OpusClip, Chopcast detect highlights via speech—useless for drone footage |

### What Exists vs. What's Missing

| What Exists | What's Missing |
|-------------|----------------|
| LightCut (mobile, quick edits) | Desktop app for full-quality 4K/5K |
| OpusClip/Chopcast (speech-based) | Visual/motion-based detection |
| Premiere/Resolve (manual) | Automated highlight finding |
| Generic AI video tools | Drone-specific intelligence (gimbal, telemetry) |
| Single "interesting" definition | Multiple creative profiles |

---

## 1.3 Target Users

### Tier 1: Content Creators (Primary)
- YouTube drone channels
- Travel vloggers using drones for B-roll
- Real estate content creators
- FPV pilots posting to social

**Problem:** "I have amazing footage but posting is a second job."  
**Price sensitivity:** $10-30/month or $100-200 one-time

### Tier 2: Prosumers
- Hobbyist pilots who want to share
- Vacation drone footage people

**Problem:** "I have 200GB from my trip and I've posted nothing."  
**Price sensitivity:** $50-100 one-time

### Tier 3: Commercial (Future)
- Real estate photography businesses
- Inspection companies

**Problem:** "We fly 20 properties a week and editing is our bottleneck."  
**Price sensitivity:** $50-100/month per seat

---

## 1.4 Supported Hardware

### Drones (Launch)

| Drone | LRF Support | SRT Support | Notes |
|-------|-------------|-------------|-------|
| Mavic 3 / 3 Pro / 3 Cine | ✅ | ✅ | Full support |
| Air 3 | ✅ | ✅ | LRF may be 1 frame short |
| Mini 4 Pro | ✅ | ✅ | Full support |
| Mini 3 Pro | ✅ | ✅ | Full support |
| Mini 3 (non-Pro) | ❌ | ✅ | Proxies generated via FFmpeg |
| Air 3S | ❌ | ✅ | Proxies generated via FFmpeg |

**Requirement:** Drone must generate `.SRT` telemetry files (default when Video Captions enabled)

### Computer Requirements
- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4) — required for hardware video acceleration
- 8GB RAM minimum, 16GB recommended
- 10GB free disk space for app + proxy cache

---

# 2. User Experience

## 2.1 Core Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER JOURNEY                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  INGEST  │ ─▶ │ ANALYZE  │ ─▶ │  REVIEW  │ ─▶ │  EXPORT  │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                      │
│  Connect drone   App scans      User browses    User exports        │
│  or insert SD    footage for    clips through   selections as       │
│  card            highlights     different       clips, Premiere     │
│                  using          creative        project, or         │
│                  telemetry +    profiles and    auto-generated      │
│                  CV analysis    builds          video               │
│                                 selections                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 Stage 1: Ingest

### Input Methods

| Method | How It Works |
|--------|--------------|
| USB Drone Connection | Drone mounts as mass storage via USB-C |
| SD Card | Insert in card reader, mounts as volume |
| Folder Selection | Drag & drop or browse to folder |

### What Happens During Ingest

1. Detect DJI folder structure (`DCIM/100MEDIA/`, `LRF/`)
2. Identify video files (`.MP4`, `.MOV`) and paired telemetry (`.SRT`)
3. Extract metadata (duration, resolution, framerate, timestamp)
4. Parse `.SRT` files for telemetry
5. **LRF Optimization:** Check for pre-generated proxies
   - If LRF exists and valid → copy/rename to proxy folder (~30 seconds)
   - If LRF missing/invalid → generate via FFmpeg (~15 minutes/hour)
6. Generate thumbnails (1 frame per second)
7. Store in SQLite database

### Critical Design: Analyze in Place
- **Never copy** large 4K source files
- Source stays on SD card/external drive
- Only proxies, thumbnails, metadata stored locally
- 100GB footage → ~2-3GB local cache

---

## 2.3 Stage 2: Analyze

### Analysis Pipeline

**Track A: Telemetry Analysis (Rust — Instant)**
- Parse `.SRT` file (text processing)
- Extract per-second: gimbal pitch/yaw/roll, GPS, altitude, ISO, shutter
- Calculate deltas (rate of change)
- Compute telemetry-based signals

**Track B: Visual Analysis (Python — On Proxy)**
- Motion magnitude via optical flow
- Scene change detection via frame differencing
- Visual quality assessment

### Raw Signals Computed

| Signal | Source | Description |
|--------|--------|-------------|
| `motion_magnitude` | OpenCV | Movement in frame (0-1) |
| `gimbal_pitch_delta` | SRT | Camera tilt rate |
| `gimbal_yaw_delta` | SRT | Camera pan rate |
| `gimbal_smoothness` | SRT | Inverse of jitter |
| `altitude_delta` | SRT | Ascending/descending rate |
| `gps_speed` | SRT | Horizontal movement speed |
| `iso_value` | SRT | Quality indicator |
| `visual_quality` | OpenCV | Sharpness, exposure |
| `duration` | File | Segment length |

**Key insight:** 70%+ of scoring from SRT parsing alone—no video decoding needed.

---

## 2.4 Stage 3: Review (Multi-Profile Selection)

### The Core Innovation

Users browse clips through **multiple creative lenses** and build a **unified selection** from across profiles. Switching profiles is instant (re-sort, no re-analysis).

### Profile Definitions

| Profile | Target User | Prioritizes | Deprioritizes |
|---------|-------------|-------------|---------------|
| **Cinematic** | Travel vloggers, wedding | Slow pans, smooth gimbal, long duration | Fast motion, jerky |
| **Action** | FPV pilots, sports | High motion, altitude changes, short bursts | Stability |
| **Precision** | Real estate, inspectors | Sharpness, -90° angles, hovering | Motion |
| **Social** | TikTok/Reels creators | Center-weighted, very short clips | Wide shots |
| **Discovery** | Everyone | Shows all clips (minimal filtering) | Nothing hidden |

### Selection Behavior

- Selections persist across profile switches
- Same clip may appear in multiple profiles (different scores)
- Clips can be reordered via drag-and-drop
- Selections are profile-agnostic (just clip IDs)

### Adjusting Clip Boundaries

Users can fine-tune in/out points for any selected clip.

---

## 2.5 Stage 4: Export

### Export Path A: Quick Clips

**User intent:** "Give me the selected clips as separate files"

**Output:**
```
~/Desktop/SkyClip Export Dec 23/
├── clip_001_sunset_pan.mp4
├── clip_002_diving_shot.mp4
└── export_metadata.json
```

**Options:**
- Format: MP4 (H.264 or H.265)
- Resolution: Original, 4K, 1080p
- Aspect Ratio: 16:9, 9:16, 1:1
- Handles: None, +0.5s, +1s, +2s
- Mode: Fast (stream copy, ±1-2s keyframe snap) or Precise (re-encode, frame-exact)

### Export Path B: Premiere Pro Handoff

**User intent:** "I want to edit these myself in Premiere"

**Output:**
```
~/Desktop/SkyClip Export Dec 23/
├── SkyClip_Project.xml    (FCP XML v4)
└── README.txt
```

**XML contains:**
- Sequence with selected clips on timeline
- In/out points matching user selections
- Markers at algorithm-detected highlights
- File paths to ORIGINAL source files (not proxies)

**Critical:** User must keep source media mounted when opening in Premiere.

### Export Path C: Auto Video

**User intent:** "Just make me something I can post"

**Output:**
```
~/Desktop/SkyClip Export Dec 23/
└── SkyClip_Highlight_Reel.mp4
```

**Features:**
- Compiled video of selected clips
- Automatic transitions (cross-dissolve default)
- Optional background music
- Platform-optimized encoding

---

## 2.6 Library

All imported footage persists in a searchable library:

- Flights grouped by date and GPS location
- Keyword search (Phase 2: semantic search via CLIP)
- Filter by date range, location
- Re-open any flight for review/export

---

# 3. Technical Architecture

## 3.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      SKYCLIP ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     DESKTOP APP (Tauri 2.0)                   │  │
│  │                                                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │              FRONTEND (React + TypeScript)              │  │  │
│  │  │  • Ingest UI    • Review UI    • Export UI    • Library │  │  │
│  │  └──────────────────────────┬──────────────────────────────┘  │  │
│  │                             │ Tauri Commands (IPC)            │  │
│  │  ┌──────────────────────────▼──────────────────────────────┐  │  │
│  │  │              RUST CORE (Tauri Main Process)             │  │  │
│  │  │  • File system ops       • SRT parsing                  │  │  │
│  │  │  • SQLite database       • FFmpeg orchestration         │  │  │
│  │  │  • Profile scoring       • Premiere XML generation      │  │  │
│  │  └──────────────────────────┬──────────────────────────────┘  │  │
│  │                             │ Sidecar IPC (when needed)       │  │
│  │  ┌──────────────────────────▼──────────────────────────────┐  │  │
│  │  │              PYTHON SIDECAR (Visual Analysis)           │  │  │
│  │  │  • Motion analysis (OpenCV)                             │  │  │
│  │  │  • Scene change detection                               │  │  │
│  │  │  • (Future) CLIP embeddings, YOLO                       │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     LOCAL FILE SYSTEM                         │  │
│  │                                                                │  │
│  │  ~/Library/Application Support/SkyClip/                       │  │
│  │  ├── library.db           (SQLite + sqlite-vss)               │  │
│  │  ├── proxies/             (720p preview files)                │  │
│  │  ├── thumbnails/          (JPEGs, 1 per second)               │  │
│  │  └── profiles/            (JSON weight configs)               │  │
│  │                                                                │  │
│  │  EXTERNAL (User's Media - Never Copied)                       │  │
│  │  /Volumes/DJI_SD/DCIM/100MEDIA/*.MP4                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3.2 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Desktop Framework | Tauri 2.0 | Lightweight, native, Rust backend |
| Frontend | React 18 + TypeScript | Developer expertise |
| Styling | Tailwind CSS | Rapid UI development |
| Backend Logic | Rust | Fast file I/O, text parsing |
| Visual Analysis | Python 3.11+ | OpenCV ecosystem |
| Database | SQLite + sqlite-vss | Self-contained, vector search |
| Video Processing | FFmpeg 6.x | Industry standard |

---

## 3.3 FFmpeg Configuration

### Hardware Acceleration (MANDATORY)

```bash
-hwaccel videotoolbox
```

**Performance impact:**
- Without: ~3 hours per hour of 4K HEVC
- With: ~15 minutes per hour

### Proxy Generation

```bash
ffmpeg -hwaccel videotoolbox \
  -i source.mp4 \
  -vf "scale=1280:720" \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac -b:a 128k \
  -movflags +faststart \
  output_proxy.mp4
```

### Stream Copy Export (Fast Mode)

```bash
ffmpeg -ss 00:01:30 -to 00:02:00 -i input.mp4 \
  -c copy -avoid_negative_ts make_zero \
  output.mp4
```

**Note:** Cuts snap to keyframes (±1-2 seconds). Use `-c:v libx264 -crf 18` for frame-exact.

---

## 3.4 Database Schema

```sql
CREATE TABLE flights (
    id TEXT PRIMARY KEY,
    name TEXT,
    import_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    source_path TEXT,
    location_name TEXT,
    gps_center_lat REAL,
    gps_center_lon REAL,
    total_duration_sec REAL,
    total_clips INTEGER
);

CREATE TABLE source_clips (
    id TEXT PRIMARY KEY,
    flight_id TEXT REFERENCES flights(id),
    filename TEXT,
    source_path TEXT,
    proxy_path TEXT,
    proxy_source TEXT,  -- 'lrf' or 'generated'
    srt_path TEXT,
    duration_sec REAL,
    resolution_width INTEGER,
    resolution_height INTEGER,
    framerate REAL,
    recorded_at DATETIME
);

CREATE TABLE segments (
    id TEXT PRIMARY KEY,
    source_clip_id TEXT REFERENCES source_clips(id),
    start_time_ms INTEGER,
    end_time_ms INTEGER,
    duration_ms INTEGER,
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
    has_scene_change BOOLEAN,
    
    -- User state
    is_selected BOOLEAN DEFAULT FALSE,
    user_adjusted_start_ms INTEGER,
    user_adjusted_end_ms INTEGER
);

CREATE TABLE selections (
    id TEXT PRIMARY KEY,
    flight_id TEXT REFERENCES flights(id),
    segment_id TEXT REFERENCES segments(id),
    sequence_order INTEGER,
    added_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

# 4. Scoring & Profiles

## 4.1 Score Calculation

Profiles apply different weight coefficients to raw signals:

```
Profile Score = Σ (weight_i × signal_i)
```

Score calculated on-demand when profile tab viewed. Same segment has different scores per profile.

## 4.2 Profile Weights

### Cinematic
```json
{
  "motion_magnitude": 0.3,
  "gimbal_smoothness": 2.0,
  "gimbal_pitch_delta": 0.5,
  "gimbal_yaw_delta": 0.5,
  "visual_quality": 1.5,
  "duration_bonus": 1.0,
  "rapid_rotation_penalty": -1.0,
  "min_duration_sec": 5.0
}
```

### Action
```json
{
  "motion_magnitude": 2.0,
  "gimbal_smoothness": 0.0,
  "gimbal_pitch_delta": 1.0,
  "gimbal_yaw_delta": 1.0,
  "visual_quality": 0.5,
  "duration_bonus": -0.5,
  "altitude_delta": 1.5,
  "gps_speed": 1.5,
  "rapid_rotation_penalty": 1.5,
  "min_duration_sec": 2.0
}
```

### Precision
```json
{
  "motion_magnitude": -0.5,
  "gimbal_smoothness": 1.0,
  "visual_quality": 2.5,
  "gps_speed": -0.3,
  "pitch_angle_bonus": { "target": -90, "weight": 2.0 },
  "min_duration_sec": 3.0
}
```

### Social
```json
{
  "motion_magnitude": 1.0,
  "gimbal_smoothness": 1.0,
  "visual_quality": 1.0,
  "duration_bonus": -1.0,
  "center_frame_bonus": 1.5,
  "min_duration_sec": 2.0,
  "max_duration_sec": 8.0
}
```

### Discovery
```json
{
  "min_score_threshold": 0.0
}
```
(All weights = 1.0, shows everything)

---

# 5. Development Phases

## Phase 1: Foundation (Weeks 1-2)

| Task | Deliverable |
|------|-------------|
| Tauri project setup | Basic window, IPC working |
| SRT parser (Rust) | Extract all telemetry fields |
| SQLite schema | All tables, CRUD operations |
| FFmpeg wrapper | VideoToolbox acceleration |
| LRF detection | Validate and use DJI proxies |
| Proxy generation | Fallback to FFmpeg when needed |
| Thumbnail extraction | 1fps JPEGs |

**Milestone:** Import folder → parse SRTs → generate proxies → populate database

## Phase 2: Analysis Engine (Weeks 3-4)

| Task | Deliverable |
|------|-------------|
| Telemetry signal extraction | Gimbal deltas, GPS speed, ISO quality |
| Segment detection | Split clips into candidates |
| Python sidecar | OpenCV motion analysis |
| Profile definitions | 5 profiles with weights |
| Score calculator | On-demand per profile |

**Milestone:** Footage analyzed → segments scored differently per profile

## Phase 3: User Interface (Weeks 5-6)

| Task | Deliverable |
|------|-------------|
| Ingest view | Folder picker, progress bars |
| Review layout | Profile tabs, thumbnail grid |
| Video preview | Proxy playback |
| Selection management | Add/remove, reorder |
| Clip adjustment | In/out point trimming |

**Milestone:** Full workflow functional in UI

## Phase 4: Export (Weeks 7-8)

| Task | Deliverable |
|------|-------------|
| Clip export | Fast (stream copy) and Precise modes |
| Multi-format | 16:9, 9:16, 1:1 |
| Premiere XML | FCP XML v4 with source paths |
| Auto video | Concat + transitions |

**Milestone:** All three export paths working

## Phase 5: Polish (Weeks 9-10)

| Task | Deliverable |
|------|-------------|
| Device detection | USB/SD card mount watching |
| Settings panel | Preferences |
| Error handling | Graceful failures |
| DMG packaging | Installable app |

**Milestone:** Shippable v1.0

## Future Phases

- **Phase 6:** Semantic search (CLIP embeddings)
- **Phase 7:** Learning system ("My Style")
- **Phase 8:** Object detection (YOLO)
- **Phase 9:** Windows support

---

# 6. Research Findings

## 6.1 LRF Proxy Optimization

### What LRF Files Are
- Low Resolution Files: 720p (960×720) H.264 proxies
- Bitrate: 7-8 Mbit/s
- Automatically generated by DJI drones
- Rename `.LRF` to `.MP4` to play in any player

### Drone Compatibility

| Drone | LRF Files |
|-------|-----------|
| Mavic 3 / 3 Pro / 3 Cine | ✅ Yes |
| Air 3 | ✅ Yes |
| Mini 4 Pro | ✅ Yes |
| Mini 3 Pro | ✅ Yes |
| Mini 3 (non-Pro) | ❌ No |
| Air 3S | ❌ No |

**Result:** ~90% of target users have LRF files.

### Critical Issues

1. **Frame Rate Mismatch:** LRF capped at 30fps (60fps/120fps source → 30fps LRF). Fine for preview, NOT for NLE proxies.

2. **Duration Bug:** DJI Air 3 has 50/50 chance LRF is 1 frame shorter. Validate during ingest.

3. **File Location:** LRF files in separate `LRF/` folder, not alongside MP4s.

### LRF Validation Flow

```
1. Scan DCIM folder for .MP4/.MOV files
2. For each video:
   - Check if matching .LRF exists in LRF/ folder
   - If YES → Validate:
     - File exists and non-zero size
     - Playable (FFprobe check)
     - Duration matches source (±1 frame tolerance)
   - If valid → Copy & rename to proxy folder
   - If invalid → Generate via FFmpeg
3. Log proxy source (LRF vs generated)
```

### Performance Impact

**With LRF (90% of users):**
- 1hr 4K footage: ~5 min ingest
  - LRF copy: 30s
  - SRT parse: 2s
  - Thumbnails: 60s
  - Motion analysis: 3-5 min

**Without LRF (10% of users):**
- 1hr 4K footage: ~20 min ingest (FFmpeg proxy generation)

## 6.2 FFmpeg Stream Copy

### Confirmed Valid
- `-c copy` performs lossless stream copy
- Speed: ~2 seconds per clip vs ~30 seconds for re-encode
- Quality: 100% original

### Keyframe Snapping
- Stream copy only cuts at keyframes (I-frames)
- DJI drones: 1-2 second GOP (Group of Pictures)
- Result: ±1-2 second boundary shift

### Best Practice

```bash
ffmpeg -ss 00:01:30 -to 00:02:00 -i input.mp4 \
  -c copy -avoid_negative_ts make_zero \
  output.mp4
```

### Export Modes

| Mode | Speed | Quality | Precision | Best For |
|------|-------|---------|-----------|----------|
| Fast | ~2 sec | 100% | ±1-2 sec | Quick sharing |
| Precise | ~30 sec | 98% | Frame-exact | Professional |

---

# 7. Appendices

## 7.1 DJI Folder Structure

```
/Volumes/DJI_MAVIC_3/
├── DCIM/
│   ├── 100MEDIA/
│   │   ├── DJI_0001.MP4    # 4K source
│   │   ├── DJI_0001.SRT    # Telemetry
│   │   └── ...
│   ├── PANORAMA/
│   └── TIMELAPSE/
├── LRF/
│   └── 100MEDIA/
│       ├── DJI_0001.LRF    # 720p proxy
│       └── ...
└── MISC/
```

## 7.2 SRT Telemetry Format

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

## 7.3 FFmpeg Command Reference

### Thumbnail Extraction
```bash
ffmpeg -hwaccel videotoolbox \
  -i input.mp4 \
  -vf "fps=1,scale=320:180" \
  -q:v 3 \
  thumbnails/%04d.jpg
```

### Vertical Crop (9:16)
```bash
ffmpeg -hwaccel videotoolbox \
  -i input.mp4 \
  -vf "crop=ih*9/16:ih,scale=1080:1920" \
  -c:v libx264 -crf 18 \
  output_vertical.mp4
```

### Concatenation with Crossfade
```bash
ffmpeg -hwaccel videotoolbox \
  -i clip1.mp4 -i clip2.mp4 \
  -filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=4.5[v]" \
  -map "[v]" \
  output.mp4
```

---

## 7.4 Project File Structure

```
skyclip/
├── src-tauri/                    # Rust backend
│   ├── src/
│   │   ├── main.rs
│   │   ├── commands/             # Tauri IPC
│   │   ├── services/             # srt_parser, ffmpeg, database
│   │   └── models/
│   ├── Cargo.toml
│   └── tauri.conf.json
├── src/                          # React frontend
│   ├── components/
│   ├── hooks/
│   └── stores/
├── python/                       # Python sidecar
│   └── skyclip_analyzer/
├── profiles/                     # JSON configs
├── docs/
│   └── SPEC.md
├── CLAUDE.md
└── package.json
```

---

**End of Specification**
