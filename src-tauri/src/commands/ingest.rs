use crate::models::{Flight, SourceClip, TelemetryFrame};
use crate::services::{Database, FFmpeg, SrtParser};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::State;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestProgress {
    pub stage: String,
    pub current: u32,
    pub total: u32,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResult {
    pub flight_id: String,
    pub clips_count: u32,
    pub lrf_used: u32,
    pub proxies_generated: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipInfo {
    pub filename: String,
    pub source_path: String,
    pub srt_path: Option<String>,
    pub lrf_path: Option<String>,
    pub duration_sec: Option<f64>,
    pub resolution: Option<String>,
    pub framerate: Option<f64>,
}

pub struct AppState {
    pub db: Mutex<Option<Database>>,
    pub app_data_dir: Mutex<PathBuf>,
}

/// Initialize the database
#[tauri::command]
pub async fn init_database(state: State<'_, AppState>) -> Result<(), String> {
    let app_data_dir = state.app_data_dir.lock().await.clone();
    let db_path = app_data_dir.join("library.db");

    // Ensure directory exists
    std::fs::create_dir_all(&app_data_dir).map_err(|e| e.to_string())?;

    let db = Database::new(&db_path)
        .await
        .map_err(|e| e.to_string())?;

    *state.db.lock().await = Some(db);
    Ok(())
}

/// Scan a folder for DJI footage
#[tauri::command]
pub async fn scan_folder(folder_path: String) -> Result<Vec<ClipInfo>, String> {
    let folder = Path::new(&folder_path);
    if !folder.exists() {
        return Err("Folder does not exist".to_string());
    }

    let mut clips = Vec::new();
    let ffmpeg = FFmpeg::new().map_err(|e| e.to_string())?;

    // Look for DJI folder structure
    let dcim_path = folder.join("DCIM");
    let media_folders = if dcim_path.exists() {
        find_media_folders(&dcim_path)?
    } else {
        // Direct folder with media files
        vec![folder.to_path_buf()]
    };

    for media_folder in media_folders {
        let entries = std::fs::read_dir(&media_folder).map_err(|e| e.to_string())?;

        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "mp4" || ext_lower == "mov" {
                    let filename = path.file_name().unwrap().to_string_lossy().to_string();
                    let stem = path.file_stem().unwrap().to_string_lossy().to_string();

                    // Look for matching SRT file
                    let srt_path = media_folder.join(format!("{stem}.SRT"));
                    let srt_exists = srt_path.exists();

                    // Look for matching LRF file
                    let lrf_path = find_lrf_file(folder, &media_folder, &stem);

                    // Get video info
                    let video_info = ffmpeg.probe(&path).ok();

                    clips.push(ClipInfo {
                        filename,
                        source_path: path.to_string_lossy().to_string(),
                        srt_path: if srt_exists {
                            Some(srt_path.to_string_lossy().to_string())
                        } else {
                            None
                        },
                        lrf_path: lrf_path.map(|p| p.to_string_lossy().to_string()),
                        duration_sec: video_info.as_ref().map(|v| v.duration_sec),
                        resolution: video_info
                            .as_ref()
                            .map(|v| format!("{}x{}", v.width, v.height)),
                        framerate: video_info.as_ref().map(|v| v.framerate),
                    });
                }
            }
        }
    }

    clips.sort_by(|a, b| a.filename.cmp(&b.filename));
    Ok(clips)
}

/// Ingest footage from a folder
#[tauri::command]
pub async fn ingest_folder(
    state: State<'_, AppState>,
    folder_path: String,
    flight_name: String,
) -> Result<IngestResult, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    let app_data_dir = state.app_data_dir.lock().await.clone();

    let proxies_dir = app_data_dir.join("proxies");
    let thumbnails_dir = app_data_dir.join("thumbnails");
    let srt_dir = app_data_dir.join("srt");
    std::fs::create_dir_all(&proxies_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&thumbnails_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&srt_dir).map_err(|e| e.to_string())?;

    // Scan for clips
    let clips = scan_folder(folder_path.clone()).await?;
    if clips.is_empty() {
        return Err("No video clips found in folder".to_string());
    }

    // Create flight record
    let mut flight = Flight::new(flight_name, folder_path.clone());
    flight.total_clips = Some(clips.len() as i32);

    db.insert_flight(&flight).await.map_err(|e| e.to_string())?;

    let ffmpeg = FFmpeg::new().map_err(|e| e.to_string())?;
    let srt_parser = SrtParser::new();

    let mut lrf_used = 0u32;
    let mut proxies_generated = 0u32;

    for clip_info in &clips {
        let mut source_clip = SourceClip::new(
            flight.id.clone(),
            clip_info.filename.clone(),
            clip_info.source_path.clone(),
        );

        source_clip.srt_path = clip_info.srt_path.clone();
        source_clip.duration_sec = clip_info.duration_sec;

        if let Some(res) = &clip_info.resolution {
            if let Some((w, h)) = res.split_once('x') {
                source_clip.resolution_width = w.parse().ok();
                source_clip.resolution_height = h.parse().ok();
            }
        }
        source_clip.framerate = clip_info.framerate;

        // Copy and parse SRT file if available
        if let Some(srt_path) = &clip_info.srt_path {
            // Copy SRT to local storage so analysis works without source media
            let srt_filename = format!("{}.srt", source_clip.id);
            let local_srt_path = srt_dir.join(&srt_filename);
            if std::fs::copy(srt_path, &local_srt_path).is_ok() {
                source_clip.srt_path = Some(local_srt_path.to_string_lossy().to_string());
            }

            // Parse SRT for metadata
            if let Ok(frames) = srt_parser.parse_file(srt_path) {
                // Extract first timestamp as recorded_at
                if let Some(first_frame) = frames.first() {
                    source_clip.recorded_at = first_frame.timestamp;
                }
            }
        }

        // Handle proxy generation
        let proxy_filename = format!("{}.mp4", source_clip.id);
        let proxy_path = proxies_dir.join(&proxy_filename);

        if let Some(lrf_path) = &clip_info.lrf_path {
            // Validate and use LRF file
            if validate_lrf(lrf_path, clip_info.duration_sec) {
                std::fs::copy(lrf_path, &proxy_path).map_err(|e| e.to_string())?;
                source_clip.proxy_path = Some(proxy_path.to_string_lossy().to_string());
                source_clip.proxy_source = Some("lrf".to_string());
                lrf_used += 1;
            } else {
                // LRF invalid, generate proxy
                let proxy_path_str = proxy_path.to_string_lossy().to_string();
                ffmpeg
                    .generate_proxy(&clip_info.source_path, &proxy_path_str)
                    .map_err(|e| e.to_string())?;
                source_clip.proxy_path = Some(proxy_path.to_string_lossy().to_string());
                source_clip.proxy_source = Some("generated".to_string());
                proxies_generated += 1;
            }
        } else {
            // No LRF, generate proxy
            let proxy_path_str = proxy_path.to_string_lossy().to_string();
            ffmpeg
                .generate_proxy(&clip_info.source_path, &proxy_path_str)
                .map_err(|e| e.to_string())?;
            source_clip.proxy_path = Some(proxy_path.to_string_lossy().to_string());
            source_clip.proxy_source = Some("generated".to_string());
            proxies_generated += 1;
        }

        // Extract thumbnails from proxy (much faster than 4K source)
        let clip_thumb_dir = thumbnails_dir.join(&source_clip.id);
        std::fs::create_dir_all(&clip_thumb_dir).map_err(|e| e.to_string())?;

        let clip_thumb_dir_str = clip_thumb_dir.to_string_lossy().to_string();
        // Use proxy if available, otherwise fall back to source
        let thumb_source = source_clip
            .proxy_path
            .as_ref()
            .unwrap_or(&clip_info.source_path);
        let _thumbnails = ffmpeg
            .extract_thumbnails(thumb_source, &clip_thumb_dir_str, "thumb")
            .map_err(|e| e.to_string())?;

        db.insert_clip(&source_clip)
            .await
            .map_err(|e| e.to_string())?;
    }

    Ok(IngestResult {
        flight_id: flight.id,
        clips_count: clips.len() as u32,
        lrf_used,
        proxies_generated,
    })
}

/// List all flights
#[tauri::command]
pub async fn list_flights(state: State<'_, AppState>) -> Result<Vec<Flight>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.list_flights().await.map_err(|e| e.to_string())
}

/// Delete a flight and all associated data
#[tauri::command]
pub async fn delete_flight(state: State<'_, AppState>, flight_id: String) -> Result<(), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.delete_flight(&flight_id).await.map_err(|e| e.to_string())
}

/// Get clips for a flight
#[tauri::command]
pub async fn get_flight_clips(
    state: State<'_, AppState>,
    flight_id: String,
) -> Result<Vec<SourceClip>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    db.get_clips_for_flight(&flight_id)
        .await
        .map_err(|e| e.to_string())
}

/// Parse an SRT file and return telemetry frames
#[tauri::command]
pub fn parse_srt(srt_path: String) -> Result<Vec<TelemetryFrame>, String> {
    let parser = SrtParser::new();
    parser.parse_file(&srt_path).map_err(|e| e.to_string())
}

// Helper functions

fn find_media_folders(dcim_path: &Path) -> Result<Vec<PathBuf>, String> {
    let mut folders = Vec::new();

    for entry in std::fs::read_dir(dcim_path).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().unwrap().to_string_lossy();
            // DJI uses 100MEDIA, 101MEDIA, etc.
            if name.ends_with("MEDIA") || name == "PANORAMA" || name == "TIMELAPSE" {
                folders.push(path);
            }
        }
    }

    Ok(folders)
}

fn find_lrf_file(base_folder: &Path, media_folder: &Path, stem: &str) -> Option<PathBuf> {
    // Strategy 1: Check same folder as MP4 (some DJI drones put LRF alongside MP4)
    let same_folder_lrf = media_folder.join(format!("{stem}.LRF"));
    if same_folder_lrf.exists() {
        return Some(same_folder_lrf);
    }

    // Strategy 2: Check separate LRF/ folder at base level
    // Structure: base/LRF/100/ or base/LRF/100MEDIA/
    let lrf_base = base_folder.join("LRF");
    if lrf_base.exists() {
        // Try matching folder name (100MEDIA -> 100MEDIA or 100)
        if let Some(media_name) = media_folder.file_name() {
            let media_name_str = media_name.to_string_lossy();

            // Try exact match first (100MEDIA)
            let lrf_exact = lrf_base.join(&*media_name_str).join(format!("{stem}.LRF"));
            if lrf_exact.exists() {
                return Some(lrf_exact);
            }

            // Try without MEDIA suffix (100MEDIA -> 100)
            if media_name_str.ends_with("MEDIA") {
                let short_name = media_name_str.trim_end_matches("MEDIA");
                let lrf_short = lrf_base.join(short_name).join(format!("{stem}.LRF"));
                if lrf_short.exists() {
                    return Some(lrf_short);
                }
            }
        }

        // Fallback: search recursively in LRF folder
        for entry in walkdir(lrf_base).ok()?.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e.to_string_lossy().to_lowercase()) == Some("lrf".to_string()) {
                if path.file_stem().map(|s| s.to_string_lossy()) == Some(stem.into()) {
                    return Some(path.to_path_buf());
                }
            }
        }
    }

    None
}

fn walkdir(path: PathBuf) -> Result<impl Iterator<Item = Result<std::fs::DirEntry, std::io::Error>>, std::io::Error> {
    fn walk_recursive(path: PathBuf) -> Box<dyn Iterator<Item = Result<std::fs::DirEntry, std::io::Error>>> {
        match std::fs::read_dir(&path) {
            Ok(entries) => {
                let iter = entries.flat_map(move |entry| {
                    match entry {
                        Ok(e) => {
                            let path = e.path();
                            if path.is_dir() {
                                let sub = walk_recursive(path);
                                Box::new(std::iter::once(Ok(e)).chain(sub)) as Box<dyn Iterator<Item = _>>
                            } else {
                                Box::new(std::iter::once(Ok(e)))
                            }
                        }
                        Err(e) => Box::new(std::iter::once(Err(e))),
                    }
                });
                Box::new(iter)
            }
            Err(e) => Box::new(std::iter::once(Err(e))),
        }
    }
    Ok(walk_recursive(path))
}

fn validate_lrf(lrf_path: &str, expected_duration: Option<f64>) -> bool {
    let path = Path::new(lrf_path);

    // Check file exists and is non-zero
    match std::fs::metadata(path) {
        Ok(meta) if meta.len() > 0 => {}
        _ => return false,
    }

    // Validate with ffprobe
    let ffmpeg = match FFmpeg::new() {
        Ok(f) => f,
        Err(_) => return false,
    };

    let lrf_info = match ffmpeg.probe(path) {
        Ok(info) => info,
        Err(_) => return false,
    };

    // Check duration matches (±1 frame tolerance at 30fps = ~33ms)
    if let Some(expected) = expected_duration {
        let diff = (lrf_info.duration_sec - expected).abs();
        if diff > 0.1 {
            // More than 100ms difference
            return false;
        }
    }

    true
}
