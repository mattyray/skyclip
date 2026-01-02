use crate::commands::ingest::AppState;
use crate::models::Segment;
use crate::services::{ScoreCalculator, SrtParser, TelemetryAnalyzer};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tauri::State;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzeResult {
    pub clip_id: String,
    pub segments_created: u32,
    pub top_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentWithScores {
    pub segment: Segment,
    pub scores: HashMap<String, f64>,
}

/// Analyze a single clip and generate scored segments
#[tauri::command]
pub async fn analyze_clip(
    state: State<'_, AppState>,
    clip_id: String,
    profile_id: Option<String>,
) -> Result<AnalyzeResult, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;
    let app_data_dir = state.app_data_dir.lock().await.clone();

    // Get the clip
    let clip = db
        .get_clip(&clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip not found")?;

    // Check for SRT file
    let srt_path = clip.srt_path.ok_or("Clip has no SRT telemetry file")?;

    // Parse telemetry
    let parser = SrtParser::new();
    let frames = parser.parse_file(&srt_path).map_err(|e| e.to_string())?;

    if frames.is_empty() {
        return Err("No telemetry frames found in SRT file".to_string());
    }

    // Initialize analyzer and score calculator
    let analyzer = TelemetryAnalyzer::new();
    let mut score_calc = ScoreCalculator::new();

    // Load profiles
    let profiles_dir = app_data_dir.join("profiles");
    if profiles_dir.exists() {
        score_calc
            .load_profiles_from_dir(&profiles_dir)
            .map_err(|e| e.to_string())?;
    }

    // Also try loading from bundled profiles
    let bundled_profiles = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("profiles"));
    if let Some(bundled) = bundled_profiles {
        if bundled.exists() {
            let _ = score_calc.load_profiles_from_dir(&bundled);
        }
    }

    // If no profiles loaded, use default discovery profile
    if score_calc.get_profiles().is_empty() {
        score_calc
            .load_profile(DEFAULT_DISCOVERY_PROFILE)
            .map_err(|e| e.to_string())?;
    }

    // Detect segments based on profile thresholds
    let active_profile = profile_id.as_deref().unwrap_or("discovery");
    let (min_dur, max_dur) = match score_calc.get_profile(active_profile) {
        Some(p) => (
            p.thresholds.min_duration_sec.unwrap_or(5.0),
            p.thresholds.max_duration_sec.unwrap_or(30.0),
        ),
        None => (5.0, 30.0),
    };

    let segment_indices = analyzer.detect_segments(&frames, min_dur, max_dur);

    // Delete existing segments for this clip
    db.delete_segments_for_clip(&clip_id)
        .await
        .map_err(|e| e.to_string())?;

    let thumbnails_dir = app_data_dir.join("thumbnails").join(&clip_id);
    let mut segments_created = 0u32;
    let mut top_score = 0.0f64;

    for (start_idx, end_idx) in segment_indices {
        let segment_frames = &frames[start_idx..end_idx];
        if segment_frames.is_empty() {
            continue;
        }

        let start_time_ms = segment_frames.first().map(|f| f.start_time_ms).unwrap_or(0);
        let end_time_ms = segment_frames.last().map(|f| f.end_time_ms).unwrap_or(0);
        let duration_sec = (end_time_ms - start_time_ms) as f64 / 1000.0;

        // Analyze segment
        let signals = analyzer.analyze_frames(segment_frames);

        // Calculate scores for all profiles
        let scores = score_calc.calculate_all_scores(&signals);

        // Get score for active profile
        let active_score = scores.get(active_profile).copied().unwrap_or(0.0);

        // Check if passes thresholds
        if !score_calc.passes_thresholds(active_profile, &signals, duration_sec) {
            continue;
        }

        // Find thumbnail for this segment (use frame at segment start)
        let thumb_second = (start_time_ms / 1000) as u32;
        let thumbnail_path = thumbnails_dir.join(format!("thumb_{:04}.jpg", thumb_second));
        let thumbnail_path_str = if thumbnail_path.exists() {
            Some(thumbnail_path.to_string_lossy().to_string())
        } else {
            None
        };

        // Create segment
        let mut segment = Segment::new(clip_id.clone(), start_time_ms, end_time_ms);
        segment.thumbnail_path = thumbnail_path_str;
        segment.motion_magnitude = Some(signals.motion_magnitude);
        segment.gimbal_pitch_delta_avg = Some(signals.gimbal_pitch_delta_avg);
        segment.gimbal_yaw_delta_avg = Some(signals.gimbal_yaw_delta_avg);
        segment.gimbal_smoothness = Some(signals.gimbal_smoothness);
        segment.altitude_delta = Some(signals.altitude_delta);
        segment.gps_speed_avg = Some(signals.gps_speed_avg);
        segment.iso_avg = Some(signals.iso_avg);

        db.insert_segment(&segment)
            .await
            .map_err(|e| e.to_string())?;

        // Store scores for this segment
        for (profile_id, score) in &scores {
            db.insert_segment_score(&segment.id, profile_id, *score)
                .await
                .map_err(|e| e.to_string())?;
        }

        segments_created += 1;
        if active_score > top_score {
            top_score = active_score;
        }
    }

    Ok(AnalyzeResult {
        clip_id,
        segments_created,
        top_score,
    })
}

/// Analyze all clips in a flight
#[tauri::command]
pub async fn analyze_flight(
    state: State<'_, AppState>,
    flight_id: String,
    profile_id: Option<String>,
) -> Result<Vec<AnalyzeResult>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    let clips = db
        .get_clips_for_flight(&flight_id)
        .await
        .map_err(|e| e.to_string())?;

    drop(db_guard); // Release lock before calling analyze_clip

    let mut results = Vec::new();
    for clip in clips {
        if clip.srt_path.is_some() {
            match analyze_clip(state.clone(), clip.id.clone(), profile_id.clone()).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Failed to analyze clip {}: {}", clip.filename, e);
                }
            }
        }
    }

    Ok(results)
}

/// Get segments for a clip with their scores
#[tauri::command]
pub async fn get_clip_segments(
    state: State<'_, AppState>,
    clip_id: String,
) -> Result<Vec<SegmentWithScores>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    let segments = db
        .get_segments_for_clip(&clip_id)
        .await
        .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for segment in segments {
        let scores = db
            .get_segment_scores(&segment.id)
            .await
            .map_err(|e| e.to_string())?;
        results.push(SegmentWithScores { segment, scores });
    }

    Ok(results)
}

/// Get top segments across a flight, sorted by score
#[tauri::command]
pub async fn get_top_segments(
    state: State<'_, AppState>,
    flight_id: String,
    profile_id: String,
    limit: Option<u32>,
) -> Result<Vec<SegmentWithScores>, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    let limit = limit.unwrap_or(20);
    let segments = db
        .get_top_segments_for_flight(&flight_id, &profile_id, limit)
        .await
        .map_err(|e| e.to_string())?;

    let mut results = Vec::new();
    for segment in segments {
        let scores = db
            .get_segment_scores(&segment.id)
            .await
            .map_err(|e| e.to_string())?;
        results.push(SegmentWithScores { segment, scores });
    }

    Ok(results)
}

/// List available profiles
#[tauri::command]
pub async fn list_profiles(state: State<'_, AppState>) -> Result<Vec<ProfileInfo>, String> {
    let app_data_dir = state.app_data_dir.lock().await.clone();
    let mut score_calc = ScoreCalculator::new();

    // Load profiles
    let profiles_dir = app_data_dir.join("profiles");
    if profiles_dir.exists() {
        let _ = score_calc.load_profiles_from_dir(&profiles_dir);
    }

    let bundled_profiles = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("profiles"));
    if let Some(bundled) = bundled_profiles {
        if bundled.exists() {
            let _ = score_calc.load_profiles_from_dir(&bundled);
        }
    }

    if score_calc.get_profiles().is_empty() {
        let _ = score_calc.load_profile(DEFAULT_DISCOVERY_PROFILE);
    }

    Ok(score_calc
        .get_profiles()
        .iter()
        .map(|p| ProfileInfo {
            id: p.id.clone(),
            name: p.name.clone(),
            description: p.description.clone(),
        })
        .collect())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileInfo {
    pub id: String,
    pub name: String,
    pub description: String,
}

const DEFAULT_DISCOVERY_PROFILE: &str = r#"{
  "id": "discovery",
  "name": "Discovery",
  "description": "Balanced scoring. Find all potentially good moments.",
  "weights": {
    "gimbal_smoothness": 0.20,
    "gimbal_pitch_delta": 0.15,
    "gimbal_yaw_delta": 0.15,
    "gps_speed": 0.15,
    "motion_magnitude": 0.15,
    "altitude_delta": 0.10,
    "iso_penalty": 0.10
  },
  "thresholds": {
    "min_duration_sec": 3,
    "max_duration_sec": 30
  }
}"#;
