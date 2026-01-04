use crate::services::{Director, SegmentContext, DirectorResponse, EditSequence, EditDecision};
use std::fs;
use std::path::PathBuf;
use tauri::State;
use std::sync::Mutex;

/// Simple file-based storage for API key (in app data directory)
fn get_api_key_path() -> PathBuf {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let config_dir = home.join(".skyclip");
    fs::create_dir_all(&config_dir).ok();
    config_dir.join("anthropic_api_key")
}

#[tauri::command]
pub async fn save_api_key(api_key: String) -> Result<(), String> {
    let path = get_api_key_path();
    fs::write(&path, &api_key).map_err(|e| format!("Failed to save API key: {}", e))?;
    Ok(())
}

#[tauri::command]
pub async fn get_api_key() -> Result<Option<String>, String> {
    let path = get_api_key_path();
    if path.exists() {
        let key = fs::read_to_string(&path).map_err(|e| format!("Failed to read API key: {}", e))?;
        let key = key.trim().to_string();
        if key.is_empty() {
            Ok(None)
        } else {
            Ok(Some(key))
        }
    } else {
        Ok(None)
    }
}

#[tauri::command]
pub async fn clear_api_key() -> Result<(), String> {
    let path = get_api_key_path();
    if path.exists() {
        fs::remove_file(&path).map_err(|e| format!("Failed to clear API key: {}", e))?;
    }
    Ok(())
}

/// Segment info from frontend
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SegmentInput {
    pub id: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub thumbnail_path: Option<String>,
    pub gimbal_pitch_delta: Option<f64>,
    pub gimbal_yaw_delta: Option<f64>,
    pub gimbal_smoothness: Option<f64>,
    pub gps_speed: Option<f64>,
    pub altitude_delta: Option<f64>,
    pub score: Option<f64>,
}

#[tauri::command]
pub async fn director_generate_edit(
    prompt: String,
    segments: Vec<SegmentInput>,
    target_duration_sec: Option<f64>,
) -> Result<EditSequence, String> {
    // Get API key
    let api_key = get_api_key().await?
        .ok_or_else(|| "No Anthropic API key configured. Please add your API key in settings.".to_string())?;

    // Convert to SegmentContext
    let contexts: Vec<SegmentContext> = segments
        .iter()
        .map(|s| SegmentContext {
            id: s.id.clone(),
            duration_sec: (s.end_ms - s.start_ms) as f64 / 1000.0,
            start_ms: s.start_ms,
            end_ms: s.end_ms,
            gimbal_pitch_delta: s.gimbal_pitch_delta.unwrap_or(0.0),
            gimbal_yaw_delta: s.gimbal_yaw_delta.unwrap_or(0.0),
            gimbal_smoothness: s.gimbal_smoothness.unwrap_or(1.0),
            gps_speed: s.gps_speed.unwrap_or(0.0),
            altitude_delta: s.altitude_delta.unwrap_or(0.0),
            score: s.score.unwrap_or(50.0),
        })
        .collect();

    // Collect thumbnail paths
    let thumbnail_paths: Vec<String> = segments
        .iter()
        .filter_map(|s| s.thumbnail_path.clone())
        .collect();

    // Call Claude
    let director = Director::new(api_key);
    let response = director
        .generate_edit(&prompt, contexts, thumbnail_paths, target_duration_sec)
        .await
        .map_err(|e| format!("Director API error: {}", e))?;

    // Convert to EditSequence format
    let decisions: Vec<EditDecision> = response
        .edit_sequence
        .into_iter()
        .map(|d| EditDecision {
            clip_id: d.segment_id,
            sequence_order: d.sequence_order,
            adjusted_start_ms: d.in_point_ms,
            adjusted_end_ms: d.out_point_ms,
            transition_type: d.transition_to_next,
            transition_duration_ms: d.transition_duration_ms,
            confidence: 0.9, // AI suggestions get high confidence
            reasoning: d.reasoning,
        })
        .collect();

    let total_duration_ms = (response.total_duration_sec * 1000.0) as i64;

    Ok(EditSequence {
        decisions,
        total_duration_ms,
        style: format!("AI Director: {}", response.style_notes),
        was_reordered: true, // AI may have reordered
    })
}
