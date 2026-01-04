use serde::{Deserialize, Serialize};
use tauri::State;

use crate::commands::ingest::AppState;
use crate::services::{PythonSidecar, ClipInfo, FFmpeg, ConcatClip};

/// Visual analysis result for a segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAnalysisResult {
    pub segment_id: String,
    pub motion_avg: Option<f64>,
    pub motion_direction: Option<f64>,
    pub dominant_color: Option<(u8, u8, u8)>,
    pub is_golden_hour: Option<bool>,
    pub scene_type: Option<String>,
    pub has_subject: Option<bool>,
}

/// Edit decision for frontend display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditDecisionResponse {
    pub clip_id: String,
    pub sequence_order: i32,
    pub adjusted_start_ms: i64,
    pub adjusted_end_ms: i64,
    pub transition_type: String,
    pub transition_duration_ms: i64,
    pub confidence: f64,
    pub reasoning: String,
}

/// Edit sequence for frontend display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditSequenceResponse {
    pub decisions: Vec<EditDecisionResponse>,
    pub total_duration_ms: i64,
    pub style: String,
    pub was_reordered: bool,
}

/// Check if Python sidecar is available
#[tauri::command]
pub async fn check_python_available() -> Result<bool, String> {
    match PythonSidecar::new() {
        Ok(sidecar) => Ok(sidecar.is_available()),
        Err(_) => Ok(false),
    }
}

/// Install Python dependencies
#[tauri::command]
pub async fn install_python_deps() -> Result<(), String> {
    let sidecar = PythonSidecar::new()?;
    sidecar.install_dependencies()
}

/// Run visual analysis on a segment
#[tauri::command]
pub async fn analyze_segment_visual(
    state: State<'_, AppState>,
    segment_id: String,
    include_objects: bool,
    include_semantic: bool,
) -> Result<VisualAnalysisResult, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    // Get segment and clip info
    let segment = db
        .get_segment(&segment_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Segment not found")?;

    let clip = db
        .get_clip(&segment.source_clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip not found")?;

    // Use proxy if available, otherwise source
    let video_path = clip.proxy_path.as_ref().unwrap_or(&clip.source_path);

    // Run Python analysis
    let sidecar = PythonSidecar::new()?;
    let analysis = sidecar.analyze_clip(
        video_path,
        segment.start_time_ms,
        Some(segment.end_time_ms),
        include_objects,
        include_semantic,
    )?;

    // Extract key metrics for frontend
    Ok(VisualAnalysisResult {
        segment_id,
        motion_avg: analysis.motion.as_ref().map(|m| m.avg_magnitude),
        motion_direction: analysis.motion.as_ref().map(|m| m.dominant_direction),
        dominant_color: analysis
            .color
            .as_ref()
            .and_then(|c| c.dominant_colors.first().cloned()),
        is_golden_hour: analysis.color.as_ref().map(|c| c.is_golden_hour),
        scene_type: analysis.semantic.as_ref().map(|s| s.scene_type.clone()),
        has_subject: analysis.objects.as_ref().map(|o| o.has_consistent_subject),
    })
}

/// Generate an edit sequence for selected segments
#[tauri::command]
pub async fn generate_edit_sequence(
    state: State<'_, AppState>,
    segment_ids: Vec<String>,
    style: String,
    reorder: bool,
) -> Result<EditSequenceResponse, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    // Build clip info list
    let mut clips = Vec::new();
    for segment_id in &segment_ids {
        let segment = db
            .get_segment(segment_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or("Segment not found")?;

        let clip = db
            .get_clip(&segment.source_clip_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or("Clip not found")?;

        let video_path = clip.proxy_path.as_ref().unwrap_or(&clip.source_path);

        clips.push(ClipInfo {
            clip_id: segment_id.clone(),
            video_path: video_path.clone(),
            start_ms: segment.start_time_ms,
            end_ms: segment.end_time_ms,
        });
    }

    // Try Python edit sequence generation, fall back to basic if unavailable
    let sequence = match PythonSidecar::new() {
        Ok(sidecar) => {
            match sidecar.generate_edit_sequence(clips.clone(), &style, reorder, false) {
                Ok(seq) => seq,
                Err(_) => create_fallback_sequence(clips, &style),
            }
        }
        Err(_) => create_fallback_sequence(clips, &style),
    };

    Ok(EditSequenceResponse {
        decisions: sequence
            .decisions
            .into_iter()
            .map(|d| EditDecisionResponse {
                clip_id: d.clip_id,
                sequence_order: d.sequence_order,
                adjusted_start_ms: d.adjusted_start_ms,
                adjusted_end_ms: d.adjusted_end_ms,
                transition_type: d.transition_type,
                transition_duration_ms: d.transition_duration_ms,
                confidence: d.confidence,
                reasoning: d.reasoning,
            })
            .collect(),
        total_duration_ms: sequence.total_duration_ms,
        style: sequence.style,
        was_reordered: sequence.was_reordered,
    })
}

/// Create a basic edit sequence without Python analysis
fn create_fallback_sequence(clips: Vec<ClipInfo>, style: &str) -> crate::services::EditSequence {
    let (default_transition, transition_duration) = match style {
        "action" => ("cut", 200),
        "social" => ("cut", 300),
        _ => ("dissolve", 500), // cinematic default
    };

    let mut total_duration_ms = 0i64;
    let decisions: Vec<crate::services::EditDecision> = clips
        .iter()
        .enumerate()
        .map(|(i, clip)| {
            let duration = clip.end_ms - clip.start_ms;
            total_duration_ms += duration;
            crate::services::EditDecision {
                clip_id: clip.clip_id.clone(),
                sequence_order: i as i32,
                adjusted_start_ms: clip.start_ms,
                adjusted_end_ms: clip.end_ms,
                transition_type: default_transition.to_string(),
                transition_duration_ms: transition_duration,
                confidence: 0.5,
                reasoning: format!("Basic {} transition (Python unavailable)", style),
            }
        })
        .collect();

    crate::services::EditSequence {
        decisions,
        total_duration_ms,
        style: style.to_string(),
        was_reordered: false,
    }
}

/// Suggest transition between two segments
#[tauri::command]
pub async fn suggest_transition(
    state: State<'_, AppState>,
    segment_a_id: String,
    segment_b_id: String,
    style: String,
) -> Result<(String, i64, f64, String), String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    // Get segment and clip info for both
    let segment_a = db
        .get_segment(&segment_a_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Segment A not found")?;
    let clip_a = db
        .get_clip(&segment_a.source_clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip A not found")?;

    let segment_b = db
        .get_segment(&segment_b_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Segment B not found")?;
    let clip_b = db
        .get_clip(&segment_b.source_clip_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or("Clip B not found")?;

    let video_a = clip_a.proxy_path.as_ref().unwrap_or(&clip_a.source_path);
    let video_b = clip_b.proxy_path.as_ref().unwrap_or(&clip_b.source_path);

    let sidecar = PythonSidecar::new()?;
    sidecar.suggest_transition(
        ClipInfo {
            clip_id: segment_a_id,
            video_path: video_a.clone(),
            start_ms: segment_a.start_time_ms,
            end_ms: segment_a.end_time_ms,
        },
        ClipInfo {
            clip_id: segment_b_id,
            video_path: video_b.clone(),
            start_ms: segment_b.start_time_ms,
            end_ms: segment_b.end_time_ms,
        },
        &style,
    )
}

/// Render input for highlight reel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderClipInput {
    pub segment_id: String,
    pub adjusted_start_ms: i64,
    pub adjusted_end_ms: i64,
    pub transition_type: String,
    pub transition_duration_ms: i64,
}

/// Render result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderResult {
    pub output_path: String,
    pub duration_sec: f64,
    pub clips_count: usize,
}

/// Render a highlight reel from an edit sequence
#[tauri::command]
pub async fn render_highlight_reel(
    state: State<'_, AppState>,
    clips: Vec<RenderClipInput>,
    output_path: String,
    use_source: bool,
) -> Result<RenderResult, String> {
    let db_guard = state.db.lock().await;
    let db = db_guard.as_ref().ok_or("Database not initialized")?;

    // Build ConcatClip list by resolving segment IDs to paths
    let mut concat_clips = Vec::new();
    let mut total_duration_ms: i64 = 0;

    for clip_input in &clips {
        let segment = db
            .get_segment(&clip_input.segment_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Segment not found: {}", clip_input.segment_id))?;

        let source_clip = db
            .get_clip(&segment.source_clip_id)
            .await
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("Clip not found: {}", segment.source_clip_id))?;

        // Use source for quality, proxy for speed
        let input_path = if use_source {
            source_clip.source_path.clone()
        } else {
            source_clip
                .proxy_path
                .clone()
                .unwrap_or(source_clip.source_path.clone())
        };

        // Use adjusted times from edit sequence
        let start_sec = clip_input.adjusted_start_ms as f64 / 1000.0;
        let end_sec = clip_input.adjusted_end_ms as f64 / 1000.0;

        concat_clips.push(ConcatClip {
            input_path,
            start_sec,
            end_sec,
            transition_type: clip_input.transition_type.clone(),
            transition_duration_ms: clip_input.transition_duration_ms,
        });

        total_duration_ms += clip_input.adjusted_end_ms - clip_input.adjusted_start_ms;
    }

    drop(db_guard);

    // Render using FFmpeg
    let ffmpeg = FFmpeg::new().map_err(|e| e.to_string())?;
    ffmpeg
        .concat_with_transitions(concat_clips.clone(), &output_path, true)
        .map_err(|e| e.to_string())?;

    Ok(RenderResult {
        output_path,
        duration_sec: total_duration_ms as f64 / 1000.0,
        clips_count: clips.len(),
    })
}
