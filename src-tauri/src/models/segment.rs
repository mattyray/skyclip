use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub id: String,
    pub source_clip_id: String,
    pub start_time_ms: i64,
    pub end_time_ms: i64,
    pub duration_ms: i64,
    pub thumbnail_path: Option<String>,

    // Raw signals (computed once)
    pub motion_magnitude: Option<f64>,
    pub gimbal_pitch_delta_avg: Option<f64>,
    pub gimbal_yaw_delta_avg: Option<f64>,
    pub gimbal_smoothness: Option<f64>,
    pub altitude_delta: Option<f64>,
    pub gps_speed_avg: Option<f64>,
    pub iso_avg: Option<f64>,
    pub visual_quality: Option<f64>,
    pub has_scene_change: Option<bool>,

    // User state
    pub is_selected: bool,
    pub user_adjusted_start_ms: Option<i64>,
    pub user_adjusted_end_ms: Option<i64>,
}

impl Segment {
    pub fn new(source_clip_id: String, start_time_ms: i64, end_time_ms: i64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_clip_id,
            start_time_ms,
            end_time_ms,
            duration_ms: end_time_ms - start_time_ms,
            thumbnail_path: None,
            motion_magnitude: None,
            gimbal_pitch_delta_avg: None,
            gimbal_yaw_delta_avg: None,
            gimbal_smoothness: None,
            altitude_delta: None,
            gps_speed_avg: None,
            iso_avg: None,
            visual_quality: None,
            has_scene_change: None,
            is_selected: false,
            user_adjusted_start_ms: None,
            user_adjusted_end_ms: None,
        }
    }
}
