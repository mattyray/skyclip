use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceClip {
    pub id: String,
    pub flight_id: String,
    pub filename: String,
    pub source_path: String,
    pub proxy_path: Option<String>,
    pub proxy_source: Option<String>, // "lrf" or "generated"
    pub srt_path: Option<String>,
    pub duration_sec: Option<f64>,
    pub resolution_width: Option<i32>,
    pub resolution_height: Option<i32>,
    pub framerate: Option<f64>,
    pub recorded_at: Option<DateTime<Utc>>,
}

impl SourceClip {
    pub fn new(flight_id: String, filename: String, source_path: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            flight_id,
            filename,
            source_path,
            proxy_path: None,
            proxy_source: None,
            srt_path: None,
            duration_sec: None,
            resolution_width: None,
            resolution_height: None,
            framerate: None,
            recorded_at: None,
        }
    }
}
