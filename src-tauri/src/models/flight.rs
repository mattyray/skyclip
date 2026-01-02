use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flight {
    pub id: String,
    pub name: String,
    pub import_date: DateTime<Utc>,
    pub source_path: String,
    pub location_name: Option<String>,
    pub gps_center_lat: Option<f64>,
    pub gps_center_lon: Option<f64>,
    pub total_duration_sec: Option<f64>,
    pub total_clips: Option<i32>,
}

impl Flight {
    pub fn new(name: String, source_path: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            import_date: Utc::now(),
            source_path,
            location_name: None,
            gps_center_lat: None,
            gps_center_lon: None,
            total_duration_sec: None,
            total_clips: None,
        }
    }
}
