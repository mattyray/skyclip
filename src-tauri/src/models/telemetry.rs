use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single frame of telemetry data parsed from DJI SRT files.
/// Each frame represents one second of data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryFrame {
    /// Frame index (1-based, matches SrtCnt)
    pub index: u32,
    /// Start time in milliseconds
    pub start_time_ms: i64,
    /// End time in milliseconds
    pub end_time_ms: i64,
    /// Timestamp from the drone
    pub timestamp: Option<DateTime<Utc>>,

    // Camera settings
    pub iso: Option<i32>,
    pub shutter: Option<String>, // e.g., "1/500.0"
    pub fnum: Option<i32>,       // f-number * 100 (e.g., 280 = f/2.8)
    pub ev: Option<f64>,         // exposure value
    pub color_temp: Option<i32>, // color temperature
    pub color_mode: Option<String>,
    pub focal_len: Option<f64>,

    // GPS data
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub altitude: Option<f64>,

    // Gimbal orientation
    pub gimbal_yaw: Option<f64>,
    pub gimbal_pitch: Option<f64>,
    pub gimbal_roll: Option<f64>,
}

impl TelemetryFrame {
    pub fn new(index: u32, start_time_ms: i64, end_time_ms: i64) -> Self {
        Self {
            index,
            start_time_ms,
            end_time_ms,
            timestamp: None,
            iso: None,
            shutter: None,
            fnum: None,
            ev: None,
            color_temp: None,
            color_mode: None,
            focal_len: None,
            latitude: None,
            longitude: None,
            altitude: None,
            gimbal_yaw: None,
            gimbal_pitch: None,
            gimbal_roll: None,
        }
    }
}
