use crate::models::TelemetryFrame;

/// Computed signal values for a segment of video
#[derive(Debug, Clone, Default)]
pub struct SegmentSignals {
    /// Average rate of gimbal pitch change (degrees/sec)
    pub gimbal_pitch_delta_avg: f64,
    /// Average rate of gimbal yaw change (degrees/sec)
    pub gimbal_yaw_delta_avg: f64,
    /// Gimbal smoothness score (0-1, higher = smoother)
    pub gimbal_smoothness: f64,
    /// Average GPS horizontal speed (m/s)
    pub gps_speed_avg: f64,
    /// Total altitude change over segment (meters)
    pub altitude_delta: f64,
    /// Average ISO value
    pub iso_avg: f64,
    /// Motion magnitude combining all movement signals
    pub motion_magnitude: f64,
}

pub struct TelemetryAnalyzer;

impl TelemetryAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Analyze a slice of telemetry frames and compute signal values
    pub fn analyze_frames(&self, frames: &[TelemetryFrame]) -> SegmentSignals {
        if frames.is_empty() {
            return SegmentSignals::default();
        }

        if frames.len() == 1 {
            return SegmentSignals {
                iso_avg: frames[0].iso.map(|i| i as f64).unwrap_or(0.0),
                gimbal_smoothness: 1.0, // Single frame = perfectly smooth
                ..Default::default()
            };
        }

        // Compute gimbal deltas
        let (pitch_deltas, yaw_deltas) = self.compute_gimbal_deltas(frames);
        let gimbal_pitch_delta_avg = average(&pitch_deltas);
        let gimbal_yaw_delta_avg = average(&yaw_deltas);

        // Gimbal smoothness: inverse of jitter (standard deviation of deltas)
        // If no gimbal data available, default to 1.0 (perfectly smooth - no jitter detected)
        let gimbal_smoothness = if pitch_deltas.is_empty() && yaw_deltas.is_empty() {
            1.0 // No gimbal data = assume smooth (neutral value that passes thresholds)
        } else {
            let pitch_jitter = std_dev(&pitch_deltas);
            let yaw_jitter = std_dev(&yaw_deltas);
            let combined_jitter = (pitch_jitter + yaw_jitter) / 2.0;
            // Map jitter to 0-1 smoothness (lower jitter = higher smoothness)
            // Using sigmoid-like transform: smoothness = 1 / (1 + jitter/10)
            1.0 / (1.0 + combined_jitter / 10.0)
        };

        // Compute GPS speed
        let gps_speeds = self.compute_gps_speeds(frames);
        let gps_speed_avg = average(&gps_speeds);

        // Compute altitude delta (first to last)
        let altitude_delta = self.compute_altitude_delta(frames);

        // Compute average ISO
        let iso_values: Vec<f64> = frames
            .iter()
            .filter_map(|f| f.iso.map(|i| i as f64))
            .collect();
        let iso_avg = if iso_values.is_empty() {
            0.0
        } else {
            average(&iso_values)
        };

        // Motion magnitude: combined normalized score
        // Adapt weights based on available data
        let gimbal_motion = (gimbal_pitch_delta_avg.abs() + gimbal_yaw_delta_avg.abs()) / 2.0;
        let has_gimbal_data = !pitch_deltas.is_empty() || !yaw_deltas.is_empty();

        // If no gimbal data, rely entirely on GPS; otherwise use weighted combination
        let motion_magnitude = if has_gimbal_data {
            (gimbal_motion * 0.6) + (gps_speed_avg * 0.4)
        } else {
            // GPS-only mode: use GPS speed + altitude change as motion indicator
            gps_speed_avg + (altitude_delta.abs() / 10.0) // altitude contributes up to 5 units per 50m
        };

        SegmentSignals {
            gimbal_pitch_delta_avg,
            gimbal_yaw_delta_avg,
            gimbal_smoothness,
            gps_speed_avg,
            altitude_delta,
            iso_avg,
            motion_magnitude,
        }
    }

    /// Compute frame-to-frame gimbal pitch and yaw deltas
    fn compute_gimbal_deltas(&self, frames: &[TelemetryFrame]) -> (Vec<f64>, Vec<f64>) {
        let mut pitch_deltas = Vec::new();
        let mut yaw_deltas = Vec::new();

        for window in frames.windows(2) {
            let prev = &window[0];
            let curr = &window[1];

            // Time delta in seconds
            let time_delta_sec = (curr.start_time_ms - prev.start_time_ms) as f64 / 1000.0;
            if time_delta_sec <= 0.0 {
                continue;
            }

            // Pitch delta (degrees per second)
            if let (Some(prev_pitch), Some(curr_pitch)) = (prev.gimbal_pitch, curr.gimbal_pitch) {
                let delta = (curr_pitch - prev_pitch) / time_delta_sec;
                pitch_deltas.push(delta);
            }

            // Yaw delta - handle wraparound at 180/-180
            if let (Some(prev_yaw), Some(curr_yaw)) = (prev.gimbal_yaw, curr.gimbal_yaw) {
                let mut delta = curr_yaw - prev_yaw;
                // Handle wraparound
                if delta > 180.0 {
                    delta -= 360.0;
                } else if delta < -180.0 {
                    delta += 360.0;
                }
                let delta_per_sec = delta / time_delta_sec;
                yaw_deltas.push(delta_per_sec);
            }
        }

        (pitch_deltas, yaw_deltas)
    }

    /// Compute GPS horizontal speed between consecutive frames
    fn compute_gps_speeds(&self, frames: &[TelemetryFrame]) -> Vec<f64> {
        let mut speeds = Vec::new();

        for window in frames.windows(2) {
            let prev = &window[0];
            let curr = &window[1];

            let time_delta_sec = (curr.start_time_ms - prev.start_time_ms) as f64 / 1000.0;
            if time_delta_sec <= 0.0 {
                continue;
            }

            if let (Some(lat1), Some(lon1), Some(lat2), Some(lon2)) = (
                prev.latitude,
                prev.longitude,
                curr.latitude,
                curr.longitude,
            ) {
                // Skip invalid coordinates (0,0 or very small values indicate no GPS fix)
                if lat1.abs() < 0.1 || lon1.abs() < 0.1 || lat2.abs() < 0.1 || lon2.abs() < 0.1 {
                    continue;
                }

                let distance = haversine_distance(lat1, lon1, lat2, lon2);
                let speed = distance / time_delta_sec;

                // Sanity check: max drone speed is ~30 m/s (108 km/h) for consumer drones
                // Allow up to 50 m/s to account for wind/diving, but filter GPS glitches
                if speed <= 50.0 {
                    speeds.push(speed);
                }
            }
        }

        speeds
    }

    /// Compute total altitude change from first to last frame
    fn compute_altitude_delta(&self, frames: &[TelemetryFrame]) -> f64 {
        let first_alt = frames.iter().find_map(|f| f.altitude);
        let last_alt = frames.iter().rev().find_map(|f| f.altitude);

        match (first_alt, last_alt) {
            (Some(first), Some(last)) => last - first,
            _ => 0.0,
        }
    }

    /// Detect segments from telemetry based on activity thresholds
    /// Returns (start_index, end_index) pairs
    pub fn detect_segments(
        &self,
        frames: &[TelemetryFrame],
        min_segment_sec: f64,
        max_segment_sec: f64,
    ) -> Vec<(usize, usize)> {
        if frames.is_empty() {
            return vec![];
        }

        // Calculate actual frame rate from timestamps
        let total_duration_ms = frames.last().map(|f| f.end_time_ms).unwrap_or(0)
            - frames.first().map(|f| f.start_time_ms).unwrap_or(0);
        let total_duration_sec = total_duration_ms as f64 / 1000.0;

        // Frames per second (could be 1fps for old format, 60fps for new Mavic 3)
        let fps = if total_duration_sec > 0.0 {
            frames.len() as f64 / total_duration_sec
        } else {
            1.0
        };

        let mut segments = Vec::new();
        let min_frames = (min_segment_sec * fps) as usize;
        let max_frames = (max_segment_sec * fps) as usize;

        // If clip is shorter than min segment, create one segment for the whole clip
        if frames.len() < min_frames {
            // Still create a segment if we have at least some data
            if frames.len() >= 2 {
                segments.push((0, frames.len()));
            }
            return segments;
        }

        let mut start = 0;
        while start < frames.len() {
            let end = (start + max_frames).min(frames.len());
            // Accept segment if it meets minimum OR if it's the last chunk and has some content
            if end - start >= min_frames || (end == frames.len() && end - start >= min_frames / 2) {
                segments.push((start, end));
            }
            start = end;
        }

        segments
    }
}

/// Calculate haversine distance between two GPS points in meters
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_371_000.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();

    EARTH_RADIUS_M * c
}

/// Calculate average of values
fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Calculate standard deviation
fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let avg = average(values);
    let variance = values.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(index: u32, start_ms: i64, pitch: f64, yaw: f64, lat: f64, lon: f64) -> TelemetryFrame {
        TelemetryFrame {
            index,
            start_time_ms: start_ms,
            end_time_ms: start_ms + 1000,
            timestamp: None,
            iso: Some(100),
            shutter: None,
            fnum: None,
            ev: None,
            ct: None,
            color_md: None,
            focal_len: None,
            latitude: Some(lat),
            longitude: Some(lon),
            altitude: Some(100.0),
            gimbal_yaw: Some(yaw),
            gimbal_pitch: Some(pitch),
            gimbal_roll: Some(0.0),
        }
    }

    #[test]
    fn test_gimbal_deltas() {
        let analyzer = TelemetryAnalyzer::new();
        let frames = vec![
            make_frame(1, 0, -15.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, -20.0, 10.0, 40.0, -74.0),
            make_frame(3, 2000, -25.0, 20.0, 40.0, -74.0),
        ];

        let signals = analyzer.analyze_frames(&frames);

        // Pitch goes from -15 to -25 over 2 seconds = -5 deg/sec average
        assert!((signals.gimbal_pitch_delta_avg - (-5.0)).abs() < 0.1);
        // Yaw goes from 0 to 20 over 2 seconds = 10 deg/sec average
        assert!((signals.gimbal_yaw_delta_avg - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_gps_speed() {
        let analyzer = TelemetryAnalyzer::new();
        // Two points roughly 111 meters apart (0.001 degree latitude)
        let frames = vec![
            make_frame(1, 0, 0.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, 0.0, 0.0, 40.001, -74.0),
        ];

        let signals = analyzer.analyze_frames(&frames);

        // Should be approximately 111 m/s (very fast, but validates calculation)
        assert!(signals.gps_speed_avg > 100.0 && signals.gps_speed_avg < 120.0);
    }

    #[test]
    fn test_altitude_delta() {
        let analyzer = TelemetryAnalyzer::new();
        let mut frames = vec![
            make_frame(1, 0, 0.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, 0.0, 0.0, 40.0, -74.0),
        ];
        frames[0].altitude = Some(100.0);
        frames[1].altitude = Some(150.0);

        let signals = analyzer.analyze_frames(&frames);

        assert!((signals.altitude_delta - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_smoothness() {
        let analyzer = TelemetryAnalyzer::new();

        // Smooth movement: consistent gimbal deltas
        let smooth_frames = vec![
            make_frame(1, 0, 0.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, 5.0, 5.0, 40.0, -74.0),
            make_frame(3, 2000, 10.0, 10.0, 40.0, -74.0),
            make_frame(4, 3000, 15.0, 15.0, 40.0, -74.0),
        ];

        // Jerky movement: inconsistent gimbal deltas
        let jerky_frames = vec![
            make_frame(1, 0, 0.0, 0.0, 40.0, -74.0),
            make_frame(2, 1000, 20.0, -10.0, 40.0, -74.0),
            make_frame(3, 2000, -5.0, 30.0, 40.0, -74.0),
            make_frame(4, 3000, 15.0, 5.0, 40.0, -74.0),
        ];

        let smooth_signals = analyzer.analyze_frames(&smooth_frames);
        let jerky_signals = analyzer.analyze_frames(&jerky_frames);

        // Smooth should have higher smoothness score
        assert!(smooth_signals.gimbal_smoothness > jerky_signals.gimbal_smoothness);
    }
}
