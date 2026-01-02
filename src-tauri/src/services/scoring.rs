use crate::services::SegmentSignals;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileWeights {
    #[serde(default)]
    pub gimbal_smoothness: f64,
    #[serde(default)]
    pub gimbal_pitch_delta: f64,
    #[serde(default)]
    pub gimbal_yaw_delta: f64,
    #[serde(default)]
    pub gps_speed: f64,
    #[serde(default)]
    pub altitude_delta: f64,
    #[serde(default)]
    pub iso_penalty: f64,
    #[serde(default)]
    pub motion_magnitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileThresholds {
    #[serde(default)]
    pub min_gimbal_smoothness: Option<f64>,
    #[serde(default)]
    pub max_iso: Option<f64>,
    #[serde(default)]
    pub min_gps_speed: Option<f64>,
    #[serde(default)]
    pub min_motion_magnitude: Option<f64>,
    #[serde(default)]
    pub min_duration_sec: Option<f64>,
    #[serde(default)]
    pub max_duration_sec: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub id: String,
    pub name: String,
    pub description: String,
    pub weights: ProfileWeights,
    #[serde(default)]
    pub thresholds: ProfileThresholds,
    #[serde(default)]
    pub preferences: HashMap<String, bool>,
}

impl Default for ProfileThresholds {
    fn default() -> Self {
        Self {
            min_gimbal_smoothness: None,
            max_iso: None,
            min_gps_speed: None,
            min_motion_magnitude: None,
            min_duration_sec: Some(3.0),
            max_duration_sec: Some(30.0),
        }
    }
}

pub struct ScoreCalculator {
    profiles: HashMap<String, Profile>,
}

impl ScoreCalculator {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Load profiles from a directory of JSON files
    pub fn load_profiles_from_dir<P: AsRef<Path>>(&mut self, dir: P) -> Result<(), String> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Err(format!("Profiles directory does not exist: {:?}", dir));
        }

        for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();

            if path.extension().map(|e| e == "json").unwrap_or(false) {
                let content = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
                let profile: Profile = serde_json::from_str(&content)
                    .map_err(|e| format!("Failed to parse {:?}: {}", path, e))?;
                self.profiles.insert(profile.id.clone(), profile);
            }
        }

        Ok(())
    }

    /// Load a single profile from JSON string
    pub fn load_profile(&mut self, json: &str) -> Result<(), String> {
        let profile: Profile =
            serde_json::from_str(json).map_err(|e| format!("Failed to parse profile: {}", e))?;
        self.profiles.insert(profile.id.clone(), profile);
        Ok(())
    }

    /// Get all loaded profiles
    pub fn get_profiles(&self) -> Vec<&Profile> {
        self.profiles.values().collect()
    }

    /// Get a specific profile by ID
    pub fn get_profile(&self, id: &str) -> Option<&Profile> {
        self.profiles.get(id)
    }

    /// Calculate score for a segment using a specific profile
    pub fn calculate_score(&self, profile_id: &str, signals: &SegmentSignals) -> Option<f64> {
        let profile = self.profiles.get(profile_id)?;
        Some(self.score_with_profile(profile, signals))
    }

    /// Calculate scores for a segment across all profiles
    pub fn calculate_all_scores(&self, signals: &SegmentSignals) -> HashMap<String, f64> {
        self.profiles
            .iter()
            .map(|(id, profile)| (id.clone(), self.score_with_profile(profile, signals)))
            .collect()
    }

    fn score_with_profile(&self, profile: &Profile, signals: &SegmentSignals) -> f64 {
        let weights = &profile.weights;

        // Normalize signals to 0-1 range for consistent scoring
        // These normalizations are based on typical drone telemetry ranges

        // Gimbal smoothness is already 0-1
        let smoothness_score = signals.gimbal_smoothness * weights.gimbal_smoothness;

        // Gimbal pitch delta: 0-30 deg/sec is typical range for intentional movement
        let pitch_normalized = (signals.gimbal_pitch_delta_avg.abs() / 30.0).min(1.0);
        let pitch_score = pitch_normalized * weights.gimbal_pitch_delta;

        // Gimbal yaw delta: 0-45 deg/sec for pans
        let yaw_normalized = (signals.gimbal_yaw_delta_avg.abs() / 45.0).min(1.0);
        let yaw_score = yaw_normalized * weights.gimbal_yaw_delta;

        // GPS speed: 0-20 m/s (0-72 km/h) covers most drone movement
        let speed_normalized = (signals.gps_speed_avg / 20.0).min(1.0);
        let speed_score = speed_normalized * weights.gps_speed;

        // Altitude delta: 0-50m change is significant
        let alt_normalized = (signals.altitude_delta.abs() / 50.0).min(1.0);
        let alt_score = alt_normalized * weights.altitude_delta;

        // ISO penalty: lower is better, 100-3200 range
        // Score decreases as ISO increases
        let iso_normalized = if signals.iso_avg > 0.0 {
            1.0 - ((signals.iso_avg - 100.0) / 3100.0).clamp(0.0, 1.0)
        } else {
            1.0 // No ISO data = assume good
        };
        let iso_score = iso_normalized * weights.iso_penalty;

        // Motion magnitude: 0-20 combined score
        let motion_normalized = (signals.motion_magnitude / 20.0).min(1.0);
        let motion_score = motion_normalized * weights.motion_magnitude;

        // Sum all weighted scores (should sum to ~1.0 if weights are normalized)
        let total = smoothness_score
            + pitch_score
            + yaw_score
            + speed_score
            + alt_score
            + iso_score
            + motion_score;

        // Clamp to 0-100 range for display
        (total * 100.0).clamp(0.0, 100.0)
    }

    /// Check if a segment passes the threshold requirements for a profile
    pub fn passes_thresholds(
        &self,
        profile_id: &str,
        signals: &SegmentSignals,
        duration_sec: f64,
    ) -> bool {
        let profile = match self.profiles.get(profile_id) {
            Some(p) => p,
            None => return false,
        };

        let thresholds = &profile.thresholds;

        // Check duration bounds
        if let Some(min_dur) = thresholds.min_duration_sec {
            if duration_sec < min_dur {
                return false;
            }
        }
        if let Some(max_dur) = thresholds.max_duration_sec {
            if duration_sec > max_dur {
                return false;
            }
        }

        // Check gimbal smoothness
        if let Some(min_smooth) = thresholds.min_gimbal_smoothness {
            if signals.gimbal_smoothness < min_smooth {
                return false;
            }
        }

        // Check ISO
        if let Some(max_iso) = thresholds.max_iso {
            if signals.iso_avg > max_iso {
                return false;
            }
        }

        // Check GPS speed
        if let Some(min_speed) = thresholds.min_gps_speed {
            if signals.gps_speed_avg < min_speed {
                return false;
            }
        }

        // Check motion magnitude
        if let Some(min_motion) = thresholds.min_motion_magnitude {
            if signals.motion_magnitude < min_motion {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile() -> Profile {
        Profile {
            id: "test".to_string(),
            name: "Test Profile".to_string(),
            description: "For testing".to_string(),
            weights: ProfileWeights {
                gimbal_smoothness: 0.3,
                gimbal_pitch_delta: 0.1,
                gimbal_yaw_delta: 0.1,
                gps_speed: 0.2,
                altitude_delta: 0.1,
                iso_penalty: 0.1,
                motion_magnitude: 0.1,
            },
            thresholds: ProfileThresholds::default(),
            preferences: HashMap::new(),
        }
    }

    #[test]
    fn test_score_calculation() {
        let mut calc = ScoreCalculator::new();
        calc.profiles.insert("test".to_string(), test_profile());

        let signals = SegmentSignals {
            gimbal_smoothness: 0.8,
            gimbal_pitch_delta_avg: 10.0,
            gimbal_yaw_delta_avg: 15.0,
            gps_speed_avg: 5.0,
            altitude_delta: 20.0,
            iso_avg: 200.0,
            motion_magnitude: 8.0,
        };

        let score = calc.calculate_score("test", &signals).unwrap();
        assert!(score > 0.0 && score <= 100.0);
    }

    #[test]
    fn test_thresholds() {
        let mut calc = ScoreCalculator::new();
        let mut profile = test_profile();
        profile.thresholds.min_gimbal_smoothness = Some(0.7);
        profile.thresholds.max_iso = Some(800.0);
        calc.profiles.insert("test".to_string(), profile);

        let good_signals = SegmentSignals {
            gimbal_smoothness: 0.8,
            iso_avg: 400.0,
            ..Default::default()
        };

        let bad_signals = SegmentSignals {
            gimbal_smoothness: 0.5, // Below threshold
            iso_avg: 1600.0,        // Above threshold
            ..Default::default()
        };

        assert!(calc.passes_thresholds("test", &good_signals, 10.0));
        assert!(!calc.passes_thresholds("test", &bad_signals, 10.0));
    }
}
