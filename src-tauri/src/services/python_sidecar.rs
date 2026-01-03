use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::{Command, Stdio};
use std::io::Write;

/// Visual analysis results from Python sidecar
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAnalysis {
    pub motion: Option<MotionAnalysis>,
    pub scene: Option<SceneAnalysis>,
    pub color: Option<ColorAnalysis>,
    pub objects: Option<ObjectAnalysis>,
    pub semantic: Option<SemanticAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionAnalysis {
    pub avg_magnitude: f64,
    pub peak_magnitude: f64,
    pub peak_frame: i64,
    pub dominant_direction: f64,
    pub motion_consistency: f64,
    pub action_peaks: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneChange {
    pub frame: i64,
    pub timestamp_ms: i64,
    #[serde(rename = "type")]
    pub transition_type: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneAnalysis {
    pub scene_changes: Vec<SceneChange>,
    pub avg_scene_duration_ms: f64,
    pub is_single_shot: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorAnalysis {
    pub dominant_colors: Vec<(u8, u8, u8)>,
    pub color_weights: Vec<f64>,
    pub avg_brightness: f64,
    pub avg_saturation: f64,
    pub is_low_light: bool,
    pub is_golden_hour: bool,
    pub color_consistency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectAnalysis {
    pub primary_subject: Option<String>,
    pub subject_entry_direction: Option<String>,
    pub subject_exit_direction: Option<String>,
    pub avg_subjects_per_frame: f64,
    pub has_consistent_subject: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    pub scene_type: String,
    pub top_descriptions: Vec<(String, f64)>,
    pub embedding_size: usize,
}

/// Edit decision from the suggestion engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditDecision {
    pub clip_id: String,
    pub sequence_order: i32,
    pub adjusted_start_ms: i64,
    pub adjusted_end_ms: i64,
    pub transition_type: String,
    pub transition_duration_ms: i64,
    pub confidence: f64,
    pub reasoning: String,
}

/// Complete edit sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditSequence {
    pub decisions: Vec<EditDecision>,
    pub total_duration_ms: i64,
    pub style: String,
    pub was_reordered: bool,
}

/// Clip info for edit sequence generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipInfo {
    pub clip_id: String,
    pub video_path: String,
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Python sidecar for visual analysis
pub struct PythonSidecar {
    python_path: String,
    script_dir: String,
}

impl PythonSidecar {
    pub fn new() -> Result<Self, String> {
        // Find Python executable
        let python_path = Self::find_python()?;

        // Find the script directory (relative to the executable or in dev mode)
        let script_dir = Self::find_script_dir()?;

        Ok(Self {
            python_path,
            script_dir,
        })
    }

    fn find_python() -> Result<String, String> {
        // Try common Python paths
        let candidates = [
            "python3",
            "python",
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
        ];

        for candidate in candidates {
            let output = Command::new(candidate)
                .arg("--version")
                .output();

            if let Ok(output) = output {
                if output.status.success() {
                    return Ok(candidate.to_string());
                }
            }
        }

        Err("Python 3 not found. Please install Python 3.11+".to_string())
    }

    fn find_script_dir() -> Result<String, String> {
        // In development, look for python/ directory relative to project root
        let dev_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|p| p.join("python"));

        if let Some(path) = dev_path {
            if path.exists() {
                return Ok(path.to_string_lossy().to_string());
            }
        }

        // In production, look in Resources
        if let Ok(exe_path) = std::env::current_exe() {
            let resources = exe_path
                .parent()
                .and_then(|p| p.parent())
                .map(|p| p.join("Resources").join("python"));

            if let Some(path) = resources {
                if path.exists() {
                    return Ok(path.to_string_lossy().to_string());
                }
            }
        }

        Err("Python scripts not found".to_string())
    }

    fn run_command(&self, command: &str, args: serde_json::Value) -> Result<serde_json::Value, String> {
        let request = serde_json::json!({
            "command": command,
            "args": args
        });

        let mut child = Command::new(&self.python_path)
            .arg("-m")
            .arg("skyclip_analyzer.cli")
            .current_dir(&self.script_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn Python: {}", e))?;

        // Write request to stdin
        if let Some(ref mut stdin) = child.stdin {
            stdin
                .write_all(request.to_string().as_bytes())
                .map_err(|e| format!("Failed to write to Python stdin: {}", e))?;
        }

        let output = child
            .wait_with_output()
            .map_err(|e| format!("Failed to wait for Python: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Python error: {}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let response: serde_json::Value = serde_json::from_str(&stdout)
            .map_err(|e| format!("Failed to parse Python response: {} - {}", e, stdout))?;

        if response.get("success").and_then(|v| v.as_bool()) == Some(true) {
            Ok(response["result"].clone())
        } else {
            Err(response["error"]
                .as_str()
                .unwrap_or("Unknown error")
                .to_string())
        }
    }

    /// Analyze a video clip for motion, scene, and color
    pub fn analyze_clip(
        &self,
        video_path: &str,
        start_ms: i64,
        end_ms: Option<i64>,
        include_objects: bool,
        include_semantic: bool,
    ) -> Result<VisualAnalysis, String> {
        let args = serde_json::json!({
            "video_path": video_path,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "include_objects": include_objects,
            "include_semantic": include_semantic
        });

        let result = self.run_command("analyze_clip", args)?;

        // Parse individual components
        let motion: Option<MotionAnalysis> = result
            .get("motion")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let scene: Option<SceneAnalysis> = result
            .get("scene")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let color: Option<ColorAnalysis> = result
            .get("color")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let objects: Option<ObjectAnalysis> = result
            .get("objects")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let semantic: Option<SemanticAnalysis> = result
            .get("semantic")
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        Ok(VisualAnalysis {
            motion,
            scene,
            color,
            objects,
            semantic,
        })
    }

    /// Generate edit sequence for multiple clips
    pub fn generate_edit_sequence(
        &self,
        clips: Vec<ClipInfo>,
        style: &str,
        reorder: bool,
        full_analysis: bool,
    ) -> Result<EditSequence, String> {
        let args = serde_json::json!({
            "clips": clips,
            "style": style,
            "reorder": reorder,
            "full_analysis": full_analysis
        });

        let result = self.run_command("generate_edit_sequence", args)?;
        serde_json::from_value(result).map_err(|e| format!("Failed to parse edit sequence: {}", e))
    }

    /// Suggest transition between two clips
    pub fn suggest_transition(
        &self,
        clip_a: ClipInfo,
        clip_b: ClipInfo,
        style: &str,
    ) -> Result<(String, i64, f64, String), String> {
        let args = serde_json::json!({
            "clip_a": clip_a,
            "clip_b": clip_b,
            "style": style
        });

        let result = self.run_command("suggest_transition", args)?;

        Ok((
            result["transition_type"]
                .as_str()
                .unwrap_or("cut")
                .to_string(),
            result["transition_duration_ms"].as_i64().unwrap_or(0),
            result["confidence"].as_f64().unwrap_or(0.5),
            result["reasoning"]
                .as_str()
                .unwrap_or("")
                .to_string(),
        ))
    }

    /// Check if Python sidecar is available
    pub fn is_available(&self) -> bool {
        let result = Command::new(&self.python_path)
            .arg("-c")
            .arg("import cv2; print('ok')")
            .output();

        matches!(result, Ok(output) if output.status.success())
    }

    /// Install Python dependencies
    pub fn install_dependencies(&self) -> Result<(), String> {
        let requirements_path = Path::new(&self.script_dir).join("requirements.txt");

        let output = Command::new(&self.python_path)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("-r")
            .arg(&requirements_path)
            .output()
            .map_err(|e| format!("Failed to run pip: {}", e))?;

        if output.status.success() {
            Ok(())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).to_string())
        }
    }
}

impl Default for PythonSidecar {
    fn default() -> Self {
        Self::new().expect("Failed to initialize Python sidecar")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_python() {
        let result = PythonSidecar::find_python();
        // This test will pass if Python is installed
        if result.is_ok() {
            println!("Found Python at: {}", result.unwrap());
        }
    }
}
