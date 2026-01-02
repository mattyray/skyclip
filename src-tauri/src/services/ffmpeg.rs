use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoInfo {
    pub duration_sec: f64,
    pub width: i32,
    pub height: i32,
    pub framerate: f64,
    pub codec: String,
}

pub struct FFmpeg {
    ffmpeg_path: String,
    ffprobe_path: String,
}

impl FFmpeg {
    pub fn new() -> Result<Self> {
        // Try to find ffmpeg and ffprobe in PATH
        let ffmpeg_path = which_command("ffmpeg")?;
        let ffprobe_path = which_command("ffprobe")?;

        Ok(Self {
            ffmpeg_path,
            ffprobe_path,
        })
    }

    /// Get video information using ffprobe
    pub fn probe<P: AsRef<Path>>(&self, input: P) -> Result<VideoInfo> {
        let output = Command::new(&self.ffprobe_path)
            .args([
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                "-select_streams", "v:0",
            ])
            .arg(input.as_ref())
            .output()
            .context("Failed to execute ffprobe")?;

        if !output.status.success() {
            return Err(anyhow!(
                "ffprobe failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;

        // Extract video stream info
        let stream = json["streams"]
            .as_array()
            .and_then(|s| s.first())
            .ok_or_else(|| anyhow!("No video stream found"))?;

        let width = stream["width"].as_i64().unwrap_or(0) as i32;
        let height = stream["height"].as_i64().unwrap_or(0) as i32;
        let codec = stream["codec_name"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        // Parse framerate (e.g., "30000/1001" or "30")
        let framerate = self.parse_framerate(
            stream["r_frame_rate"]
                .as_str()
                .or_else(|| stream["avg_frame_rate"].as_str())
                .unwrap_or("0"),
        );

        // Get duration from format
        let duration_sec = json["format"]["duration"]
            .as_str()
            .and_then(|d| d.parse::<f64>().ok())
            .unwrap_or(0.0);

        Ok(VideoInfo {
            duration_sec,
            width,
            height,
            framerate,
            codec,
        })
    }

    fn parse_framerate(&self, fps_str: &str) -> f64 {
        if fps_str.contains('/') {
            let parts: Vec<&str> = fps_str.split('/').collect();
            if parts.len() == 2 {
                let num: f64 = parts[0].parse().unwrap_or(0.0);
                let den: f64 = parts[1].parse().unwrap_or(1.0);
                if den != 0.0 {
                    return num / den;
                }
            }
        }
        fps_str.parse().unwrap_or(0.0)
    }

    /// Generate a proxy video with hardware acceleration
    pub fn generate_proxy<P: AsRef<Path>>(&self, input: P, output: P) -> Result<()> {
        let status = Command::new(&self.ffmpeg_path)
            .args([
                "-hwaccel", "videotoolbox",
                "-i",
            ])
            .arg(input.as_ref())
            .args([
                "-vf", "scale=1280:720",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-y", // Overwrite output
            ])
            .arg(output.as_ref())
            .status()
            .context("Failed to execute ffmpeg")?;

        if !status.success() {
            return Err(anyhow!("FFmpeg proxy generation failed"));
        }

        Ok(())
    }

    /// Extract thumbnails at 1fps
    pub fn extract_thumbnails<P: AsRef<Path>>(
        &self,
        input: P,
        output_dir: P,
        prefix: &str,
    ) -> Result<Vec<String>> {
        let output_pattern = output_dir
            .as_ref()
            .join(format!("{prefix}_%04d.jpg"));

        let status = Command::new(&self.ffmpeg_path)
            .args([
                "-hwaccel", "videotoolbox",
                "-i",
            ])
            .arg(input.as_ref())
            .args([
                "-vf", "fps=1,scale=320:180",
                "-q:v", "3",
                "-y",
            ])
            .arg(&output_pattern)
            .status()
            .context("Failed to execute ffmpeg for thumbnails")?;

        if !status.success() {
            return Err(anyhow!("FFmpeg thumbnail extraction failed"));
        }

        // List generated thumbnails
        let mut thumbnails = Vec::new();
        let output_dir = output_dir.as_ref();
        if output_dir.is_dir() {
            for entry in std::fs::read_dir(output_dir)? {
                let entry = entry?;
                let filename = entry.file_name().to_string_lossy().to_string();
                if filename.starts_with(prefix) && filename.ends_with(".jpg") {
                    thumbnails.push(entry.path().to_string_lossy().to_string());
                }
            }
        }
        thumbnails.sort();

        Ok(thumbnails)
    }

    /// Fast export using stream copy (keyframe-aligned cuts)
    pub fn export_fast<P: AsRef<Path>>(
        &self,
        input: P,
        output: P,
        start_sec: f64,
        end_sec: f64,
    ) -> Result<()> {
        let start_time = format_time(start_sec);
        let end_time = format_time(end_sec);

        let status = Command::new(&self.ffmpeg_path)
            .args(["-ss", &start_time, "-to", &end_time, "-i"])
            .arg(input.as_ref())
            .args([
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-y",
            ])
            .arg(output.as_ref())
            .status()
            .context("Failed to execute ffmpeg for export")?;

        if !status.success() {
            return Err(anyhow!("FFmpeg fast export failed"));
        }

        Ok(())
    }

    /// Precise export with re-encode (frame-exact cuts)
    pub fn export_precise<P: AsRef<Path>>(
        &self,
        input: P,
        output: P,
        start_sec: f64,
        end_sec: f64,
    ) -> Result<()> {
        let start_time = format_time(start_sec);
        let end_time = format_time(end_sec);

        let status = Command::new(&self.ffmpeg_path)
            .args([
                "-hwaccel", "videotoolbox",
                "-ss", &start_time,
                "-to", &end_time,
                "-i",
            ])
            .arg(input.as_ref())
            .args([
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-c:a", "aac",
                "-b:a", "192k",
                "-y",
            ])
            .arg(output.as_ref())
            .status()
            .context("Failed to execute ffmpeg for precise export")?;

        if !status.success() {
            return Err(anyhow!("FFmpeg precise export failed"));
        }

        Ok(())
    }
}

impl Default for FFmpeg {
    fn default() -> Self {
        Self::new().expect("FFmpeg not found in PATH")
    }
}

fn which_command(name: &str) -> Result<String> {
    let output = Command::new("which")
        .arg(name)
        .output()
        .context(format!("Failed to find {name}"))?;

    if !output.status.success() {
        return Err(anyhow!("{} not found in PATH", name));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn format_time(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = seconds % 60.0;
    format!("{:02}:{:02}:{:06.3}", hours, minutes, secs)
}
