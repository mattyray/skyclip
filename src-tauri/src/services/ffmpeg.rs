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

    /// Check if a video file has an audio stream
    fn check_has_audio(&self, path: &str) -> bool {
        let output = Command::new(&self.ffprobe_path)
            .args([
                "-v", "quiet",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
            ])
            .arg(path)
            .output();

        match output {
            Ok(out) => !out.stdout.is_empty(),
            Err(_) => false, // Assume no audio if probe fails
        }
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

/// Clip definition for concatenation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcatClip {
    pub input_path: String,
    pub start_sec: f64,
    pub end_sec: f64,
    pub transition_type: String,      // "cut", "dissolve", "dip_black"
    pub transition_duration_ms: i64,
}

impl FFmpeg {
    /// Concatenate multiple clips with transitions into a single output
    pub fn concat_with_transitions(
        &self,
        clips: Vec<ConcatClip>,
        output_path: &str,
        use_hw_accel: bool,
    ) -> Result<()> {
        if clips.is_empty() {
            return Err(anyhow!("No clips to concatenate"));
        }

        // For a single clip or all "cut" transitions, use simple concat
        let has_transitions = clips.iter().skip(1).any(|c| c.transition_type != "cut");

        if clips.len() == 1 || !has_transitions {
            return self.concat_simple(&clips, output_path, use_hw_accel);
        }

        // Use filter_complex with xfade for transitions
        self.concat_with_xfade(&clips, output_path, use_hw_accel)
    }

    /// Simple concatenation without transitions (or all cuts)
    fn concat_simple(&self, clips: &[ConcatClip], output_path: &str, use_hw_accel: bool) -> Result<()> {
        // Create a temp file for the concat list
        let temp_dir = std::env::temp_dir();
        let concat_list_path = temp_dir.join(format!("skyclip_concat_{}.txt", uuid::Uuid::new_v4()));
        let temp_clips_dir = temp_dir.join(format!("skyclip_clips_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_clips_dir)?;

        // First, extract each segment to a temp file
        let mut temp_files = Vec::new();
        for (i, clip) in clips.iter().enumerate() {
            let temp_output = temp_clips_dir.join(format!("clip_{:04}.mp4", i));
            let start_time = format_time(clip.start_sec);
            let end_time = format_time(clip.end_sec);

            let mut args = vec![];
            if use_hw_accel {
                args.extend(["-hwaccel", "videotoolbox"]);
            }
            args.extend([
                "-ss", &start_time,
                "-to", &end_time,
                "-i", &clip.input_path,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-y",
            ]);

            let status = Command::new(&self.ffmpeg_path)
                .args(&args)
                .arg(&temp_output)
                .status()
                .context("Failed to extract clip segment")?;

            if !status.success() {
                // Clean up
                let _ = std::fs::remove_dir_all(&temp_clips_dir);
                return Err(anyhow!("Failed to extract clip segment {}", i));
            }

            temp_files.push(temp_output);
        }

        // Write concat list
        let concat_content: String = temp_files
            .iter()
            .map(|p| format!("file '{}'\n", p.to_string_lossy()))
            .collect();
        std::fs::write(&concat_list_path, &concat_content)?;

        // Concatenate
        let status = Command::new(&self.ffmpeg_path)
            .args([
                "-f", "concat",
                "-safe", "0",
                "-i",
            ])
            .arg(&concat_list_path)
            .args(["-c", "copy", "-y"])
            .arg(output_path)
            .status()
            .context("Failed to concatenate clips")?;

        // Clean up temp files
        let _ = std::fs::remove_file(&concat_list_path);
        let _ = std::fs::remove_dir_all(&temp_clips_dir);

        if !status.success() {
            return Err(anyhow!("FFmpeg concatenation failed"));
        }

        Ok(())
    }

    /// Concatenation with xfade transitions
    fn concat_with_xfade(&self, clips: &[ConcatClip], output_path: &str, use_hw_accel: bool) -> Result<()> {
        // First extract all clips to temp files with consistent encoding
        let temp_dir = std::env::temp_dir();
        let temp_clips_dir = temp_dir.join(format!("skyclip_xfade_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_clips_dir)?;

        let mut temp_files = Vec::new();
        let mut clip_durations = Vec::new();

        for (i, clip) in clips.iter().enumerate() {
            let temp_output = temp_clips_dir.join(format!("clip_{:04}.mp4", i));
            let duration = clip.end_sec - clip.start_sec;
            clip_durations.push(duration);
            let start_time = format_time(clip.start_sec);
            let end_time = format_time(clip.end_sec);

            // First check if source has audio
            let has_audio = self.check_has_audio(&clip.input_path);

            let mut args = vec![];
            if use_hw_accel {
                args.extend(["-hwaccel", "videotoolbox"]);
            }
            args.extend([
                "-ss", &start_time,
                "-to", &end_time,
                "-i", &clip.input_path,
            ]);

            // If no audio, add silent audio source
            if !has_audio {
                let duration_str = format!("{}", duration);
                args.extend([
                    "-f", "lavfi",
                    "-i", &format!("anullsrc=channel_layout=stereo:sample_rate=48000:duration={}", duration_str),
                ]);
            }

            args.extend([
                "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=30",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                "-shortest",
                "-y",
            ]);

            let status = Command::new(&self.ffmpeg_path)
                .args(&args)
                .arg(&temp_output)
                .status()
                .context("Failed to extract clip segment for xfade")?;

            if !status.success() {
                let _ = std::fs::remove_dir_all(&temp_clips_dir);
                return Err(anyhow!("Failed to extract clip {} for xfade", i));
            }

            temp_files.push(temp_output);
        }

        // Build the filter_complex string
        let mut filter_parts = Vec::new();
        let mut video_labels = Vec::new();
        let mut audio_labels = Vec::new();

        // Input labels
        for i in 0..temp_files.len() {
            video_labels.push(format!("[{}:v]", i));
            audio_labels.push(format!("[{}:a]", i));
        }

        // Build xfade chain for video
        if clips.len() >= 2 {
            let mut current_video = video_labels[0].clone();
            let mut cumulative_duration = clip_durations[0];

            for i in 1..clips.len() {
                let transition = &clips[i].transition_type;
                let trans_duration_sec = clips[i].transition_duration_ms as f64 / 1000.0;
                let trans_duration_sec = trans_duration_sec.max(0.1).min(2.0); // Clamp between 0.1 and 2 seconds

                // Calculate offset (when transition starts)
                let offset = (cumulative_duration - trans_duration_sec).max(0.0);

                let xfade_transition = match transition.as_str() {
                    "dissolve" => "fade",
                    "dip_black" => "fadeblack",
                    _ => "fade", // Default to fade for unknown transitions
                };

                let out_label = format!("[v{}]", i);
                filter_parts.push(format!(
                    "{}{}xfade=transition={}:duration={}:offset={}{}",
                    current_video,
                    video_labels[i],
                    xfade_transition,
                    trans_duration_sec,
                    offset,
                    out_label
                ));

                current_video = out_label;
                cumulative_duration = offset + clip_durations[i];
            }

            // Audio crossfade
            let mut current_audio = audio_labels[0].clone();
            cumulative_duration = clip_durations[0];

            for i in 1..clips.len() {
                let trans_duration_sec = clips[i].transition_duration_ms as f64 / 1000.0;
                let trans_duration_sec = trans_duration_sec.max(0.1).min(2.0);
                let offset = (cumulative_duration - trans_duration_sec).max(0.0);

                let out_label = format!("[a{}]", i);
                filter_parts.push(format!(
                    "{}{}acrossfade=d={}:c1=tri:c2=tri{}",
                    current_audio,
                    audio_labels[i],
                    trans_duration_sec,
                    out_label
                ));

                current_audio = out_label;
                cumulative_duration = offset + clip_durations[i];
            }

            let filter_complex = filter_parts.join(";");
            let final_video = format!("[v{}]", clips.len() - 1);
            let final_audio = format!("[a{}]", clips.len() - 1);

            // Build ffmpeg command
            let mut cmd = Command::new(&self.ffmpeg_path);

            for temp_file in &temp_files {
                cmd.arg("-i").arg(temp_file);
            }

            let status = cmd
                .args([
                    "-filter_complex", &filter_complex,
                    "-map", &final_video,
                    "-map", &final_audio,
                    "-c:v", "libx264",
                    "-crf", "18",
                    "-preset", "fast",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-y",
                ])
                .arg(output_path)
                .status()
                .context("Failed to execute ffmpeg with xfade")?;

            // Clean up
            let _ = std::fs::remove_dir_all(&temp_clips_dir);

            if !status.success() {
                return Err(anyhow!("FFmpeg xfade concatenation failed"));
            }
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
    // Common installation paths for ffmpeg on macOS
    let common_paths = [
        format!("/opt/homebrew/bin/{}", name),      // Apple Silicon Homebrew
        format!("/usr/local/bin/{}", name),          // Intel Homebrew / manual install
        format!("/usr/bin/{}", name),                // System install
        format!("/opt/local/bin/{}", name),          // MacPorts
    ];

    // First check common paths directly (works when launched from Finder)
    for path in &common_paths {
        if std::path::Path::new(path).exists() {
            return Ok(path.clone());
        }
    }

    // Fallback to which command (works when launched from terminal)
    let output = Command::new("which")
        .arg(name)
        .output()
        .context(format!("Failed to find {name}"))?;

    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(path);
        }
    }

    Err(anyhow!(
        "{} not found. Please install FFmpeg via: brew install ffmpeg",
        name
    ))
}

fn format_time(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = seconds % 60.0;
    format!("{:02}:{:02}:{:06.3}", hours, minutes, secs)
}
