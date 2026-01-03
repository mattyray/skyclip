use crate::models::TelemetryFrame;
use anyhow::{Context, Result};
use chrono::{NaiveDateTime, TimeZone, Utc};
use regex::Regex;
use std::fs;
use std::path::Path;

pub struct SrtParser {
    // Regex patterns for parsing SRT content
    time_pattern: Regex,
    iso_pattern: Regex,
    shutter_pattern: Regex,
    fnum_pattern: Regex,
    ev_pattern: Regex,
    ct_pattern: Regex,
    color_md_pattern: Regex,
    focal_len_pattern: Regex,
    latitude_pattern: Regex,
    longitude_pattern: Regex,
    altitude_pattern: Regex,
    gb_yaw_pattern: Regex,
    gb_pitch_pattern: Regex,
    gb_roll_pattern: Regex,
    timestamp_pattern: Regex,
    srt_cnt_pattern: Regex,
}

impl SrtParser {
    pub fn new() -> Self {
        Self {
            time_pattern: Regex::new(r"(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})").unwrap(),
            iso_pattern: Regex::new(r"\[iso\s*:\s*(\d+)\]").unwrap(),
            shutter_pattern: Regex::new(r"\[shutter\s*:\s*([^\]]+)\]").unwrap(),
            fnum_pattern: Regex::new(r"\[fnum\s*:\s*(\d+)\]").unwrap(),
            ev_pattern: Regex::new(r"\[ev\s*:\s*([^\]]+)\]").unwrap(),
            ct_pattern: Regex::new(r"\[ct\s*:\s*(\d+)\]").unwrap(),
            color_md_pattern: Regex::new(r"\[color_md\s*:\s*([^\]]+)\]").unwrap(),
            focal_len_pattern: Regex::new(r"\[focal_len\s*:\s*([^\]]+)\]").unwrap(),
            latitude_pattern: Regex::new(r"\[latitude\s*:\s*([^\]]+)\]").unwrap(),
            longitude_pattern: Regex::new(r"\[longitude\s*:\s*([^\]]+)\]").unwrap(),
            // Match both "altitude: X" and "rel_alt: X abs_alt: Y" formats
            altitude_pattern: Regex::new(r"\[(?:altitude|rel_alt)\s*:\s*([^\]\s]+)").unwrap(),
            gb_yaw_pattern: Regex::new(r"\[gb_yaw\s*:\s*([^\]]+)\]").unwrap(),
            gb_pitch_pattern: Regex::new(r"\[gb_pitch\s*:\s*([^\]]+)\]").unwrap(),
            gb_roll_pattern: Regex::new(r"\[gb_roll\s*:\s*([^\]]+)\]").unwrap(),
            timestamp_pattern: Regex::new(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})").unwrap(),
            srt_cnt_pattern: Regex::new(r"SrtCnt\s*:\s*(\d+)").unwrap(),
        }
    }

    /// Parse an SRT file and return a vector of telemetry frames
    pub fn parse_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<TelemetryFrame>> {
        let content = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read SRT file: {:?}", path.as_ref()))?;

        self.parse_content(&content)
    }

    /// Parse SRT content string and return telemetry frames
    pub fn parse_content(&self, content: &str) -> Result<Vec<TelemetryFrame>> {
        let mut frames = Vec::new();

        // Split by blank lines to get individual subtitle blocks
        let blocks: Vec<&str> = content.split("\n\n").collect();

        for block in blocks {
            if block.trim().is_empty() {
                continue;
            }

            if let Some(frame) = self.parse_block(block) {
                frames.push(frame);
            }
        }

        Ok(frames)
    }

    fn parse_block(&self, block: &str) -> Option<TelemetryFrame> {
        let lines: Vec<&str> = block.lines().collect();
        if lines.len() < 3 {
            return None;
        }

        // First line is the index
        let index: u32 = lines[0].trim().parse().ok()?;

        // Second line is the timecode
        let (start_ms, end_ms) = self.parse_timecode(lines[1])?;

        let mut frame = TelemetryFrame::new(index, start_ms, end_ms);

        // Rest is the content with telemetry data
        let content = lines[2..].join("\n");

        // Extract SrtCnt if present (validation)
        if let Some(caps) = self.srt_cnt_pattern.captures(&content) {
            let _srt_cnt: u32 = caps[1].parse().unwrap_or(0);
        }

        // Extract timestamp
        if let Some(caps) = self.timestamp_pattern.captures(&content) {
            if let Ok(dt) = NaiveDateTime::parse_from_str(&caps[1], "%Y-%m-%d %H:%M:%S%.3f") {
                frame.timestamp = Some(Utc.from_utc_datetime(&dt));
            }
        }

        // Extract camera settings
        if let Some(caps) = self.iso_pattern.captures(&content) {
            frame.iso = caps[1].parse().ok();
        }
        if let Some(caps) = self.shutter_pattern.captures(&content) {
            frame.shutter = Some(caps[1].trim().to_string());
        }
        if let Some(caps) = self.fnum_pattern.captures(&content) {
            frame.fnum = caps[1].parse().ok();
        }
        if let Some(caps) = self.ev_pattern.captures(&content) {
            frame.ev = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.ct_pattern.captures(&content) {
            frame.color_temp = caps[1].parse().ok();
        }
        if let Some(caps) = self.color_md_pattern.captures(&content) {
            frame.color_mode = Some(caps[1].trim().to_string());
        }
        if let Some(caps) = self.focal_len_pattern.captures(&content) {
            frame.focal_len = caps[1].trim().parse().ok();
        }

        // Extract GPS data
        if let Some(caps) = self.latitude_pattern.captures(&content) {
            frame.latitude = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.longitude_pattern.captures(&content) {
            frame.longitude = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.altitude_pattern.captures(&content) {
            frame.altitude = caps[1].trim().parse().ok();
        }

        // Extract gimbal orientation
        if let Some(caps) = self.gb_yaw_pattern.captures(&content) {
            frame.gimbal_yaw = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.gb_pitch_pattern.captures(&content) {
            frame.gimbal_pitch = caps[1].trim().parse().ok();
        }
        if let Some(caps) = self.gb_roll_pattern.captures(&content) {
            frame.gimbal_roll = caps[1].trim().parse().ok();
        }

        Some(frame)
    }

    fn parse_timecode(&self, line: &str) -> Option<(i64, i64)> {
        let caps = self.time_pattern.captures(line)?;

        let start_h: i64 = caps[1].parse().ok()?;
        let start_m: i64 = caps[2].parse().ok()?;
        let start_s: i64 = caps[3].parse().ok()?;
        let start_ms: i64 = caps[4].parse().ok()?;

        let end_h: i64 = caps[5].parse().ok()?;
        let end_m: i64 = caps[6].parse().ok()?;
        let end_s: i64 = caps[7].parse().ok()?;
        let end_ms: i64 = caps[8].parse().ok()?;

        let start = start_h * 3600000 + start_m * 60000 + start_s * 1000 + start_ms;
        let end = end_h * 3600000 + end_m * 60000 + end_s * 1000 + end_ms;

        Some((start, end))
    }
}

impl Default for SrtParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_srt_block() {
        let content = r#"1
00:00:00,000 --> 00:00:01,000
<font size="28">SrtCnt : 1, DiffTime : 1000ms
2025-12-23 14:32:15.123
[iso : 100] [shutter : 1/500.0] [fnum : 280] [ev : 0]
[ct : 5500] [color_md : default] [focal_len : 24.00]
[latitude : 40.7128] [longitude : -74.0060] [altitude: 150.0]
[gb_yaw : 45.2] [gb_pitch : -15.3] [gb_roll : 0.1]</font>"#;

        let parser = SrtParser::new();
        let frames = parser.parse_content(content).unwrap();

        assert_eq!(frames.len(), 1);
        let frame = &frames[0];

        assert_eq!(frame.index, 1);
        assert_eq!(frame.start_time_ms, 0);
        assert_eq!(frame.end_time_ms, 1000);
        assert_eq!(frame.iso, Some(100));
        assert_eq!(frame.shutter, Some("1/500.0".to_string()));
        assert_eq!(frame.fnum, Some(280));
        assert_eq!(frame.latitude, Some(40.7128));
        assert_eq!(frame.longitude, Some(-74.0060));
        assert_eq!(frame.altitude, Some(150.0));
        assert_eq!(frame.gimbal_yaw, Some(45.2));
        assert_eq!(frame.gimbal_pitch, Some(-15.3));
        assert_eq!(frame.gimbal_roll, Some(0.1));
    }
}
