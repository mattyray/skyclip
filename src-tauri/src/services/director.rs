use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Segment info passed to Claude for editing decisions
#[derive(Debug, Clone, Serialize)]
pub struct SegmentContext {
    pub id: String,
    pub duration_sec: f64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub gimbal_pitch_delta: f64,
    pub gimbal_yaw_delta: f64,
    pub gimbal_smoothness: f64,
    pub gps_speed: f64,
    pub altitude_delta: f64,
    pub score: f64,
}

/// Claude's response for a single clip decision
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DirectorClipDecision {
    pub segment_id: String,
    pub sequence_order: i32,
    pub in_point_ms: i64,
    pub out_point_ms: i64,
    pub transition_to_next: String,
    pub transition_duration_ms: i64,
    pub reasoning: String,
}

/// Claude's full response
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DirectorResponse {
    pub edit_sequence: Vec<DirectorClipDecision>,
    pub total_duration_sec: f64,
    pub style_notes: String,
}

/// Request body for Claude API
#[derive(Debug, Serialize)]
struct ClaudeRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<ClaudeMessage>,
}

#[derive(Debug, Serialize)]
struct ClaudeMessage {
    role: String,
    content: Vec<ClaudeContent>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ClaudeContent {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
}

#[derive(Debug, Serialize)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

/// Response from Claude API
#[derive(Debug, Deserialize)]
struct ClaudeApiResponse {
    content: Vec<ClaudeResponseContent>,
}

#[derive(Debug, Deserialize)]
struct ClaudeResponseContent {
    text: Option<String>,
}

pub struct Director {
    api_key: String,
}

impl Director {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }

    /// Generate an edit sequence based on the director's natural language instructions
    pub async fn generate_edit(
        &self,
        prompt: &str,
        segments: Vec<SegmentContext>,
        thumbnail_paths: Vec<String>,
        target_duration_sec: Option<f64>,
    ) -> Result<DirectorResponse> {
        // Build the content array
        let mut content: Vec<ClaudeContent> = Vec::new();

        // System context and user prompt
        let system_text = self.build_system_prompt(&segments, target_duration_sec);
        let user_text = format!(
            "{}\n\nDIRECTOR'S INSTRUCTIONS:\n\"{}\"",
            system_text, prompt
        );
        content.push(ClaudeContent::Text { text: user_text });

        // Add thumbnails as base64 images (limit to first 20 to control costs)
        let max_thumbnails = 20.min(thumbnail_paths.len());
        for (i, thumb_path) in thumbnail_paths.iter().take(max_thumbnails).enumerate() {
            if let Ok(image_data) = self.load_thumbnail_as_base64(thumb_path) {
                content.push(ClaudeContent::Text {
                    text: format!("Segment {} thumbnail (ID: {}):", i + 1, segments.get(i).map(|s| s.id.as_str()).unwrap_or("unknown")),
                });
                content.push(ClaudeContent::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: "image/jpeg".to_string(),
                        data: image_data,
                    },
                });
            }
        }

        // Add response format instructions
        content.push(ClaudeContent::Text {
            text: self.get_response_format_instructions(),
        });

        let request = ClaudeRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 4096,
            messages: vec![ClaudeMessage {
                role: "user".to_string(),
                content,
            }],
        };

        // Make API call
        let client = reqwest::Client::new();
        let response = client
            .post("https://api.anthropic.com/v1/messages")
            .header("Content-Type", "application/json")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to call Claude API: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("Claude API error ({}): {}", status, error_text));
        }

        let api_response: ClaudeApiResponse = response
            .json()
            .await
            .map_err(|e| anyhow!("Failed to parse Claude response: {}", e))?;

        // Extract text response
        let response_text = api_response
            .content
            .iter()
            .find_map(|c| c.text.clone())
            .ok_or_else(|| anyhow!("No text in Claude response"))?;

        // Parse JSON from response
        self.parse_director_response(&response_text)
    }

    fn build_system_prompt(&self, segments: &[SegmentContext], target_duration: Option<f64>) -> String {
        let segments_json = serde_json::to_string_pretty(segments).unwrap_or_default();

        let duration_instruction = if let Some(dur) = target_duration {
            format!("Target duration: {} seconds. Select and trim clips to achieve this.", dur)
        } else {
            "No specific duration target - include what best matches the vision.".to_string()
        };

        format!(
            r#"You are a professional drone footage editor creating highlight reels. You'll receive:
1. Segment metadata (telemetry data from the drone)
2. Thumbnail images showing what each segment looks like

Your job is to select clips, determine their order, choose transitions, and set in/out points to create a cohesive edit.

AVAILABLE SEGMENTS:
{}

{}

EDITING GUIDELINES:
- gimbal_pitch_delta: Negative = tilting down (reveals), Positive = tilting up
- gimbal_yaw_delta: Positive = panning right, Negative = panning left
- gimbal_smoothness: 0-1, higher = smoother camera movement (better for cinematic)
- gps_speed: m/s, higher = faster drone movement (good for action)
- altitude_delta: Positive = ascending, Negative = descending
- score: Pre-computed quality score (0-100)

TRANSITION TYPES:
- "cut" - Instant cut (good for matching motion, action sequences)
- "dissolve" - Crossfade (good for different scenes, time passing, cinematic feel)
- "dip_black" - Fade through black (good for major scene changes, dramatic moments)

GENERAL PRINCIPLES:
- Match motion direction between clips when using cuts
- Use dissolves when scene content is very different
- Start with establishing shots, end with resolution
- Build energy through the edit
- Smoother gimbal = longer holds; jerky = quicker cuts"#,
            segments_json, duration_instruction
        )
    }

    fn get_response_format_instructions(&self) -> String {
        r#"

Respond with ONLY valid JSON in this exact format (no markdown, no explanation outside JSON):
{
  "edit_sequence": [
    {
      "segment_id": "the segment id",
      "sequence_order": 1,
      "in_point_ms": 0,
      "out_point_ms": 5000,
      "transition_to_next": "dissolve",
      "transition_duration_ms": 500,
      "reasoning": "Brief explanation of why this clip and edit choice"
    }
  ],
  "total_duration_sec": 30.0,
  "style_notes": "Overall notes about the edit"
}"#.to_string()
    }

    fn load_thumbnail_as_base64(&self, path: &str) -> Result<String> {
        let path = Path::new(path);
        if !path.exists() {
            return Err(anyhow!("Thumbnail not found: {}", path.display()));
        }

        let bytes = fs::read(path)?;
        Ok(BASE64.encode(&bytes))
    }

    fn parse_director_response(&self, response: &str) -> Result<DirectorResponse> {
        // Try to find JSON in the response (Claude sometimes adds text around it)
        let json_start = response.find('{');
        let json_end = response.rfind('}');

        match (json_start, json_end) {
            (Some(start), Some(end)) if end > start => {
                let json_str = &response[start..=end];
                serde_json::from_str(json_str)
                    .map_err(|e| anyhow!("Failed to parse director response JSON: {}\n\nRaw response:\n{}", e, json_str))
            }
            _ => Err(anyhow!("No valid JSON found in response:\n{}", response)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_response() {
        let director = Director::new("test".to_string());
        let response = r#"{
            "edit_sequence": [
                {
                    "segment_id": "seg_001",
                    "sequence_order": 1,
                    "in_point_ms": 0,
                    "out_point_ms": 5000,
                    "transition_to_next": "dissolve",
                    "transition_duration_ms": 500,
                    "reasoning": "Test"
                }
            ],
            "total_duration_sec": 5.0,
            "style_notes": "Test notes"
        }"#;

        let result = director.parse_director_response(response);
        assert!(result.is_ok());
        let parsed = result.unwrap();
        assert_eq!(parsed.edit_sequence.len(), 1);
        assert_eq!(parsed.edit_sequence[0].segment_id, "seg_001");
    }
}
