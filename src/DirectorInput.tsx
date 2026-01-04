import { useState, useCallback } from "react";
import { invoke } from "@tauri-apps/api/core";

interface SegmentData {
  id: string;
  start_ms: number;
  end_ms: number;
  thumbnail_path: string | null;
  gimbal_pitch_delta: number | null;
  gimbal_yaw_delta: number | null;
  gimbal_smoothness: number | null;
  gps_speed: number | null;
  altitude_delta: number | null;
  score: number;
}

interface EditSequence {
  decisions: any[];
  total_duration_ms: number;
  style: string;
  was_reordered: boolean;
}

interface DirectorInputProps {
  segments: SegmentData[];
  onSequenceGenerated: (sequence: EditSequence) => void;
  onError: (error: string) => void;
  disabled?: boolean;
}

export function DirectorInput({ segments, onSequenceGenerated, onError, disabled }: DirectorInputProps) {
  const [prompt, setPrompt] = useState("");
  const [targetDuration, setTargetDuration] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim() || segments.length < 2) return;

    setIsGenerating(true);

    try {
      const sequence = await invoke<EditSequence>("director_generate_edit", {
        prompt: prompt.trim(),
        segments,
        targetDurationSec: targetDuration ? parseInt(targetDuration) : null,
      });

      onSequenceGenerated(sequence);
      setPrompt("");
    } catch (e) {
      onError(`AI Director failed: ${e}`);
    } finally {
      setIsGenerating(false);
    }
  }, [prompt, segments, targetDuration, onSequenceGenerated, onError]);

  return (
    <div className="director-controls">
      <textarea
        placeholder="Describe your vision... e.g., 'Make a 30-second dramatic sunset reveal, start with an establishing shot, build to the most exciting moment, end on a calm beach scene. Cinematic feel with smooth transitions.'"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        disabled={isGenerating || disabled}
        rows={3}
      />
      <div className="director-options">
        <label>
          Target duration:
          <input
            type="number"
            placeholder="Auto"
            value={targetDuration}
            onChange={(e) => setTargetDuration(e.target.value)}
            disabled={isGenerating || disabled}
            min={5}
            max={300}
          />
          <span>seconds</span>
        </label>
      </div>
      <button
        onClick={handleGenerate}
        disabled={isGenerating || !prompt.trim() || segments.length < 2 || disabled}
        className="director-button"
      >
        {isGenerating ? "AI is thinking..." : `Ask AI Director (${segments.length} clips)`}
      </button>
      <p className="director-note">
        Uses Claude API (~$0.07-0.25 per request depending on # of clips)
      </p>
    </div>
  );
}
