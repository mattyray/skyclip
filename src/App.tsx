import { useEffect, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import "./App.css";

interface ClipInfo {
  filename: string;
  source_path: string;
  srt_path: string | null;
  lrf_path: string | null;
  duration_sec: number | null;
  resolution: string | null;
  framerate: number | null;
}

interface IngestResult {
  flight_id: string;
  clips_count: number;
  lrf_used: number;
  proxies_generated: number;
}

interface AnalyzeResult {
  clip_id: string;
  segments_created: number;
  top_score: number;
}

interface Flight {
  id: string;
  name: string;
  import_date: string;
  source_path: string;
  total_clips: number | null;
}

interface SourceClip {
  id: string;
  flight_id: string;
  filename: string;
  source_path: string;
  proxy_path: string | null;
  srt_path: string | null;
  duration_sec: number | null;
}

interface Segment {
  id: string;
  source_clip_id: string;
  start_time_ms: number;
  end_time_ms: number;
  duration_ms: number;
  thumbnail_path: string | null;
  motion_magnitude: number | null;
  gimbal_smoothness: number | null;
  gps_speed_avg: number | null;
  is_selected: boolean;
}

interface SegmentWithScores {
  segment: Segment;
  scores: Record<string, number>;
}

interface ProfileInfo {
  id: string;
  name: string;
  description: string;
}

interface SegmentWithClip {
  segment: Segment;
  clip_id: string;
  clip_filename: string;
  proxy_path: string | null;
  source_path: string;
}

interface ExportResult {
  output_path: string;
  duration_sec: number;
}

interface RenderResult {
  output_path: string;
  duration_sec: number;
  clips_count: number;
}

interface RenderClipInput {
  segment_id: string;
  adjusted_start_ms: number;
  adjusted_end_ms: number;
  transition_type: string;
  transition_duration_ms: number;
}

interface EditDecision {
  clip_id: string;
  sequence_order: number;
  adjusted_start_ms: number;
  adjusted_end_ms: number;
  transition_type: string;
  transition_duration_ms: number;
  confidence: number;
  reasoning: string;
}

interface EditSequence {
  decisions: EditDecision[];
  total_duration_ms: number;
  style: string;
  was_reordered: boolean;
}

type View = "import" | "library" | "flight" | "analyze" | "highlight";

function App() {
  const [initialized, setInitialized] = useState(false);
  const [currentView, setCurrentView] = useState<View>("import");
  const [flights, setFlights] = useState<Flight[]>([]);
  const [scannedClips, setScannedClips] = useState<ClipInfo[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Flight detail view state
  const [selectedFlight, setSelectedFlight] = useState<Flight | null>(null);
  const [flightClips, setFlightClips] = useState<SourceClip[]>([]);

  // Analysis state
  const [profiles, setProfiles] = useState<ProfileInfo[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<string>("discovery");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analyzeResults, setAnalyzeResults] = useState<AnalyzeResult[]>([]);
  const [topSegments, setTopSegments] = useState<SegmentWithScores[]>([]);

  // Preview state
  const [previewSegment, setPreviewSegment] = useState<SegmentWithClip | null>(null);
  const [isExporting, setIsExporting] = useState(false);

  // Highlight reel state
  const [selectedSegments, setSelectedSegments] = useState<Set<string>>(new Set());
  const [editSequence, setEditSequence] = useState<EditSequence | null>(null);
  const [editStyle, setEditStyle] = useState<string>("cinematic");
  const [isGeneratingSequence, setIsGeneratingSequence] = useState(false);
  const [isRenderingHighlight, setIsRenderingHighlight] = useState(false);
  const [pythonAvailable, setPythonAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    initApp();
  }, []);

  async function initApp() {
    try {
      await invoke("init_database");
      setInitialized(true);
      await loadFlights();
      await loadProfiles();
      // Check Python availability
      try {
        const available = await invoke<boolean>("check_python_available");
        setPythonAvailable(available);
      } catch {
        setPythonAvailable(false);
      }
    } catch (e) {
      setError(`Failed to initialize: ${e}`);
    }
  }

  async function loadFlights() {
    try {
      const result = await invoke<Flight[]>("list_flights");
      setFlights(result);
    } catch (e) {
      setError(`Failed to load flights: ${e}`);
    }
  }

  async function loadProfiles() {
    try {
      const result = await invoke<ProfileInfo[]>("list_profiles");
      setProfiles(result);
      if (result.length > 0 && !result.find((p) => p.id === selectedProfile)) {
        setSelectedProfile(result[0].id);
      }
    } catch (e) {
      console.error("Failed to load profiles:", e);
    }
  }

  async function selectFolder() {
    try {
      const folder = await open({
        directory: true,
        title: "Select DJI Footage Folder",
      });

      if (folder) {
        setSelectedFolder(folder as string);
        setScannedClips([]);
        setIngestResult(null);
        setError(null);

        const clips = await invoke<ClipInfo[]>("scan_folder", {
          folderPath: folder,
        });
        setScannedClips(clips);
      }
    } catch (e) {
      setError(`Failed to scan folder: ${e}`);
    }
  }

  async function startIngest() {
    if (!selectedFolder) return;

    setIsIngesting(true);
    setError(null);

    try {
      const flightName = `Flight ${new Date().toLocaleDateString()}`;
      const result = await invoke<IngestResult>("ingest_folder", {
        folderPath: selectedFolder,
        flightName,
      });

      setIngestResult(result);
      await loadFlights();
    } catch (e) {
      setError(`Ingest failed: ${e}`);
    } finally {
      setIsIngesting(false);
    }
  }

  async function openFlight(flight: Flight) {
    setSelectedFlight(flight);
    setCurrentView("flight");
    setError(null);

    try {
      const clips = await invoke<SourceClip[]>("get_flight_clips", {
        flightId: flight.id,
      });
      setFlightClips(clips);
    } catch (e) {
      setError(`Failed to load clips: ${e}`);
    }
  }

  async function handleDeleteFlight(flightId: string) {
    if (!confirm("Delete this flight and all its data?")) return;

    try {
      await invoke("delete_flight", { flightId });
      await loadFlights();
      if (selectedFlight?.id === flightId) {
        setSelectedFlight(null);
        setCurrentView("library");
      }
    } catch (e) {
      setError(`Failed to delete flight: ${e}`);
    }
  }

  async function analyzeFlight() {
    if (!selectedFlight) return;

    setIsAnalyzing(true);
    setError(null);
    setAnalyzeResults([]);
    setTopSegments([]);

    try {
      const results = await invoke<AnalyzeResult[]>("analyze_flight", {
        flightId: selectedFlight.id,
        profileId: selectedProfile,
      });

      setAnalyzeResults(results);

      // Load top segments
      const segments = await invoke<SegmentWithScores[]>("get_top_segments", {
        flightId: selectedFlight.id,
        profileId: selectedProfile,
        limit: 20,
      });

      setTopSegments(segments);
      setCurrentView("analyze");
    } catch (e) {
      setError(`Analysis failed: ${e}`);
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function openPreview(segmentId: string) {
    try {
      const segmentWithClip = await invoke<SegmentWithClip>("get_segment_with_clip", {
        segmentId,
      });
      setPreviewSegment(segmentWithClip);
    } catch (e) {
      setError(`Failed to load segment: ${e}`);
    }
  }

  async function exportSegment(segmentId: string, useSource: boolean) {
    const { save } = await import("@tauri-apps/plugin-dialog");

    const outputPath = await save({
      title: "Export Segment",
      filters: [{ name: "Video", extensions: ["mp4"] }],
      defaultPath: `segment_${segmentId.slice(0, 8)}.mp4`,
    });

    if (!outputPath) return;

    setIsExporting(true);
    try {
      const result = await invoke<ExportResult>("export_segment", {
        segmentId,
        outputPath,
        useSource,
      });
      alert(`Exported ${result.duration_sec.toFixed(1)}s clip to:\n${result.output_path}`);
    } catch (e) {
      setError(`Export failed: ${e}`);
    } finally {
      setIsExporting(false);
    }
  }

  function toggleSegmentSelection(segmentId: string) {
    setSelectedSegments((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(segmentId)) {
        newSet.delete(segmentId);
      } else {
        newSet.add(segmentId);
      }
      return newSet;
    });
    // Clear edit sequence when selection changes
    setEditSequence(null);
  }

  function selectAllSegments() {
    setSelectedSegments(new Set(topSegments.map((s) => s.segment.id)));
    setEditSequence(null);
  }

  function clearSelection() {
    setSelectedSegments(new Set());
    setEditSequence(null);
  }

  async function generateEditSequence() {
    if (selectedSegments.size < 2) {
      setError("Select at least 2 segments to create a highlight reel");
      return;
    }

    setIsGeneratingSequence(true);
    setError(null);

    try {
      const segmentIds = Array.from(selectedSegments);
      const sequence = await invoke<EditSequence>("generate_edit_sequence", {
        segmentIds,
        style: editStyle,
        reorder: true,
      });
      setEditSequence(sequence);
      setCurrentView("highlight");
    } catch (e) {
      setError(`Failed to generate edit sequence: ${e}`);
    } finally {
      setIsGeneratingSequence(false);
    }
  }

  function updateTransitionType(index: number, newType: string) {
    if (!editSequence) return;
    const newDecisions = [...editSequence.decisions];
    newDecisions[index] = { ...newDecisions[index], transition_type: newType };
    setEditSequence({ ...editSequence, decisions: newDecisions });
  }

  function moveClipUp(index: number) {
    if (!editSequence || index === 0) return;
    const newDecisions = [...editSequence.decisions];
    [newDecisions[index - 1], newDecisions[index]] = [newDecisions[index], newDecisions[index - 1]];
    // Update sequence orders
    newDecisions.forEach((d, i) => (d.sequence_order = i));
    setEditSequence({ ...editSequence, decisions: newDecisions });
  }

  function moveClipDown(index: number) {
    if (!editSequence || index >= editSequence.decisions.length - 1) return;
    const newDecisions = [...editSequence.decisions];
    [newDecisions[index], newDecisions[index + 1]] = [newDecisions[index + 1], newDecisions[index]];
    // Update sequence orders
    newDecisions.forEach((d, i) => (d.sequence_order = i));
    setEditSequence({ ...editSequence, decisions: newDecisions });
  }

  function removeFromSequence(index: number) {
    if (!editSequence) return;
    const newDecisions = editSequence.decisions.filter((_, i) => i !== index);
    newDecisions.forEach((d, i) => (d.sequence_order = i));
    const newDuration = newDecisions.reduce(
      (sum, d) => sum + (d.adjusted_end_ms - d.adjusted_start_ms),
      0
    );
    setEditSequence({ ...editSequence, decisions: newDecisions, total_duration_ms: newDuration });
  }

  async function renderHighlightReel(useSource: boolean = false) {
    if (!editSequence || editSequence.decisions.length === 0) return;

    const { save } = await import("@tauri-apps/plugin-dialog");

    const outputPath = await save({
      title: "Save Highlight Reel",
      filters: [{ name: "Video", extensions: ["mp4"] }],
      defaultPath: `highlight_${editStyle}_${Date.now()}.mp4`,
    });

    if (!outputPath) return;

    setIsRenderingHighlight(true);
    setError(null);

    try {
      // Build clips array from edit sequence decisions
      const clips: RenderClipInput[] = editSequence.decisions.map((decision) => ({
        segment_id: decision.clip_id,
        adjusted_start_ms: decision.adjusted_start_ms,
        adjusted_end_ms: decision.adjusted_end_ms,
        transition_type: decision.transition_type,
        transition_duration_ms: decision.transition_duration_ms,
      }));

      const result = await invoke<RenderResult>("render_highlight_reel", {
        clips,
        outputPath,
        useSource,
      });

      alert(
        `Highlight reel exported!\n\n` +
        `Duration: ${result.duration_sec.toFixed(1)}s\n` +
        `Clips: ${result.clips_count}\n` +
        `Location: ${result.output_path}`
      );
    } catch (e) {
      setError(`Failed to render highlight reel: ${e}`);
    } finally {
      setIsRenderingHighlight(false);
    }
  }

  function getSegmentById(id: string): SegmentWithScores | undefined {
    return topSegments.find((s) => s.segment.id === id);
  }

  function formatDuration(sec: number | null): string {
    if (!sec) return "--:--";
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  function formatTimeMs(ms: number): string {
    const totalSec = Math.floor(ms / 1000);
    const mins = Math.floor(totalSec / 60);
    const secs = totalSec % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  }

  if (!initialized) {
    return (
      <main className="container">
        <h1>SkyClip</h1>
        <p>Initializing...</p>
        {error && <p className="error">{error}</p>}
      </main>
    );
  }

  return (
    <main className="container">
      <header>
        <h1>SkyClip</h1>
        <p className="tagline">Drone footage analysis and highlight extraction</p>
        <nav className="nav-tabs">
          <button
            className={currentView === "import" ? "active" : ""}
            onClick={() => setCurrentView("import")}
          >
            Import
          </button>
          <button
            className={currentView === "library" ? "active" : ""}
            onClick={() => setCurrentView("library")}
          >
            Library ({flights.length})
          </button>
          {selectedFlight && (
            <button
              className={currentView === "flight" || currentView === "analyze" ? "active" : ""}
              onClick={() => setCurrentView("flight")}
            >
              {selectedFlight.name}
            </button>
          )}
        </nav>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {currentView === "import" && (
        <section className="ingest-section">
          <h2>Import Footage</h2>
          <button onClick={selectFolder} disabled={isIngesting}>
            Select DJI Folder
          </button>

          {selectedFolder && (
            <div className="folder-info">
              <p>
                <strong>Selected:</strong> {selectedFolder}
              </p>
              <p>
                <strong>Clips found:</strong> {scannedClips.length}
              </p>
            </div>
          )}

          {scannedClips.length > 0 && (
            <>
              <table className="clips-table">
                <thead>
                  <tr>
                    <th>Filename</th>
                    <th>Duration</th>
                    <th>Resolution</th>
                    <th>FPS</th>
                    <th>SRT</th>
                    <th>LRF</th>
                  </tr>
                </thead>
                <tbody>
                  {scannedClips.map((clip) => (
                    <tr key={clip.source_path}>
                      <td>{clip.filename}</td>
                      <td>{formatDuration(clip.duration_sec)}</td>
                      <td>{clip.resolution || "--"}</td>
                      <td>{clip.framerate?.toFixed(1) || "--"}</td>
                      <td>{clip.srt_path ? "Yes" : "No"}</td>
                      <td>{clip.lrf_path ? "Yes" : "No"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <button
                onClick={startIngest}
                disabled={isIngesting}
                className="ingest-button"
              >
                {isIngesting ? "Importing..." : "Import Footage"}
              </button>
            </>
          )}

          {ingestResult && (
            <div className="ingest-result">
              <h3>Import Complete</h3>
              <p>Clips imported: {ingestResult.clips_count}</p>
              <p>LRF proxies used: {ingestResult.lrf_used}</p>
              <p>Proxies generated: {ingestResult.proxies_generated}</p>
            </div>
          )}
        </section>
      )}

      {currentView === "library" && (
        <section className="library-section">
          <h2>Library</h2>
          {flights.length === 0 ? (
            <p className="empty-state">No flights imported yet</p>
          ) : (
            <ul className="flights-list">
              {flights.map((flight) => (
                <li key={flight.id}>
                  <div className="flight-info" onClick={() => openFlight(flight)}>
                    <strong>{flight.name}</strong>
                    <span className="flight-path">{flight.source_path}</span>
                    <span className="flight-meta">
                      {flight.total_clips} clips &bull;{" "}
                      {new Date(flight.import_date).toLocaleDateString()}
                    </span>
                  </div>
                  <button
                    className="delete-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteFlight(flight.id);
                    }}
                  >
                    Delete
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {currentView === "flight" && selectedFlight && (
        <section className="flight-detail-section">
          <h2>{selectedFlight.name}</h2>
          <p className="flight-meta">
            {flightClips.length} clips &bull;{" "}
            {new Date(selectedFlight.import_date).toLocaleDateString()}
          </p>

          <div className="analyze-controls">
            <h3>Analyze Footage</h3>
            <div className="profile-selector">
              <label>Profile:</label>
              <select
                value={selectedProfile}
                onChange={(e) => setSelectedProfile(e.target.value)}
                disabled={isAnalyzing}
              >
                {profiles.map((profile) => (
                  <option key={profile.id} value={profile.id}>
                    {profile.name}
                  </option>
                ))}
              </select>
              {profiles.find((p) => p.id === selectedProfile) && (
                <span className="profile-desc">
                  {profiles.find((p) => p.id === selectedProfile)?.description}
                </span>
              )}
            </div>
            <button
              onClick={analyzeFlight}
              disabled={isAnalyzing}
              className="analyze-button"
            >
              {isAnalyzing ? "Analyzing..." : "Analyze Flight"}
            </button>
          </div>

          <h3>Clips</h3>
          <table className="clips-table">
            <thead>
              <tr>
                <th>Filename</th>
                <th>Duration</th>
                <th>SRT</th>
                <th>Proxy</th>
              </tr>
            </thead>
            <tbody>
              {flightClips.map((clip) => (
                <tr key={clip.id}>
                  <td>{clip.filename}</td>
                  <td>{formatDuration(clip.duration_sec)}</td>
                  <td>{clip.srt_path ? "Yes" : "No"}</td>
                  <td>{clip.proxy_path ? "Ready" : "Missing"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {currentView === "analyze" && selectedFlight && (
        <section className="analyze-results-section">
          <div className="section-header">
            <h2>Analysis Results</h2>
            <button onClick={() => setCurrentView("flight")} className="back-button">
              Back to Flight
            </button>
          </div>

          <div className="analyze-summary">
            <p>
              <strong>Profile:</strong> {profiles.find((p) => p.id === selectedProfile)?.name}
            </p>
            <p>
              <strong>Clips analyzed:</strong> {analyzeResults.length}
            </p>
            <p>
              <strong>Segments found:</strong>{" "}
              {analyzeResults.reduce((sum, r) => sum + r.segments_created, 0)}
            </p>
          </div>

          <div className="segments-header">
            <h3>Top Segments</h3>
            {topSegments.length > 0 && (
              <div className="selection-controls">
                <span className="selection-count">{selectedSegments.size} selected</span>
                <button onClick={selectAllSegments} className="secondary-button">
                  Select All
                </button>
                <button onClick={clearSelection} className="secondary-button">
                  Clear
                </button>
              </div>
            )}
          </div>

          {topSegments.length === 0 ? (
            <p className="empty-state">No segments found matching profile criteria</p>
          ) : (
            <>
              <div className="segments-grid">
                {topSegments.map((item, idx) => (
                  <div
                    key={item.segment.id}
                    className={`segment-card ${selectedSegments.has(item.segment.id) ? "selected" : ""}`}
                  >
                    <div className="segment-select">
                      <input
                        type="checkbox"
                        checked={selectedSegments.has(item.segment.id)}
                        onChange={() => toggleSegmentSelection(item.segment.id)}
                      />
                    </div>
                    <div className="segment-rank">#{idx + 1}</div>
                    <div className="segment-thumbnail" onClick={() => openPreview(item.segment.id)}>
                      {item.segment.thumbnail_path ? (
                        <img
                          src={convertFileSrc(item.segment.thumbnail_path)}
                          alt={`Segment ${idx + 1}`}
                        />
                      ) : (
                        <div className="thumbnail-placeholder">No Preview</div>
                      )}
                    </div>
                    <div className="segment-info" onClick={() => openPreview(item.segment.id)}>
                      <div className="segment-time">
                        {formatTimeMs(item.segment.start_time_ms)} -{" "}
                        {formatTimeMs(item.segment.end_time_ms)}
                      </div>
                      <div className="segment-duration">
                        {(item.segment.duration_ms / 1000).toFixed(1)}s
                      </div>
                    </div>
                    <div className="segment-scores">
                      <div className="score primary">
                        {item.scores[selectedProfile]?.toFixed(0) || "--"}
                      </div>
                      <div className="segment-signals">
                        {item.segment.gimbal_smoothness && (
                          <span title="Gimbal Smoothness">
                            Smooth: {(item.segment.gimbal_smoothness * 100).toFixed(0)}%
                          </span>
                        )}
                        {item.segment.gps_speed_avg && (
                          <span title="GPS Speed">
                            Speed: {item.segment.gps_speed_avg.toFixed(1)} m/s
                          </span>
                        )}
                        {item.segment.motion_magnitude && (
                          <span title="Motion">
                            Motion: {item.segment.motion_magnitude.toFixed(1)}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="segment-actions">
                      <button
                        onClick={() => exportSegment(item.segment.id, false)}
                        disabled={isExporting}
                        className="export-button"
                      >
                        Export (Quick)
                      </button>
                      <button
                        onClick={() => exportSegment(item.segment.id, true)}
                        disabled={isExporting}
                        className="export-button source"
                      >
                        Export (4K)
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {selectedSegments.size >= 2 && (
                <div className="highlight-reel-panel">
                  <h3>Create Highlight Reel</h3>
                  <div className="highlight-options">
                    <div className="style-selector">
                      <label>Edit Style:</label>
                      <select
                        value={editStyle}
                        onChange={(e) => setEditStyle(e.target.value)}
                        disabled={isGeneratingSequence}
                      >
                        <option value="cinematic">Cinematic (smooth, longer takes)</option>
                        <option value="action">Action (fast cuts, high energy)</option>
                        <option value="social">Social (short, punchy)</option>
                      </select>
                    </div>
                    <button
                      onClick={generateEditSequence}
                      disabled={isGeneratingSequence || selectedSegments.size < 2}
                      className="generate-button"
                    >
                      {isGeneratingSequence ? "Generating..." : `Generate Highlight Reel (${selectedSegments.size} clips)`}
                    </button>
                    {pythonAvailable === false && (
                      <p className="python-warning">
                        Python with OpenCV not detected. Using basic transitions.
                      </p>
                    )}
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      )}

      {currentView === "highlight" && editSequence && (
        <section className="highlight-editor-section">
          <div className="section-header">
            <h2>Highlight Reel Editor</h2>
            <button onClick={() => setCurrentView("analyze")} className="back-button">
              Back to Segments
            </button>
          </div>

          <div className="highlight-summary">
            <p>
              <strong>Style:</strong> {editSequence.style}
            </p>
            <p>
              <strong>Total Duration:</strong> {(editSequence.total_duration_ms / 1000).toFixed(1)}s
            </p>
            <p>
              <strong>Clips:</strong> {editSequence.decisions.length}
            </p>
            {editSequence.was_reordered && (
              <p className="reorder-note">Clips were reordered for better flow</p>
            )}
          </div>

          <div className="timeline-editor">
            <h3>Timeline</h3>
            <p className="timeline-help">
              Drag to reorder, change transitions, or remove clips. AI suggestions shown with confidence %.
            </p>

            <div className="timeline-clips">
              {editSequence.decisions.map((decision, idx) => {
                const segment = getSegmentById(decision.clip_id);
                return (
                  <div key={decision.clip_id} className="timeline-clip">
                    <div className="clip-controls">
                      <button
                        onClick={() => moveClipUp(idx)}
                        disabled={idx === 0}
                        className="reorder-btn"
                        title="Move up"
                      >
                        ^
                      </button>
                      <button
                        onClick={() => moveClipDown(idx)}
                        disabled={idx === editSequence.decisions.length - 1}
                        className="reorder-btn"
                        title="Move down"
                      >
                        v
                      </button>
                    </div>

                    <div className="clip-thumbnail">
                      {segment?.segment.thumbnail_path ? (
                        <img
                          src={convertFileSrc(segment.segment.thumbnail_path)}
                          alt={`Clip ${idx + 1}`}
                          onClick={() => openPreview(decision.clip_id)}
                        />
                      ) : (
                        <div className="thumbnail-placeholder">Preview</div>
                      )}
                    </div>

                    <div className="clip-info">
                      <div className="clip-number">Clip {idx + 1}</div>
                      <div className="clip-timing">
                        {formatTimeMs(decision.adjusted_start_ms)} - {formatTimeMs(decision.adjusted_end_ms)}
                        <span className="clip-duration">
                          ({((decision.adjusted_end_ms - decision.adjusted_start_ms) / 1000).toFixed(1)}s)
                        </span>
                      </div>
                    </div>

                    <div className="clip-transition">
                      {idx < editSequence.decisions.length - 1 && (
                        <>
                          <label>Transition:</label>
                          <select
                            value={decision.transition_type}
                            onChange={(e) => updateTransitionType(idx, e.target.value)}
                          >
                            <option value="cut">Hard Cut</option>
                            <option value="dissolve">Dissolve</option>
                            <option value="dip_black">Dip to Black</option>
                          </select>
                          <span className="confidence" title={decision.reasoning}>
                            {(decision.confidence * 100).toFixed(0)}% confident
                          </span>
                        </>
                      )}
                    </div>

                    <button
                      onClick={() => removeFromSequence(idx)}
                      className="remove-btn"
                      title="Remove from highlight"
                    >
                      X
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="render-controls">
            <button
              onClick={() => renderHighlightReel(false)}
              disabled={isRenderingHighlight || editSequence.decisions.length === 0}
              className="render-button"
            >
              {isRenderingHighlight ? "Rendering..." : "Export (Quick)"}
            </button>
            <button
              onClick={() => renderHighlightReel(true)}
              disabled={isRenderingHighlight || editSequence.decisions.length === 0}
              className="render-button source"
            >
              {isRenderingHighlight ? "Rendering..." : "Export (4K Source)"}
            </button>
          </div>
        </section>
      )}

      {/* Preview Modal */}
      {previewSegment && (
        <div className="preview-modal" onClick={() => setPreviewSegment(null)}>
          <div className="preview-content" onClick={(e) => e.stopPropagation()}>
            <div className="preview-header">
              <h3>Preview: {previewSegment.clip_filename}</h3>
              <button onClick={() => setPreviewSegment(null)} className="close-button">
                Close
              </button>
            </div>
            <video
              controls
              autoPlay
              src={convertFileSrc(previewSegment.proxy_path || previewSegment.source_path)}
              style={{ maxWidth: "100%", maxHeight: "60vh" }}
              onLoadedMetadata={(e) => {
                const video = e.currentTarget;
                video.currentTime = previewSegment.segment.start_time_ms / 1000;
              }}
              onTimeUpdate={(e) => {
                const video = e.currentTarget;
                const endTime = previewSegment.segment.end_time_ms / 1000;
                if (video.currentTime >= endTime) {
                  video.pause();
                  video.currentTime = previewSegment.segment.start_time_ms / 1000;
                }
              }}
            />
            <div className="preview-info">
              <p>
                Segment: {formatTimeMs(previewSegment.segment.start_time_ms)} -{" "}
                {formatTimeMs(previewSegment.segment.end_time_ms)} (
                {(previewSegment.segment.duration_ms / 1000).toFixed(1)}s)
              </p>
              <p className="preview-note">
                Preview loops the selected segment. Use scrubber to explore full clip.
              </p>
            </div>
            <div className="preview-actions">
              <button
                onClick={() => exportSegment(previewSegment.segment.id, false)}
                disabled={isExporting}
                className="export-button"
              >
                {isExporting ? "Exporting..." : "Export (Quick)"}
              </button>
              <button
                onClick={() => exportSegment(previewSegment.segment.id, true)}
                disabled={isExporting}
                className="export-button source"
              >
                {isExporting ? "Exporting..." : "Export (4K Source)"}
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default App;
