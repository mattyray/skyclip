import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
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

type View = "import" | "library" | "flight" | "analyze";

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

  useEffect(() => {
    initApp();
  }, []);

  async function initApp() {
    try {
      await invoke("init_database");
      setInitialized(true);
      await loadFlights();
      await loadProfiles();
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

          <h3>Top Segments</h3>
          {topSegments.length === 0 ? (
            <p className="empty-state">No segments found matching profile criteria</p>
          ) : (
            <div className="segments-grid">
              {topSegments.map((item, idx) => (
                <div key={item.segment.id} className="segment-card">
                  <div className="segment-rank">#{idx + 1}</div>
                  <div className="segment-info">
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
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  );
}

export default App;
