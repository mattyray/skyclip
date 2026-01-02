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

interface Flight {
  id: string;
  name: string;
  import_date: string;
  source_path: string;
  total_clips: number | null;
}

function App() {
  const [initialized, setInitialized] = useState(false);
  const [flights, setFlights] = useState<Flight[]>([]);
  const [scannedClips, setScannedClips] = useState<ClipInfo[]>([]);
  const [selectedFolder, setSelectedFolder] = useState<string | null>(null);
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState<IngestResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    initApp();
  }, []);

  async function initApp() {
    try {
      await invoke("init_database");
      setInitialized(true);
      await loadFlights();
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

  function formatDuration(sec: number | null): string {
    if (!sec) return "--:--";
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
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
      </header>

      {error && <div className="error-banner">{error}</div>}

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

      <section className="library-section">
        <h2>Library</h2>
        {flights.length === 0 ? (
          <p className="empty-state">No flights imported yet</p>
        ) : (
          <ul className="flights-list">
            {flights.map((flight) => (
              <li key={flight.id}>
                <strong>{flight.name}</strong>
                <span className="flight-meta">
                  {flight.total_clips} clips &bull;{" "}
                  {new Date(flight.import_date).toLocaleDateString()}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}

export default App;
