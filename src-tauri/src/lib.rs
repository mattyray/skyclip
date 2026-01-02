mod commands;
mod models;
mod services;

use commands::{
    analyze_clip, analyze_flight, get_clip_segments, get_flight_clips, get_top_segments,
    ingest_folder, init_database, list_flights, list_profiles, parse_srt, scan_folder, AppState,
};
use tauri::Manager;
use tokio::sync::Mutex;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let app_data_dir = app
                .path()
                .app_data_dir()
                .expect("Failed to get app data directory");

            app.manage(AppState {
                db: Mutex::new(None),
                app_data_dir: Mutex::new(app_data_dir),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            init_database,
            scan_folder,
            ingest_folder,
            list_flights,
            get_flight_clips,
            parse_srt,
            analyze_clip,
            analyze_flight,
            get_clip_segments,
            get_top_segments,
            list_profiles,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
