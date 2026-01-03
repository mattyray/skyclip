use anyhow::Result;
use sqlx::{sqlite::SqlitePoolOptions, Pool, Row, Sqlite};
use std::path::Path;

use crate::models::{Flight, SourceClip, Segment};

pub struct Database {
    pool: Pool<Sqlite>,
}

impl Database {
    /// Create a new database connection
    pub async fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db_url = format!("sqlite:{}?mode=rwc", path.as_ref().display());

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(&db_url)
            .await?;

        let db = Self { pool };
        db.run_migrations().await?;

        Ok(db)
    }

    /// Run database migrations
    async fn run_migrations(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS flights (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                import_date TEXT NOT NULL,
                source_path TEXT NOT NULL,
                location_name TEXT,
                gps_center_lat REAL,
                gps_center_lon REAL,
                total_duration_sec REAL,
                total_clips INTEGER
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS source_clips (
                id TEXT PRIMARY KEY,
                flight_id TEXT NOT NULL REFERENCES flights(id),
                filename TEXT NOT NULL,
                source_path TEXT NOT NULL,
                proxy_path TEXT,
                proxy_source TEXT,
                srt_path TEXT,
                duration_sec REAL,
                resolution_width INTEGER,
                resolution_height INTEGER,
                framerate REAL,
                recorded_at TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS segments (
                id TEXT PRIMARY KEY,
                source_clip_id TEXT NOT NULL REFERENCES source_clips(id),
                start_time_ms INTEGER NOT NULL,
                end_time_ms INTEGER NOT NULL,
                duration_ms INTEGER NOT NULL,
                thumbnail_path TEXT,
                motion_magnitude REAL,
                gimbal_pitch_delta_avg REAL,
                gimbal_yaw_delta_avg REAL,
                gimbal_smoothness REAL,
                altitude_delta REAL,
                gps_speed_avg REAL,
                iso_avg REAL,
                visual_quality REAL,
                has_scene_change INTEGER,
                is_selected INTEGER DEFAULT 0,
                user_adjusted_start_ms INTEGER,
                user_adjusted_end_ms INTEGER
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS selections (
                id TEXT PRIMARY KEY,
                flight_id TEXT NOT NULL REFERENCES flights(id),
                segment_id TEXT NOT NULL REFERENCES segments(id),
                sequence_order INTEGER NOT NULL,
                added_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS segment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                segment_id TEXT NOT NULL REFERENCES segments(id),
                profile_id TEXT NOT NULL,
                score REAL NOT NULL,
                UNIQUE(segment_id, profile_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_segment_scores_segment ON segment_scores(segment_id)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_segment_scores_profile ON segment_scores(profile_id, score DESC)",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    // Flight operations
    pub async fn insert_flight(&self, flight: &Flight) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO flights (id, name, import_date, source_path, location_name, gps_center_lat, gps_center_lon, total_duration_sec, total_clips)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&flight.id)
        .bind(&flight.name)
        .bind(flight.import_date.to_rfc3339())
        .bind(&flight.source_path)
        .bind(&flight.location_name)
        .bind(flight.gps_center_lat)
        .bind(flight.gps_center_lon)
        .bind(flight.total_duration_sec)
        .bind(flight.total_clips)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_flight(&self, id: &str) -> Result<Option<Flight>> {
        let row = sqlx::query(
            "SELECT id, name, import_date, source_path, location_name, gps_center_lat, gps_center_lon, total_duration_sec, total_clips FROM flights WHERE id = ?"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => {
                let import_date_str: String = row.get("import_date");
                Ok(Some(Flight {
                    id: row.get("id"),
                    name: row.get("name"),
                    import_date: chrono::DateTime::parse_from_rfc3339(&import_date_str)?.with_timezone(&chrono::Utc),
                    source_path: row.get("source_path"),
                    location_name: row.get("location_name"),
                    gps_center_lat: row.get("gps_center_lat"),
                    gps_center_lon: row.get("gps_center_lon"),
                    total_duration_sec: row.get("total_duration_sec"),
                    total_clips: row.get("total_clips"),
                }))
            }
            None => Ok(None),
        }
    }

    pub async fn list_flights(&self) -> Result<Vec<Flight>> {
        let rows = sqlx::query(
            "SELECT id, name, import_date, source_path, location_name, gps_center_lat, gps_center_lon, total_duration_sec, total_clips FROM flights ORDER BY import_date DESC"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut flights = Vec::new();
        for row in rows {
            let import_date_str: String = row.get("import_date");
            flights.push(Flight {
                id: row.get("id"),
                name: row.get("name"),
                import_date: chrono::DateTime::parse_from_rfc3339(&import_date_str)?.with_timezone(&chrono::Utc),
                source_path: row.get("source_path"),
                location_name: row.get("location_name"),
                gps_center_lat: row.get("gps_center_lat"),
                gps_center_lon: row.get("gps_center_lon"),
                total_duration_sec: row.get("total_duration_sec"),
                total_clips: row.get("total_clips"),
            });
        }

        Ok(flights)
    }

    // SourceClip operations
    pub async fn insert_clip(&self, clip: &SourceClip) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO source_clips (id, flight_id, filename, source_path, proxy_path, proxy_source, srt_path, duration_sec, resolution_width, resolution_height, framerate, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&clip.id)
        .bind(&clip.flight_id)
        .bind(&clip.filename)
        .bind(&clip.source_path)
        .bind(&clip.proxy_path)
        .bind(&clip.proxy_source)
        .bind(&clip.srt_path)
        .bind(clip.duration_sec)
        .bind(clip.resolution_width)
        .bind(clip.resolution_height)
        .bind(clip.framerate)
        .bind(clip.recorded_at.map(|dt| dt.to_rfc3339()))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_clips_for_flight(&self, flight_id: &str) -> Result<Vec<SourceClip>> {
        let rows = sqlx::query(
            "SELECT id, flight_id, filename, source_path, proxy_path, proxy_source, srt_path, duration_sec, resolution_width, resolution_height, framerate, recorded_at FROM source_clips WHERE flight_id = ?"
        )
        .bind(flight_id)
        .fetch_all(&self.pool)
        .await?;

        let mut clips = Vec::new();
        for row in rows {
            let recorded_at_str: Option<String> = row.get("recorded_at");
            clips.push(SourceClip {
                id: row.get("id"),
                flight_id: row.get("flight_id"),
                filename: row.get("filename"),
                source_path: row.get("source_path"),
                proxy_path: row.get("proxy_path"),
                proxy_source: row.get("proxy_source"),
                srt_path: row.get("srt_path"),
                duration_sec: row.get("duration_sec"),
                resolution_width: row.get("resolution_width"),
                resolution_height: row.get("resolution_height"),
                framerate: row.get("framerate"),
                recorded_at: recorded_at_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&chrono::Utc))),
            });
        }

        Ok(clips)
    }

    pub async fn update_clip_proxy(&self, clip_id: &str, proxy_path: &str, proxy_source: &str) -> Result<()> {
        sqlx::query("UPDATE source_clips SET proxy_path = ?, proxy_source = ? WHERE id = ?")
            .bind(proxy_path)
            .bind(proxy_source)
            .bind(clip_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Delete a flight and all associated data (clips, segments, scores)
    pub async fn delete_flight(&self, flight_id: &str) -> Result<()> {
        // Get clip IDs for this flight to clean up segments and scores
        let clip_ids: Vec<String> = sqlx::query_scalar(
            "SELECT id FROM source_clips WHERE flight_id = ?"
        )
        .bind(flight_id)
        .fetch_all(&self.pool)
        .await?;

        // Delete segment scores for all clips in this flight
        for clip_id in &clip_ids {
            sqlx::query("DELETE FROM segment_scores WHERE segment_id IN (SELECT id FROM segments WHERE source_clip_id = ?)")
                .bind(clip_id)
                .execute(&self.pool)
                .await?;

            sqlx::query("DELETE FROM segments WHERE source_clip_id = ?")
                .bind(clip_id)
                .execute(&self.pool)
                .await?;
        }

        // Delete clips
        sqlx::query("DELETE FROM source_clips WHERE flight_id = ?")
            .bind(flight_id)
            .execute(&self.pool)
            .await?;

        // Delete flight
        sqlx::query("DELETE FROM flights WHERE id = ?")
            .bind(flight_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    // Segment operations
    pub async fn insert_segment(&self, segment: &Segment) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO segments (id, source_clip_id, start_time_ms, end_time_ms, duration_ms, thumbnail_path, motion_magnitude, gimbal_pitch_delta_avg, gimbal_yaw_delta_avg, gimbal_smoothness, altitude_delta, gps_speed_avg, iso_avg, visual_quality, has_scene_change, is_selected, user_adjusted_start_ms, user_adjusted_end_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&segment.id)
        .bind(&segment.source_clip_id)
        .bind(segment.start_time_ms)
        .bind(segment.end_time_ms)
        .bind(segment.duration_ms)
        .bind(&segment.thumbnail_path)
        .bind(segment.motion_magnitude)
        .bind(segment.gimbal_pitch_delta_avg)
        .bind(segment.gimbal_yaw_delta_avg)
        .bind(segment.gimbal_smoothness)
        .bind(segment.altitude_delta)
        .bind(segment.gps_speed_avg)
        .bind(segment.iso_avg)
        .bind(segment.visual_quality)
        .bind(segment.has_scene_change)
        .bind(segment.is_selected)
        .bind(segment.user_adjusted_start_ms)
        .bind(segment.user_adjusted_end_ms)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_segments_for_clip(&self, clip_id: &str) -> Result<Vec<Segment>> {
        let rows = sqlx::query(
            "SELECT id, source_clip_id, start_time_ms, end_time_ms, duration_ms, thumbnail_path, motion_magnitude, gimbal_pitch_delta_avg, gimbal_yaw_delta_avg, gimbal_smoothness, altitude_delta, gps_speed_avg, iso_avg, visual_quality, has_scene_change, is_selected, user_adjusted_start_ms, user_adjusted_end_ms FROM segments WHERE source_clip_id = ? ORDER BY start_time_ms"
        )
        .bind(clip_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(Self::rows_to_segments(rows))
    }

    pub async fn get_segment(&self, segment_id: &str) -> Result<Option<Segment>> {
        let row = sqlx::query(
            "SELECT id, source_clip_id, start_time_ms, end_time_ms, duration_ms, thumbnail_path, motion_magnitude, gimbal_pitch_delta_avg, gimbal_yaw_delta_avg, gimbal_smoothness, altitude_delta, gps_speed_avg, iso_avg, visual_quality, has_scene_change, is_selected, user_adjusted_start_ms, user_adjusted_end_ms FROM segments WHERE id = ?"
        )
        .bind(segment_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => {
                let segments = Self::rows_to_segments(vec![row]);
                Ok(segments.into_iter().next())
            }
            None => Ok(None),
        }
    }

    pub async fn get_clip(&self, clip_id: &str) -> Result<Option<SourceClip>> {
        let row = sqlx::query(
            "SELECT id, flight_id, filename, source_path, proxy_path, proxy_source, srt_path, duration_sec, resolution_width, resolution_height, framerate, recorded_at FROM source_clips WHERE id = ?"
        )
        .bind(clip_id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => {
                let recorded_at_str: Option<String> = row.get("recorded_at");
                Ok(Some(SourceClip {
                    id: row.get("id"),
                    flight_id: row.get("flight_id"),
                    filename: row.get("filename"),
                    source_path: row.get("source_path"),
                    proxy_path: row.get("proxy_path"),
                    proxy_source: row.get("proxy_source"),
                    srt_path: row.get("srt_path"),
                    duration_sec: row.get("duration_sec"),
                    resolution_width: row.get("resolution_width"),
                    resolution_height: row.get("resolution_height"),
                    framerate: row.get("framerate"),
                    recorded_at: recorded_at_str.and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok().map(|dt| dt.with_timezone(&chrono::Utc))),
                }))
            }
            None => Ok(None),
        }
    }

    pub async fn delete_segments_for_clip(&self, clip_id: &str) -> Result<()> {
        // First delete scores for these segments
        sqlx::query(
            "DELETE FROM segment_scores WHERE segment_id IN (SELECT id FROM segments WHERE source_clip_id = ?)"
        )
        .bind(clip_id)
        .execute(&self.pool)
        .await?;

        // Then delete the segments
        sqlx::query("DELETE FROM segments WHERE source_clip_id = ?")
            .bind(clip_id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn insert_segment_score(&self, segment_id: &str, profile_id: &str, score: f64) -> Result<()> {
        sqlx::query(
            "INSERT OR REPLACE INTO segment_scores (segment_id, profile_id, score) VALUES (?, ?, ?)"
        )
        .bind(segment_id)
        .bind(profile_id)
        .bind(score)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_segment_scores(&self, segment_id: &str) -> Result<std::collections::HashMap<String, f64>> {
        let rows = sqlx::query(
            "SELECT profile_id, score FROM segment_scores WHERE segment_id = ?"
        )
        .bind(segment_id)
        .fetch_all(&self.pool)
        .await?;

        let mut scores = std::collections::HashMap::new();
        for row in rows {
            let profile_id: String = row.get("profile_id");
            let score: f64 = row.get("score");
            scores.insert(profile_id, score);
        }

        Ok(scores)
    }

    pub async fn get_top_segments_for_flight(&self, flight_id: &str, profile_id: &str, limit: u32) -> Result<Vec<Segment>> {
        let rows = sqlx::query(
            r#"
            SELECT s.id, s.source_clip_id, s.start_time_ms, s.end_time_ms, s.duration_ms, s.thumbnail_path,
                   s.motion_magnitude, s.gimbal_pitch_delta_avg, s.gimbal_yaw_delta_avg, s.gimbal_smoothness,
                   s.altitude_delta, s.gps_speed_avg, s.iso_avg, s.visual_quality, s.has_scene_change,
                   s.is_selected, s.user_adjusted_start_ms, s.user_adjusted_end_ms
            FROM segments s
            JOIN source_clips c ON s.source_clip_id = c.id
            JOIN segment_scores sc ON s.id = sc.segment_id
            WHERE c.flight_id = ? AND sc.profile_id = ?
            ORDER BY sc.score DESC
            LIMIT ?
            "#
        )
        .bind(flight_id)
        .bind(profile_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;

        Ok(Self::rows_to_segments(rows))
    }

    fn rows_to_segments(rows: Vec<sqlx::sqlite::SqliteRow>) -> Vec<Segment> {
        rows.into_iter()
            .map(|row| {
                let has_scene_change: Option<i32> = row.get("has_scene_change");
                let is_selected: i32 = row.get("is_selected");
                Segment {
                    id: row.get("id"),
                    source_clip_id: row.get("source_clip_id"),
                    start_time_ms: row.get("start_time_ms"),
                    end_time_ms: row.get("end_time_ms"),
                    duration_ms: row.get("duration_ms"),
                    thumbnail_path: row.get("thumbnail_path"),
                    motion_magnitude: row.get("motion_magnitude"),
                    gimbal_pitch_delta_avg: row.get("gimbal_pitch_delta_avg"),
                    gimbal_yaw_delta_avg: row.get("gimbal_yaw_delta_avg"),
                    gimbal_smoothness: row.get("gimbal_smoothness"),
                    altitude_delta: row.get("altitude_delta"),
                    gps_speed_avg: row.get("gps_speed_avg"),
                    iso_avg: row.get("iso_avg"),
                    visual_quality: row.get("visual_quality"),
                    has_scene_change: has_scene_change.map(|v| v != 0),
                    is_selected: is_selected != 0,
                    user_adjusted_start_ms: row.get("user_adjusted_start_ms"),
                    user_adjusted_end_ms: row.get("user_adjusted_end_ms"),
                }
            })
            .collect()
    }
}
