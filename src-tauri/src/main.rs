// Tauri shell: opens the window and runs the Python sidecar next to it.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent};

const SIDECAR_PORT: u16 = 8765;
const APP_DIR_NAME: &str = "DualTrackDiarization";

struct Sidecar(Mutex<Option<Child>>);

fn sidecar_running() -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], SIDECAR_PORT));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn dev_sidecar_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("python-sidecar")
}

/// Python interpreter: env override, then the venv made by scripts/setup_mac.sh,
/// then a repo-local venv for development.
fn find_python() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DIARIZATION_PYTHON") {
        return Some(PathBuf::from(p));
    }
    let mut candidates = Vec::new();
    if let Ok(home) = std::env::var("HOME") {
        candidates.push(
            Path::new(&home)
                .join("Library/Application Support")
                .join(APP_DIR_NAME)
                .join("venv/bin/python"),
        );
    }
    candidates.push(dev_sidecar_dir().join(".venv/bin/python"));
    candidates.into_iter().find(|p| p.exists())
}

/// Sidecar sources: env override, then the copy bundled in the app, then the repo.
fn find_sidecar_dir(app: &tauri::AppHandle) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DIARIZATION_SIDECAR_DIR") {
        return Some(PathBuf::from(p));
    }
    let bundled = app
        .path()
        .resource_dir()
        .ok()
        .map(|d| d.join("python-sidecar"));
    [bundled, Some(dev_sidecar_dir())]
        .into_iter()
        .flatten()
        .find(|d| d.join("main.py").exists())
}

fn spawn_sidecar(app: &tauri::AppHandle) -> Option<Child> {
    if sidecar_running() {
        eprintln!("sidecar already running on port {SIDECAR_PORT}, reusing it");
        return None;
    }
    let python = match find_python() {
        Some(p) => p,
        None => {
            eprintln!("Python environment not found. Run scripts/setup_mac.sh first.");
            return None;
        }
    };
    let dir = find_sidecar_dir(app)?;
    match Command::new(python)
        .arg("main.py")
        .current_dir(&dir)
        .env("SIDECAR_PORT", SIDECAR_PORT.to_string())
        .stdin(Stdio::null())
        .spawn()
    {
        Ok(child) => Some(child),
        Err(err) => {
            eprintln!("failed to start sidecar: {err}");
            None
        }
    }
}

fn main() {
    let app = tauri::Builder::default()
        .manage(Sidecar(Mutex::new(None)))
        .setup(|app| {
            let child = spawn_sidecar(app.handle());
            *app.state::<Sidecar>().0.lock().unwrap() = child;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the app");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(mut child) = handle.state::<Sidecar>().0.lock().unwrap().take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    });
}
