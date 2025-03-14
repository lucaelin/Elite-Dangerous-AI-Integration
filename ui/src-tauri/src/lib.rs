use chrono::Local;
use serde_json::Value;
use std::env;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use tauri::path::BaseDirectory;
use tauri::Emitter;
use tauri::Manager;
use tauri::State;
use tokio::io::AsyncBufReadExt;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::Mutex;

// Define a function to get the commit hash, which will be set at build time
// If not set, it will default to "development"
fn get_commit_hash_value() -> &'static str {
    option_env!("COMMIT_HASH").unwrap_or("development")
}

const DEV_EXE_PATH: &str = "python";
const DEV_EXE_CWD: &str = "../..";
const DEV_EXE_ARGS: &[&str] = &["-u", "./src/Chat.py"];

const PROD_EXE_ARGS: &[&str] = &[];

// Logger for application communication
struct Logger {
    file: Arc<Mutex<File>>,
}

impl Logger {
    fn new(app_handle: &tauri::AppHandle) -> Result<Self, String> {
        // Create logs directory in the app data directory
        let log_dir = app_handle
            .path()
            .app_data_dir()
            .map_err(|e| format!("Failed to get app data directory: {}", e))?;

        std::fs::create_dir_all(&log_dir)
            .map_err(|e| format!("Failed to create log directory: {}", e))?;

        // Create log file with timestamp in filename
        let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let log_file_path = log_dir.join(format!("covas_log_{}.txt", timestamp));

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&log_file_path)
            .map_err(|e| format!("Failed to open log file: {}", e))?;

        println!("Logging to: {}", log_file_path.display());

        Ok(Logger {
            file: Arc::new(Mutex::new(file)),
        })
    }

    async fn log(&self, source: &str, message: &str) -> Result<(), String> {
        let timestamp = Local::now().format("[%Y-%m-%d %H:%M:%S%.3f]").to_string();
        let log_entry = format!("{} {}: {}\n", timestamp, source, message);

        let mut file = self.file.lock().await;
        file.write_all(log_entry.as_bytes())
            .map_err(|e| format!("Failed to write to log file: {}", e))?;
        file.flush()
            .map_err(|e| format!("Failed to flush log file: {}", e))?;

        Ok(())
    }

    // Helper function to redact sensitive information in messages
    fn redact_sensitive_info(message: &str) -> String {
        match serde_json::from_str::<Value>(message) {
            Ok(json_value) => {
                if let Some(msg_type) = json_value.get("type").and_then(|t| t.as_str()) {
                    if msg_type == "change_config" {
                        return "REDACTED: change_config".to_string();
                    } else if msg_type == "config" {
                        return "REDACTED: config".to_string();
                    }
                }
                message.to_string()
            }
            Err(_) => message.to_string(), // If not valid JSON, return as is
        }
    }
}

fn get_exe_config(
    window: &tauri::Window,
) -> Result<(String, String, &'static [&'static str]), String> {
    if cfg!(debug_assertions) {
        Ok((
            DEV_EXE_PATH.to_string(),
            DEV_EXE_CWD.to_string(),
            DEV_EXE_ARGS,
        ))
    } else {
        let app_handle = window.app_handle();
        let resource_path = app_handle
            .path()
            .resolve("resources/Chat", BaseDirectory::Resource)
            .map_err(|e| format!("Failed to resolve Chat executable: {}", e))?
            .to_str()
            .ok_or("Failed to convert resource path to string")?
            .to_string();

        let working_path = app_handle
            .path()
            .resolve("resources", BaseDirectory::Resource)
            .map_err(|e| format!("Failed to resolve Chat executable working directory: {}", e))?
            .to_str()
            .ok_or("Failed to convert working path to string")?
            .to_string();

        Ok((resource_path, working_path, PROD_EXE_ARGS))
    }
}

struct ProcessHandle {
    child: Child,
    stdin: Arc<tokio::sync::Mutex<ChildStdin>>,
}

#[derive(Default)]
struct AppState {
    process_handle: Arc<tokio::sync::Mutex<Option<ProcessHandle>>>,
    logger: Arc<Mutex<Option<Arc<Logger>>>>,
}

#[tauri::command]
async fn start_process(window: tauri::Window, state: State<'_, AppState>) -> Result<(), String> {
    // Initialize logger if it doesn't exist
    {
        let logger_lock = state.logger.lock().await;
        if logger_lock.is_none() {
            drop(logger_lock); // Release the lock before creating a new logger

            // Create new logger
            let logger = Logger::new(&window.app_handle())?;
            let logger_arc = Arc::new(logger);

            // Store logger in state
            let mut logger_lock = state.logger.lock().await;
            *logger_lock = Some(logger_arc);
        }
    }

    // Get logger reference
    let logger = {
        let logger_lock = state.logger.lock().await;
        logger_lock
            .as_ref()
            .expect("Logger should be initialized")
            .clone()
    };

    // Log process start
    logger.log("SYSTEM", "Starting process").await?;

    let mut proc_handle_lock = state.process_handle.lock().await;
    if proc_handle_lock.is_some() {
        let error_msg = "Process already running.";
        logger.log("ERROR", error_msg).await?;
        return Err(error_msg.into());
    }

    let (exe_path, exe_cwd, exe_args) = get_exe_config(&window)?;

    logger
        .log(
            "SYSTEM",
            &format!(
                "Launching: {} in {} with args: {:?}",
                exe_path, exe_cwd, exe_args
            ),
        )
        .await?;

    let mut command = Command::new(exe_path.clone());
    command
        .args(exe_args)
        .current_dir(exe_cwd.clone())
        .kill_on_drop(true)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let mut child = command.spawn().map_err(|e| {
        let error_msg = format!(
            "Failed to spawn process: {} - {} in {}",
            e, exe_path, exe_cwd
        );
        // We can't use .await in a closure used in .map_err, so log outside
        format!("{}", error_msg)
    })?;

    // Log successful process spawn
    logger
        .log(
            "SYSTEM",
            &format!("Process spawned with PID: {:?}", child.id()),
        )
        .await?;

    let stdout = child.stdout.take().ok_or("Failed to take child stdout")?;
    let stdin = child.stdin.take().ok_or("Failed to take child stdin")?;
    let stderr = child.stderr.take().ok_or("Failed to take child stderr")?;

    let stdin = Arc::new(Mutex::new(stdin));

    // Handle stdout
    tokio::spawn({
        let window = window.clone();
        let logger = logger.clone();
        async move {
            let mut reader = BufReader::new(stdout);
            let mut buffer = Vec::new();
            loop {
                match reader.read_until(b'\n', &mut buffer).await {
                    Ok(0) => break, // EOF reached
                    Ok(_n) => {
                        if let Ok(text) = String::from_utf8(buffer.clone()) {
                            let trimmed = text.trim_end();

                            // Redact sensitive information before logging
                            let log_message = Logger::redact_sensitive_info(trimmed);

                            println!("Process stdout: {}", log_message);

                            // Log redacted stdout to file
                            if let Err(e) = logger.log("STDOUT", &log_message).await {
                                eprintln!("Failed to log stdout: {}", e);
                            }

                            // Always emit the original text to the window
                            if let Err(e) = window.emit("process-stdout", text) {
                                eprintln!("Failed to emit process-stdout event: {}", e);

                                // Log emission error
                                if let Err(log_err) = logger
                                    .log(
                                        "ERROR",
                                        &format!("Failed to emit process-stdout event: {}", e),
                                    )
                                    .await
                                {
                                    eprintln!("Failed to log error: {}", log_err);
                                }
                            }
                        } else {
                            let error_msg = "Received invalid UTF-8 data";
                            eprintln!("{}", error_msg);

                            // Log encoding error
                            if let Err(e) = logger.log("ERROR", error_msg).await {
                                eprintln!("Failed to log error: {}", e);
                            }
                        }
                        buffer.clear();
                    }
                    Err(e) => {
                        let error_msg = format!("Error reading stdout: {}", e);
                        eprintln!("{}", error_msg);

                        // Log IO error
                        if let Err(log_err) = logger.log("ERROR", &error_msg).await {
                            eprintln!("Failed to log error: {}", log_err);
                        }
                        break;
                    }
                }
            }

            // Log when stdout handling ends
            if let Err(e) = logger.log("SYSTEM", "Process stdout stream ended").await {
                eprintln!("Failed to log: {}", e);
            }
        }
    });

    // Handle stderr
    tokio::spawn({
        let logger = logger.clone();
        async move {
            let mut reader = BufReader::new(stderr);
            let mut buffer = Vec::new();
            loop {
                match reader.read_until(b'\n', &mut buffer).await {
                    Ok(0) => break, // EOF reached
                    Ok(_n) => {
                        if let Ok(text) = String::from_utf8(buffer.clone()) {
                            let trimmed = text.trim_end();

                            // Redact sensitive information before logging
                            let log_message = Logger::redact_sensitive_info(trimmed);

                            eprintln!("Process stderr: {}", log_message);

                            // Log redacted stderr to file
                            if let Err(e) = logger.log("STDERR", &log_message).await {
                                eprintln!("Failed to log stderr: {}", e);
                            }
                        } else {
                            eprintln!("Received invalid UTF-8 data from stderr");

                            // Log encoding error
                            if let Err(e) = logger
                                .log("ERROR", "Received invalid UTF-8 data from stderr")
                                .await
                            {
                                eprintln!("Failed to log error: {}", e);
                            }
                        }
                        buffer.clear();
                    }
                    Err(e) => {
                        let error_msg = format!("Error reading stderr: {}", e);
                        eprintln!("{}", error_msg);

                        // Log IO error
                        if let Err(log_err) = logger.log("ERROR", &error_msg).await {
                            eprintln!("Failed to log error: {}", log_err);
                        }
                        break;
                    }
                }
            }

            // Log when stderr handling ends
            if let Err(e) = logger.log("SYSTEM", "Process stderr stream ended").await {
                eprintln!("Failed to log: {}", e);
            }
        }
    });

    let handle = ProcessHandle { child, stdin };
    *proc_handle_lock = Some(handle);

    logger.log("SYSTEM", "Process started successfully").await?;
    Ok(())
}

#[tauri::command]
async fn send_json_line(state: State<'_, AppState>, json_line: String) -> Result<(), String> {
    // Get logger reference
    let logger = {
        let logger_lock = state.logger.lock().await;
        logger_lock
            .as_ref()
            .ok_or("Logger not initialized")?
            .clone()
    };

    // Redact sensitive information before logging
    let log_message = Logger::redact_sensitive_info(&json_line);

    // Log the message being sent to stdin (redacted if needed)
    logger.log("STDIN", &log_message).await?;

    let stdin_arc = {
        let proc_handle_lock = state.process_handle.lock().await;
        let process = proc_handle_lock.as_ref().ok_or("Process is not running.")?;
        process.stdin.clone()
    };

    let mut stdin_guard = stdin_arc.lock().await;
    stdin_guard
        .write_all(json_line.as_bytes())
        .await
        .map_err(|e| {
            let error_msg = format!("Failed to write to stdin: {}", e);
            // We can't use .await in a closure used in .map_err, so return just the error
            error_msg
        })?;

    stdin_guard.flush().await.map_err(|e| {
        let error_msg = format!("Failed to flush stdin: {}", e);
        error_msg
    })?;

    // Don't log potentially sensitive information to console either
    println!("Wrote to stdin: {}", log_message);
    Ok(())
}

#[tauri::command]
async fn stop_process(state: State<'_, AppState>) -> Result<(), String> {
    // Get logger reference
    let logger = {
        let logger_lock = state.logger.lock().await;
        if let Some(logger) = logger_lock.as_ref() {
            logger.clone()
        } else {
            return Err("Logger not initialized".into());
        }
    };

    logger.log("SYSTEM", "Stopping process").await?;

    let mut proc_handle_lock = state.process_handle.lock().await;
    if let Some(handle) = proc_handle_lock.as_mut() {
        if let Err(e) = handle.child.kill().await {
            let error_msg = format!("Failed to kill process: {}", e);
            logger.log("ERROR", &error_msg).await?;
            return Err(error_msg);
        }
        logger.log("SYSTEM", "Process killed successfully").await?;
    } else {
        logger.log("SYSTEM", "No process running to stop").await?;
    }

    *proc_handle_lock = None;
    Ok(())
}

#[tauri::command]
fn get_commit_hash() -> String {
    get_commit_hash_value().to_string()
}

#[tauri::command]
async fn create_floating_overlay(app_handle: tauri::AppHandle) -> Result<(), String> {
    let mut window_builder = tauri::WebviewWindowBuilder::new(
        &app_handle,
        "overlay",
        tauri::WebviewUrl::App("index.html#/overlay".into()),
    );
    // First, get a reference to the main window
    let main_window = app_handle
        .get_webview_window("main")
        .ok_or_else(|| "Main window not found".to_string())?;

    window_builder = window_builder
        .title("COVAS:NEXT Overlay")
        .inner_size(480.0, 480.0)
        .decorations(false)
        .transparent(true)
        .always_on_top(true)
        .skip_taskbar(false)
        .maximized(true)
        //.fullscreen(true)
        .visible(true);

    let window = window_builder
        .parent(&main_window)
        .map_err(|e| format!("Failed to assign parent window: {}", e))?
        .build()
        .map_err(|e| format!("Failed to create floating overlay window: {}", e))?;

    // Make the window non-clickable (ignore cursor events)
    window
        .set_ignore_cursor_events(true)
        .map_err(|e| format!("Failed to set window to ignore cursor events: {}", e))?;

    println!("Created floating overlay window");

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
#[tokio::main]
pub async fn run() {
    // Print the commit hash at startup for debugging
    println!(
        "Starting application with commit hash: {}",
        get_commit_hash_value()
    );

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(AppState {
            process_handle: Arc::new(Mutex::new(None)),
            logger: Arc::new(Mutex::new(None)),
        })
        .invoke_handler(tauri::generate_handler![
            start_process,
            stop_process,
            send_json_line,
            get_commit_hash,
            create_floating_overlay
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let handle = window.app_handle().clone();
                tokio::spawn(async move {
                    let state: State<'_, AppState> = handle.state();
                    let _ = stop_process(state).await;
                });
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
