#![forbid(unsafe_code)]

use std::path::PathBuf;

use turbojev_server::{HttpServer, ServerConfig};

fn main() {
    if let Err(error) = run() {
        eprintln!("turbojev-server: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let first = args.next();
    if matches!(
        first.as_deref().and_then(|value| value.to_str()),
        Some("--help" | "-h")
    ) {
        println!("Usage: turbojev-server --config <server.json>");
        return Ok(());
    }
    if first.as_deref().and_then(|value| value.to_str()) != Some("--config") {
        return Err("expected --config <server.json>".into());
    }
    let path = args
        .next()
        .map(PathBuf::from)
        .ok_or("missing config path")?;
    if args.next().is_some() {
        return Err("unexpected extra arguments".into());
    }
    let raw = std::fs::read_to_string(path)?;
    let config: ServerConfig = serde_json::from_str(&raw)?;
    HttpServer::open(config)?.serve()?;
    Ok(())
}
