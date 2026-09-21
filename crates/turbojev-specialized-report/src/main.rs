#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use std::env;
use std::error::Error;
use std::fs;
use std::path::Path;

use serde::de::DeserializeOwned;
use turbojev_specialized::SpecializationEvaluationPlan;
use turbojev_specialized_eval::{evaluate_plan, SpecializationRun};

fn main() {
    if let Err(error) = run() {
        eprintln!("turbojev-specialized-report: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.as_slice() {
        [command, plan_path, output_path, run_paths @ ..]
            if command == "evaluate" && run_paths.len() >= 2 =>
        {
            let plan: SpecializationEvaluationPlan = read_json(plan_path)?;
            let runs: Vec<SpecializationRun> = run_paths
                .iter()
                .map(|path| read_json(path))
                .collect::<Result<_, _>>()?;
            let report_id = format!("{}-report", plan.id);
            let report = evaluate_plan(report_id, &plan, &runs)?;
            let encoded = serde_json::to_string_pretty(&report)?;
            fs::write(output_path, format!("{encoded}\n"))?;
            Ok(())
        }
        _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, usage()).into()),
    }
}

fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn usage() -> &'static str {
    "usage: turbojev-specialized-report evaluate <plan.json> <report.json> <generic-run.json> <specialized-run.json> [additional-run.json ...]"
}
