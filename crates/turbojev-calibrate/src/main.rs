#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

use std::{env, fs, path::PathBuf, process::ExitCode};

use turbojev_calibration::{
    evaluate_calibration, fit_temperature_profile, CalibrationDataset, TemperatureFitConfig,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("turbojev-calibrate: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().ok_or_else(usage)?;
    match command.as_str() {
        "fit" => {
            let input = PathBuf::from(args.next().ok_or_else(usage)?);
            let output = PathBuf::from(args.next().ok_or_else(usage)?);
            let mut profile_id = "temperature-profile".to_owned();
            let mut config = TemperatureFitConfig::default();
            let rest: Vec<String> = args.collect();
            let mut index = 0;
            while index < rest.len() {
                match rest[index].as_str() {
                    "--id" => profile_id = value(&rest, &mut index, "--id")?,
                    "--ece-bins" => config.ece_bins = parse(&value(&rest, &mut index, "--ece-bins")?, "--ece-bins")?,
                    "--iterations" => config.iterations = parse(&value(&rest, &mut index, "--iterations")?, "--iterations")?,
                    "--min-temperature" => config.min_temperature = parse(&value(&rest, &mut index, "--min-temperature")?, "--min-temperature")?,
                    "--max-temperature" => config.max_temperature = parse(&value(&rest, &mut index, "--max-temperature")?, "--max-temperature")?,
                    other => return Err(format!("unknown option {other:?}\n{}", usage())),
                }
                index += 1;
            }
            let dataset = read_dataset(&input)?;
            let profile = fit_temperature_profile(&dataset, profile_id, config).map_err(|e| e.to_string())?;
            let json = serde_json::to_string_pretty(&profile).map_err(|e| e.to_string())?;
            fs::write(&output, format!("{json}\n")).map_err(|e| format!("{}: {e}", output.display()))?;
            print_metrics("before", &profile.before);
            print_metrics("after", &profile.after);
            println!("temperature: {:.9}", profile.temperature);
            println!("profile: {}", output.display());
        }
        "report" => {
            let input = PathBuf::from(args.next().ok_or_else(usage)?);
            let mut temperature = 1.0;
            let mut ece_bins = 15usize;
            let rest: Vec<String> = args.collect();
            let mut index = 0;
            while index < rest.len() {
                match rest[index].as_str() {
                    "--temperature" => temperature = parse(&value(&rest, &mut index, "--temperature")?, "--temperature")?,
                    "--ece-bins" => ece_bins = parse(&value(&rest, &mut index, "--ece-bins")?, "--ece-bins")?,
                    other => return Err(format!("unknown option {other:?}\n{}", usage())),
                }
                index += 1;
            }
            let dataset = read_dataset(&input)?;
            let metrics = evaluate_calibration(&dataset.examples, temperature, ece_bins).map_err(|e| e.to_string())?;
            print_metrics("report", &metrics);
        }
        "help" | "--help" | "-h" => println!("{}", usage()),
        _ => return Err(usage()),
    }
    Ok(())
}

fn read_dataset(path: &PathBuf) -> Result<CalibrationDataset, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let dataset: CalibrationDataset = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    dataset.validate().map_err(|e| e.to_string())?;
    Ok(dataset)
}

fn print_metrics(label: &str, metrics: &turbojev_calibration::CalibrationMetrics) {
    println!(
        "{label}: count={} accuracy={:.6} nll={:.6} brier={:.6} ece={:.6}",
        metrics.count, metrics.accuracy, metrics.nll, metrics.brier, metrics.ece
    );
}

fn value(args: &[String], index: &mut usize, option: &str) -> Result<String, String> {
    *index += 1;
    args.get(*index)
        .cloned()
        .ok_or_else(|| format!("missing value for {option}"))
}

fn parse<T: std::str::FromStr>(value: &str, option: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("invalid value {value:?} for {option}"))
}

fn usage() -> String {
    "usage:\n  turbojev-calibrate fit <dataset.json> <profile.json> [--id ID] [--ece-bins N] [--iterations N] [--min-temperature T] [--max-temperature T]\n  turbojev-calibrate report <dataset.json> [--temperature T] [--ece-bins N]".to_owned()
}
