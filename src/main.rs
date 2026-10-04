mod agent;
mod parse;
mod state;
mod tools;

use std::path::PathBuf;

const DEFAULT_REPO: &str = "openbmb/MiniCPM5-2B-GGUF";
const DEFAULT_QUANT: &str = "MiniCPM5-2B-Q4_K_M.gguf";

struct Args {
    task: Option<String>,
    think: bool,
    model: Option<PathBuf>,
    ctx: u32,
}

impl Default for Args {
    fn default() -> Self {
        Self { task: None, think: false, model: None, ctx: 4096 }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let mut words: Vec<String> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "run" => {}
            "--think" => a.think = true,
            "--model" => a.model = it.next().map(PathBuf::from),
            "--ctx" => a.ctx = it.next().and_then(|v| v.parse().ok()).unwrap_or(4096),
            "-h" | "--help" => {
                println!("kutty — minimal local agent\n\n  kutty <task...>      one-shot\n  --model PATH       use a local GGUF\n  --think            enable thinking\n  --ctx N            context size (default 4096)");
                std::process::exit(0);
            }
            other if !other.starts_with('-') => words.push(other.into()),
            _ => {}
        }
    }
    if !words.is_empty() {
        a.task = Some(words.join(" "));
    }
    a
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let cwd = std::env::current_dir()?;
    state::ensure(&cwd)?;

    let model_path = match args.model {
        Some(p) => p,
        None => hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .build()?
            .model(DEFAULT_REPO.to_string())
            .get(DEFAULT_QUANT)?,
    };

    let mut engine = agent::Engine::load(&model_path, args.ctx)?;

    match args.task {
        Some(task) => agent::run_turn(&mut engine, &cwd, &task, args.think)?,
        None => println!("usage: kutty <task...> [--think] [--model PATH] [--ctx N]"),
    }
    Ok(())
}
