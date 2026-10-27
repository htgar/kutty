mod agent;
mod state;
mod tools;

use std::path::PathBuf;

const DEFAULT_REPO: &str = "openbmb/MiniCPM5-2B-GGUF";
const DEFAULT_QUANT: &str = "MiniCPM5-2B-Q4_K_M.gguf";
const MIN_CTX: u32 = 512;

#[derive(Default)]
struct Args {
    task: Option<String>,
    think: bool,
    model: Option<PathBuf>,
    ctx: Option<u32>,
    help: bool,
    version: bool,
}

fn usage() -> &'static str {
    "kutty — minimal local agent

usage: kutty [options] <task...>

  --model PATH       use a local GGUF instead of the default download
  --ctx N            context size (default: model context, 512-4096)
  --think            ask the model to reason before answering
  --                 treat all remaining arguments as task words
  -h, --help         show this help
  --version          show the version"
}

fn parse_args_from<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut parsed = Args::default();
    let mut words: Vec<String> = Vec::new();
    let mut it = args.into_iter();
    let mut flags_done = false;

    while let Some(arg) = it.next() {
        if flags_done {
            words.push(arg);
            continue;
        }
        match arg.as_str() {
            "--" => flags_done = true,
            "--think" => parsed.think = true,
            "--model" => {
                let value = it.next().ok_or("--model requires a path")?;
                parsed.model = Some(PathBuf::from(value));
            }
            "--ctx" => {
                let value = it.next().ok_or("--ctx requires a number")?;
                let ctx: u32 = value
                    .parse()
                    .map_err(|_| format!("invalid --ctx value: {value}"))?;
                if ctx < MIN_CTX {
                    return Err(format!("--ctx must be at least {MIN_CTX} (got {ctx})"));
                }
                parsed.ctx = Some(ctx);
            }
            "-h" | "--help" => parsed.help = true,
            "--version" => parsed.version = true,
            other if other.starts_with('-') && other != "-" => {
                return Err(format!("unknown option: {other}"));
            }
            other => words.push(other.to_string()),
        }
    }

    if !words.is_empty() {
        parsed.task = Some(words.join(" "));
    }
    Ok(parsed)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args_from(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(err) => {
            eprintln!("error: {err}\n\n{}", usage());
            std::process::exit(2);
        }
    };

    if args.help {
        println!("{}", usage());
        return Ok(());
    }
    if args.version {
        println!("kutty {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let Some(task) = args.task else {
        println!("{}", usage());
        return Ok(());
    };

    let cwd = std::env::current_dir()?;
    state::ensure(&cwd)?;

    let model_path = match args.model {
        Some(path) => path,
        None => hf_hub::api::sync::ApiBuilder::new()
            .with_progress(true)
            .build()?
            .model(DEFAULT_REPO.to_string())
            .get(DEFAULT_QUANT)?,
    };

    let engine = agent::Engine::load(&model_path, args.ctx)?;
    agent::run_turn(&engine, &cwd, &task, args.think)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, String> {
        parse_args_from(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn joins_unquoted_words() {
        let args = parse(&["what", "is", "2+2?"]).unwrap();
        assert_eq!(args.task.as_deref(), Some("what is 2+2?"));
        assert_eq!(args.ctx, None);
    }

    #[test]
    fn parses_flags() {
        let args = parse(&["--think", "--ctx", "8192", "--model", "m.gguf", "hello"]).unwrap();
        assert!(args.think);
        assert_eq!(args.ctx, Some(8192));
        assert_eq!(args.model.as_deref(), Some(std::path::Path::new("m.gguf")));
        assert_eq!(args.task.as_deref(), Some("hello"));
    }

    #[test]
    fn double_dash_keeps_task_flags() {
        let args = parse(&["--", "--think", "this"]).unwrap();
        assert!(!args.think);
        assert_eq!(args.task.as_deref(), Some("--think this"));
    }

    #[test]
    fn run_is_a_task_word() {
        let args = parse(&["run", "tests"]).unwrap();
        assert_eq!(args.task.as_deref(), Some("run tests"));
    }

    #[test]
    fn rejects_small_ctx() {
        assert!(parse(&["--ctx", "0", "hi"]).is_err());
        assert!(parse(&["--ctx", "511", "hi"]).is_err());
    }

    #[test]
    fn rejects_bad_ctx() {
        assert!(parse(&["--ctx", "lots", "hi"]).is_err());
    }

    #[test]
    fn rejects_missing_values() {
        assert!(parse(&["--ctx"]).is_err());
        assert!(parse(&["--model"]).is_err());
    }

    #[test]
    fn rejects_unknown_option() {
        assert!(parse(&["--wat", "hi"]).is_err());
    }

    #[test]
    fn help_and_version_do_not_need_a_task() {
        let help = parse(&["--help"]).unwrap();
        assert!(help.help);
        assert!(help.task.is_none());

        let version = parse(&["--version"]).unwrap();
        assert!(version.version);
        assert!(version.task.is_none());
    }
}
