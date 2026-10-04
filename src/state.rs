use std::path::Path;

const MEM_DIR: &str = ".agent/memory";
const SESS_DIR: &str = ".agent/sessions";

pub fn ensure(cwd: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(cwd.join(MEM_DIR))?;
    std::fs::create_dir_all(cwd.join(SESS_DIR))?;
    Ok(())
}

pub fn head(cwd: &Path, max: usize) -> String {
    let p = cwd.join(MEM_DIR).join("MEMORY.md");
    std::fs::read_to_string(&p).map(|s| s.chars().take(max).collect()).unwrap_or_default()
}

pub fn session(cwd: &Path) -> std::io::Result<std::path::PathBuf> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dir = cwd.join(SESS_DIR);
    std::fs::create_dir_all(&dir)?;
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let p = dir.join(format!("{secs}.jsonl"));
    std::fs::OpenOptions::new().create(true).append(true).open(&p)?;
    Ok(p)
}

pub fn log(path: &Path, line: &str) -> std::io::Result<()> {
    use std::io::Write;
    writeln!(std::fs::OpenOptions::new().append(true).open(path)?, "{line}")
}
