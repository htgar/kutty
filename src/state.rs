use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MEM_DIR: &str = ".agents/memory";
const SESS_DIR: &str = ".agents/sessions";
const TMP_DIR: &str = ".agents/tmp";
const KV_DIR: &str = ".agents/kv";

static SESSION_SEQ: AtomicU64 = AtomicU64::new(0);
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

pub fn ensure(cwd: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(cwd.join(MEM_DIR))?;
    std::fs::create_dir_all(cwd.join(SESS_DIR))?;
    std::fs::create_dir_all(cwd.join(TMP_DIR))?;
    std::fs::create_dir_all(cwd.join(KV_DIR))?;
    Ok(())
}

pub fn head(cwd: &Path, max: usize) -> String {
    let p = cwd.join(MEM_DIR).join("MEMORY.md");
    std::fs::read_to_string(&p)
        .map(|s| s.chars().take(max).collect())
        .unwrap_or_default()
}

pub fn session(cwd: &Path) -> std::io::Result<PathBuf> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dir = cwd.join(SESS_DIR);
    std::fs::create_dir_all(&dir)?;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let seq = SESSION_SEQ.fetch_add(1, Ordering::Relaxed);
    let p = dir.join(format!("{millis}-{}-{seq}.jsonl", std::process::id()));
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&p)?;
    Ok(p)
}

pub fn tmp_dir(cwd: &Path) -> PathBuf {
    cwd.join(TMP_DIR)
}

pub fn kv_dir(cwd: &Path) -> PathBuf {
    cwd.join(KV_DIR)
}

/// Path for a captured tool output file. The caller writes the contents.
pub fn tmp_path(cwd: &Path, ext: &str) -> std::io::Result<PathBuf> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dir = tmp_dir(cwd);
    std::fs::create_dir_all(&dir)?;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    Ok(dir.join(format!("{millis}-{}-{seq}.{ext}", std::process::id())))
}

pub fn log(path: &Path, line: &str) -> std::io::Result<()> {
    use std::io::Write;
    writeln!(
        std::fs::OpenOptions::new().append(true).open(path)?,
        "{line}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("kutty-state-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn sessions_are_unique() {
        let dir = test_dir("unique");
        let a = session(&dir).unwrap();
        let b = session(&dir).unwrap();
        assert_ne!(a, b);
        assert!(a.exists() && b.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn memory_head_is_capped() {
        let dir = test_dir("head");
        ensure(&dir).unwrap();
        std::fs::write(dir.join(MEM_DIR).join("MEMORY.md"), "x".repeat(100)).unwrap();
        assert_eq!(head(&dir, 10).chars().count(), 10);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn tmp_paths_are_unique() {
        let dir = test_dir("tmp");
        ensure(&dir).unwrap();
        let a = tmp_path(&dir, "log").unwrap();
        let b = tmp_path(&dir, "log").unwrap();
        assert_ne!(a, b);
        assert_eq!(a.parent(), Some(tmp_dir(&dir).as_path()));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
