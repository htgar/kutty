use std::path::{Component, Path, PathBuf};
use std::process::Command;

pub fn shell(cwd: &Path, command: &str) -> String {
    let out = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .output();
    match out {
        Ok(o) => {
            let mut s = String::new();
            s.push_str(&String::from_utf8_lossy(&o.stdout));
            s.push_str(&String::from_utf8_lossy(&o.stderr));
            if s.len() > 4000 {
                s.truncate(4000);
                s.push_str("\n[truncated]");
            }
            if s.is_empty() {
                s = format!("[exit {}]", o.status.code().unwrap_or(-1));
            }
            s
        }
        Err(e) => format!("error: {e}"),
    }
}

pub fn read_file(cwd: &Path, path: &str) -> String {
    let mut p = PathBuf::from(path);
    if p.is_relative() {
        p = cwd.join(p);
    }
    if !within_cwd(cwd, &p) {
        return "error: path escapes cwd".into();
    }
    match std::fs::read_to_string(&p) {
        Ok(s) => {
            if s.len() > 8000 {
                let mut t = s[..8000].to_string();
                t.push_str("\n[truncated]");
                t
            } else {
                s
            }
        }
        Err(e) => format!("error: {e}"),
    }
}

pub fn write_file(cwd: &Path, path: &str, content: &str) -> String {
    let mut p = PathBuf::from(path);
    if p.is_relative() {
        p = cwd.join(p);
    }
    if !within_cwd(cwd, &p) {
        return "error: path escapes cwd".into();
    }
    if let Some(parent) = p.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match std::fs::write(&p, content) {
        Ok(()) => format!("wrote {} bytes", content.len()),
        Err(e) => format!("error: {e}"),
    }
}

fn within_cwd(cwd: &Path, candidate: &Path) -> bool {
    let mut out = PathBuf::new();
    for c in candidate.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out.starts_with(cwd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_cwd_and_children() {
        let cwd = Path::new("/tmp/kutty-test");
        assert!(within_cwd(cwd, cwd));
        assert!(within_cwd(cwd, Path::new("/tmp/kutty-test/a.txt")));
        assert!(within_cwd(cwd, Path::new("/tmp/kutty-test/a/../b.txt")));
    }

    #[test]
    fn rejects_escapes() {
        let cwd = Path::new("/tmp/kutty-test");
        assert!(!within_cwd(cwd, Path::new("/tmp/kutty-test/../secret")));
        assert!(!within_cwd(cwd, Path::new("/etc/passwd")));
        assert!(!within_cwd(cwd, Path::new("/tmp/kutty-test-other/a")));
    }
}
