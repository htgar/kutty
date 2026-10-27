use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::state;

const SHELL_TIMEOUT: Duration = Duration::from_secs(120);
const TOOL_OUTPUT_MAX: usize = 4000;
const READ_MAX_CHARS: usize = 8000;
const READ_DEFAULT_LINES: usize = 200;

pub fn shell(cwd: &Path, command: &str) -> String {
    let outcome = match run_shell(cwd, command) {
        Ok(outcome) => outcome,
        Err(e) => return format!("error: {e}"),
    };

    let mut text = String::new();
    let stdout = clean(&String::from_utf8_lossy(&outcome.stdout));
    let stderr = clean(&String::from_utf8_lossy(&outcome.stderr));
    let stdout = stdout.trim_end();
    let stderr = stderr.trim_end();
    if !stdout.is_empty() {
        text.push_str(stdout);
    }
    if !stderr.is_empty() {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str("[stderr]\n");
        text.push_str(stderr);
    }
    if text.is_empty() {
        text.push_str("(no output)");
    }
    match outcome.code {
        Some(code) => text.push_str(&format!("\n[exit {code}]")),
        None => text.push_str(&format!("\n[timed out after {}s]", SHELL_TIMEOUT.as_secs())),
    }

    if text.chars().count() <= TOOL_OUTPUT_MAX {
        return text;
    }
    let saved = state::tmp_path(cwd, "log")
        .ok()
        .and_then(|path| std::fs::write(&path, &text).ok().map(|()| path));
    let mut out = head_tail(&text, TOOL_OUTPUT_MAX);
    if let Some(path) = saved {
        out.push_str(&format!("\n[full output: {}]", path.display()));
    }
    out
}

pub fn read_file(cwd: &Path, path: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    let mut p = PathBuf::from(path);
    if p.is_relative() {
        p = cwd.join(p);
    }
    if !within_cwd(cwd, &p) {
        return "error: path escapes cwd".into();
    }
    let offset = offset.unwrap_or(1);
    if offset < 1 {
        return "error: offset must be >= 1".into();
    }
    let limit = limit.unwrap_or(READ_DEFAULT_LINES).max(1);
    let text = match std::fs::read_to_string(&p) {
        Ok(text) => text,
        Err(e) => return format!("error: {e}"),
    };
    if text.is_empty() {
        return "(empty file)".into();
    }

    let lines: Vec<&str> = text.lines().collect();
    if offset > lines.len() {
        return format!(
            "error: offset {offset} is past the end ({} lines)",
            lines.len()
        );
    }

    let mut out = String::new();
    let mut shown = 0usize;
    for (i, line) in lines.iter().enumerate().skip(offset - 1).take(limit) {
        let rendered = format!("{:>5}\t{}\n", i + 1, line);
        let remaining = READ_MAX_CHARS.saturating_sub(out.chars().count());
        if rendered.chars().count() > remaining {
            if out.is_empty() {
                let budget = READ_MAX_CHARS.saturating_sub(2);
                let mut truncated: String = rendered.chars().take(budget).collect();
                truncated.push('\u{2026}');
                truncated.push('\n');
                out.push_str(&truncated);
                shown += 1;
            }
            break;
        }
        out.push_str(&rendered);
        shown += 1;
    }
    let start = offset;
    let end = offset + shown - 1;
    if shown < lines.len() {
        out.push_str(&format!(
            "[lines {start}-{end} of {}; use offset/limit for more]",
            lines.len()
        ));
    }
    out.trim_end().to_string()
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

struct ShellOutcome {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    code: Option<i32>,
}

fn run_shell(cwd: &Path, command: &str) -> std::io::Result<ShellOutcome> {
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let out = child.stdout.take().map(read_all);
    let err = child.stderr.take().map(read_all);

    let deadline = Instant::now() + SHELL_TIMEOUT;
    let mut code = None;
    loop {
        match child.try_wait()? {
            Some(status) => {
                code = status.code();
                break;
            }
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                break;
            }
            None => std::thread::sleep(Duration::from_millis(25)),
        }
    }

    let stdout = out.and_then(|h| h.join().ok()).unwrap_or_default();
    let stderr = err.and_then(|h| h.join().ok()).unwrap_or_default();
    Ok(ShellOutcome {
        stdout,
        stderr,
        code,
    })
}

fn read_all(mut reader: impl Read + Send + 'static) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = reader.read_to_end(&mut buf);
        buf
    })
}

fn clean(input: &str) -> String {
    let mut stripped = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            stripped.push(c);
            continue;
        }
        match chars.peek() {
            Some('[') => {
                chars.next();
                for c in chars.by_ref() {
                    if ('@'..='~').contains(&c) {
                        break;
                    }
                }
            }
            Some(']') => {
                chars.next();
                while let Some(c) = chars.next() {
                    if c == '\x07' {
                        break;
                    }
                    if c == '\x1b' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    let mut lines = Vec::new();
    for line in stripped.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        lines.push(line.rsplit('\r').next().unwrap_or(line));
    }
    lines.join("\n")
}

fn head_tail(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let half = max_chars / 2;
    let head: String = input.chars().take(half).collect();
    let tail: String = input
        .chars()
        .rev()
        .take(half)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!(
        "{head}\n[... {} characters truncated ...]\n{tail}",
        input.chars().count() - max_chars
    )
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

    #[test]
    fn head_tail_handles_multibyte() {
        let input = "é".repeat(5000);
        let out = head_tail(&input, 100);
        assert!(out.chars().count() > 100);
        assert!(out.contains("characters truncated"));
    }

    #[test]
    fn clean_strips_ansi_and_carriage_returns() {
        let out = clean("\x1b[31mred\x1b[0m\r\n50%\r100%\n");
        assert_eq!(out, "red\n100%\n");
    }

    #[test]
    fn read_file_truncates_long_lines() {
        let dir = std::env::temp_dir().join(format!("kutty-long-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("f.txt"), "x".repeat(20_000)).unwrap();
        let out = read_file(&dir, "f.txt", None, None);
        assert!(out.chars().count() <= READ_MAX_CHARS + 2);
        assert!(out.starts_with("    1\tx"));
        assert!(out.contains('\u{2026}'));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn read_file_numbers_and_pages() {
        let dir = std::env::temp_dir().join(format!("kutty-read-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("f.txt"), "a\nb\nc\n").unwrap();
        let out = read_file(&dir, "f.txt", Some(2), Some(2));
        assert!(out.contains("2\tb"));
        assert!(out.contains("3\tc"));
        assert!(!out.contains("1\ta"));
        assert!(out.contains("lines 2-3 of 3"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
