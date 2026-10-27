use std::path::{Path, PathBuf};
use std::time::Instant;

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde_json::Value;

use crate::state;
use crate::tools;

const MAX_STEPS: usize = 20;
const MAX_NEW_TOKENS: usize = 1024;
const MIN_FREE_TOKENS: usize = 64;
const DEFAULT_CTX: u32 = 4096;
const MIN_CTX: u32 = 512;
const KV_KEEP: usize = 8;

const TOOL_CONTRACT: &str = r#"## Tools
Reply with exactly one JSON object per turn and no other text:
{"name":"shell","arguments":{"command":"ls"}}
{"name":"read_file","arguments":{"path":"README.md","offset":1,"limit":200}}
{"name":"write_file","arguments":{"path":"notes.md","content":"..."}}
{"name":"finish","arguments":{"answer":"..."}}
Use finish when the task is complete. All paths are relative to the current directory."#;

/// Grammar for exactly one tool call per turn.
///
/// The `arguments` value is validated as JSON but not against a per-tool
/// schema; the model can still send bad arguments, which the harness reports
/// back as an observation. The grammar guarantees the envelope stays parseable
/// and the tool name is one we know.
const TOOL_CALL: &str = r#"root ::= ws "{" ws thought? "\"name\"" ws ":" ws name ws "," ws "\"arguments\"" ws ":" ws value ws "}" ws
thought ::= "\"thought\"" ws ":" ws string ws ","
name ::= "\"shell\"" | "\"read_file\"" | "\"write_file\"" | "\"finish\""
value ::= object | array | string | number | ("true" | "false" | "null")
object ::= "{" ws ( string ws ":" ws value ( ws "," ws string ws ":" ws value )* )? ws "}"
array ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\""
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
ws ::= [ \t\n\r]*
"#;

type RunResult<T> = Result<T, Box<dyn std::error::Error>>;

pub struct Engine {
    backend: LlamaBackend,
    model: LlamaModel,
    ctx_size: u32,
    template: LlamaChatTemplate,
}

impl Engine {
    pub fn load(path: &Path, ctx_size: Option<u32>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut backend = LlamaBackend::init()?;
        backend.void_logs();
        let params = LlamaModelParams::default();
        let pinned = Box::pin(params);
        let model = LlamaModel::load_from_file(&backend, path, &pinned)?;
        let ctx_size = ctx_size.unwrap_or_else(|| {
            let trained = model.n_ctx_train();
            if trained == 0 {
                DEFAULT_CTX
            } else {
                trained.clamp(MIN_CTX, DEFAULT_CTX)
            }
        });
        let template = model
            .chat_template(None)
            .unwrap_or_else(|_| LlamaChatTemplate::new("chatml").expect("static template"));
        Ok(Self {
            backend,
            model,
            ctx_size,
            template,
        })
    }

    fn render(&self, messages: &[LlamaChatMessage], add_assistant: bool) -> RunResult<String> {
        Ok(self
            .model
            .apply_chat_template(&self.template, messages, add_assistant)?)
    }

    fn tokenize(&self, text: &str) -> Vec<LlamaToken> {
        let vocab = self.model.vocab();
        let mut tokens = vocab.tokenize(text.as_bytes(), false, true);
        if vocab.should_add_bos() && tokens.first().copied() != Some(vocab.bos()) {
            tokens.insert(0, vocab.bos());
        }
        tokens
    }
}

pub fn run_turn(
    engine: &Engine,
    cwd: &Path,
    task: &str,
    think: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let session_path = state::session(cwd)?;
    log_json(
        &session_path,
        &serde_json::json!({"role": "user", "content": task}),
    )?;

    let mut messages = vec![
        LlamaChatMessage::new("system".into(), build_system(cwd, think))?,
        LlamaChatMessage::new("user".into(), task.to_string())?,
    ];

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(Some(std::num::NonZeroU32::new(engine.ctx_size).unwrap()));
    let mut ctx = engine.model.new_context(&engine.backend, ctx_params)?;
    let (mut ctx_tokens, cache_hit) = prepare_prefix(engine, cwd, &mut ctx, &messages)?;

    let mut last_sample_idx: i32 = if ctx_tokens.is_empty() { -1 } else { 0 };
    let mut parse_fails = 0usize;
    let mut last_call: Option<(String, String)> = None;

    for _step in 0..MAX_STEPS {
        let prompt = engine.render(&messages, true)?;
        let tokens = engine.tokenize(&prompt);
        let ctx_size = engine.ctx_size as usize;
        if tokens.len() + MIN_FREE_TOKENS > ctx_size {
            return Err(format!(
                "conversation is {} tokens but context is {ctx_size}; need at least {MIN_FREE_TOKENS} free tokens — raise --ctx",
                tokens.len()
            )
            .into());
        }

        let t_prefill = Instant::now();
        let common = common_prefix(&ctx_tokens, &tokens);
        if common < ctx_tokens.len() {
            let _ = ctx.clear_kv_cache_seq(Some(0), Some(common as u32), None);
            ctx_tokens.truncate(common);
        }
        if common < tokens.len() {
            last_sample_idx = decode_tokens(&mut ctx, &tokens[common..], common)?;
            ctx_tokens.extend_from_slice(&tokens[common..]);
        } else if last_sample_idx < 0 {
            return Err("internal error: empty prompt".into());
        }
        let prefill_ms = t_prefill.elapsed().as_millis() as u64;

        let max_new = (ctx_size - tokens.len()).min(MAX_NEW_TOKENS);
        let t_gen = Instant::now();
        let (reply, next_idx) = generate(
            engine,
            &mut ctx,
            TOOL_CALL,
            max_new,
            last_sample_idx,
            &mut ctx_tokens,
        )?;
        last_sample_idx = next_idx;
        let gen_ms = t_gen.elapsed().as_millis() as u64;
        let generated = ctx_tokens.len().saturating_sub(tokens.len());

        log_json(
            &session_path,
            &serde_json::json!({
                "timing": {
                    "prompt_tokens": tokens.len(),
                    "prefill_ms": prefill_ms,
                    "gen_ms": gen_ms,
                    "generated_tokens": generated,
                    "cache": if cache_hit { "hit" } else { "miss" },
                }
            }),
        )?;
        log_json(
            &session_path,
            &serde_json::json!({"role": "assistant", "content": reply}),
        )?;

        let Some(call) = parse_reply(&reply) else {
            parse_fails += 1;
            log_json(
                &session_path,
                &serde_json::json!({"error": "parse_fail", "step": parse_fails}),
            )?;
            if parse_fails >= 3 {
                return Err("model did not return a valid tool call".into());
            }
            messages.push(LlamaChatMessage::new("assistant".into(), reply)?);
            messages.push(LlamaChatMessage::new(
                "user".into(),
                "Your reply was not a valid tool call. Reply with exactly one JSON object using the tool format.".into(),
            )?);
            continue;
        };
        parse_fails = 0;

        let key = format!("{}\u{0}{}", call.name, call.args);
        if let Some((prev_name, prev_key)) = &last_call {
            if prev_name == &call.name && prev_key == &key {
                return Err("same tool call repeated; stopping".into());
            }
        }
        last_call = Some((call.name.clone(), key));

        let out = match call.name.as_str() {
            "shell" => {
                let cmd = call
                    .args
                    .get("command")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                println!("[shell] {cmd}");
                tools::shell(cwd, cmd)
            }
            "read_file" => {
                let path = call.args.get("path").and_then(Value::as_str).unwrap_or("");
                let offset = call
                    .args
                    .get("offset")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize);
                let limit = call
                    .args
                    .get("limit")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize);
                println!("[read_file] {path}");
                tools::read_file(cwd, path, offset, limit)
            }
            "write_file" => {
                let path = call.args.get("path").and_then(Value::as_str).unwrap_or("");
                let content = call
                    .args
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                println!("[write_file] {path}");
                tools::write_file(cwd, path, content)
            }
            "finish" => {
                let answer = call
                    .args
                    .get("answer")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                log_json(
                    &session_path,
                    &serde_json::json!({"tool": "finish", "args": call.args}),
                )?;
                println!("{answer}");
                return Ok(());
            }
            other => format!("unknown tool: {other}"),
        };

        log_json(
            &session_path,
            &serde_json::json!({"tool": call.name, "args": call.args, "out": out}),
        )?;
        messages.push(LlamaChatMessage::new("assistant".into(), reply)?);
        messages.push(LlamaChatMessage::new(
            "user".into(),
            format!("Observation: {out}"),
        )?);
    }

    Err(format!("step limit reached ({MAX_STEPS})").into())
}

fn build_system(cwd: &Path, think: bool) -> String {
    let mut system = String::new();
    if let Ok(agents) = std::fs::read_to_string(cwd.join("AGENTS.md")) {
        let agents = agents.trim();
        if !agents.is_empty() {
            system.push_str("## AGENTS.md\n");
            system.push_str(agents);
            system.push_str("\n\n");
        }
    }
    let memory = state::head(cwd, 2048);
    let memory = memory.trim();
    if !memory.is_empty() {
        system.push_str("## MEMORY.md\n");
        system.push_str(memory);
        system.push_str("\n\n");
    }
    if think {
        system.push_str(
            "Think through the task before acting. You may include a \"thought\" string field in each tool call.\n\n",
        );
    }
    system.push_str(TOOL_CONTRACT);
    system
}

struct Call {
    name: String,
    args: Value,
}

fn parse_reply(text: &str) -> Option<Call> {
    let Value::Object(map) = serde_json::from_str::<Value>(text.trim()).ok()? else {
        return None;
    };
    let name = map.get("name")?.as_str()?.to_string();
    Some(Call {
        name,
        args: map.get("arguments").cloned().unwrap_or(Value::Null),
    })
}

fn prepare_prefix(
    engine: &Engine,
    cwd: &Path,
    ctx: &mut LlamaContext<'_>,
    messages: &[LlamaChatMessage],
) -> RunResult<(Vec<LlamaToken>, bool)> {
    if engine.model.is_hybrid() || engine.model.is_recurrent() {
        return Ok((Vec::new(), false));
    }
    let full = engine.render(messages, true)?;
    let system_only = engine.render(&messages[..1], false)?;
    if !full.starts_with(&system_only) {
        return Ok((Vec::new(), false));
    }
    let prefix_tokens = engine.tokenize(&system_only);
    let full_tokens = engine.tokenize(&full);
    if prefix_tokens.is_empty() || !full_tokens.starts_with(&prefix_tokens) {
        return Ok((Vec::new(), false));
    }

    let dir = state::kv_dir(cwd);
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("{:016x}.bin", cache_key(engine, &system_only)));

    if path.is_file() {
        if let Ok((tokens, _)) = ctx.state_seq_load_file(&path, 0, engine.ctx_size as usize) {
            if tokens == prefix_tokens {
                return Ok((tokens, true));
            }
        }
        ctx.clear_kv_cache();
    }

    decode_tokens(ctx, &prefix_tokens, 0)?;
    let _ = ctx.state_seq_save_file(&path, 0, &prefix_tokens);
    prune_kv(&dir);
    Ok((prefix_tokens, false))
}

fn decode_tokens(
    ctx: &mut LlamaContext<'_>,
    tokens: &[LlamaToken],
    start_pos: usize,
) -> RunResult<i32> {
    let n_batch = ctx.n_batch().max(1) as usize;
    let mut batch = LlamaBatch::new(n_batch, 1);
    let mut sample_idx = 0;
    for (chunk_index, chunk) in tokens.chunks(n_batch).enumerate() {
        batch.clear();
        let base = chunk_index * n_batch;
        for (offset, token) in chunk.iter().enumerate() {
            let is_last = base + offset + 1 == tokens.len();
            batch.add(*token, (start_pos + base + offset) as i32, &[0], is_last)?;
        }
        ctx.decode(&mut batch)?;
        sample_idx = batch.n_tokens() - 1;
    }
    Ok(sample_idx)
}

fn generate(
    engine: &Engine,
    ctx: &mut LlamaContext<'_>,
    grammar: &str,
    max_new: usize,
    first_idx: i32,
    ctx_tokens: &mut Vec<LlamaToken>,
) -> RunResult<(String, i32)> {
    let grammar = LlamaSampler::grammar(&engine.model, grammar, "root")?;
    let mut sampler = LlamaSampler::chain_simple([
        grammar,
        LlamaSampler::temp(0.2),
        LlamaSampler::top_p(0.95, 1),
        LlamaSampler::dist(0),
    ]);
    let mut batch = LlamaBatch::new(1, 1);
    let mut bytes: Vec<u8> = Vec::new();
    let mut sample_idx = first_idx;
    let mut next_idx = first_idx;

    for _ in 0..max_new {
        let token = sampler.sample(ctx, sample_idx);
        if engine.model.vocab().is_eog(token) {
            break;
        }
        bytes.extend_from_slice(&engine.model.vocab().token_to_piece(token, false, None));
        let pos = ctx_tokens.len() as i32;
        batch.clear();
        batch.add(token, pos, &[0], true)?;
        ctx.decode(&mut batch)?;
        ctx_tokens.push(token);
        sample_idx = 0;
        next_idx = 0;
        let text = String::from_utf8_lossy(&bytes);
        let trimmed = text.trim_end();
        if trimmed.ends_with('}') && serde_json::from_str::<Value>(trimmed).is_ok() {
            break;
        }
    }

    Ok((String::from_utf8_lossy(&bytes).to_string(), next_idx))
}

fn common_prefix(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn cache_key(engine: &Engine, prefix: &str) -> u64 {
    let mut hash = Fnv::new();
    hash.update(&engine.model.size().to_le_bytes());
    hash.update(&engine.model.n_params().to_le_bytes());
    hash.update(&engine.ctx_size.to_le_bytes());
    if let Ok(template) = engine.template.to_str() {
        hash.update(template.as_bytes());
    }
    hash.update(&[0]);
    hash.update(prefix.as_bytes());
    hash.finish()
}

fn prune_kv(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(PathBuf, std::time::SystemTime)> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "bin") {
                return None;
            }
            let modified = entry.metadata().ok()?.modified().ok()?;
            Some((path, modified))
        })
        .collect();
    if files.len() <= KV_KEEP {
        return;
    }
    files.sort_by_key(|(_, modified)| *modified);
    for (path, _) in files.iter().take(files.len() - KV_KEEP) {
        let _ = std::fs::remove_file(path);
    }
}

fn log_json(path: &Path, value: &Value) -> std::io::Result<()> {
    state::log(path, &value.to_string())
}

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grammar_lists_every_tool() {
        for name in ["shell", "read_file", "write_file", "finish"] {
            assert!(
                TOOL_CALL.contains(&format!("\\\"{name}\\\"")),
                "missing {name}"
            );
        }
    }
}
