use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use crate::parse;
use crate::state;
use crate::tools;

const MAX_STEPS: usize = 20;
const TOOL_BLURB: &str = r#"## Tool format
Call a tool with:
<function name="shell">{"command":"ls"}</function>
<function name="read_file">{"path":"README.md"}</function>
<function name="write_file">{"path":"a.md","content":"..."}</function>
<function name="finish">{"answer":"..."}</function>
Reply with exactly one <function> block per turn. No extra text."#;

pub struct Engine {
    backend: LlamaBackend,
    model: LlamaModel,
    ctx_size: u32,
}

impl Engine {
    pub fn load(path: &Path, ctx_size: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let backend = LlamaBackend::init()?;
        let params = LlamaModelParams::default();
        let pinned = Box::pin(params);
        let model = LlamaModel::load_from_file(&backend, path, &pinned)?;
        Ok(Self {
            backend,
            model,
            ctx_size,
        })
    }
}

pub fn run_turn(
    engine: &mut Engine,
    cwd: &Path,
    task: &str,
    think: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let session_path = state::session(cwd)?;
    state::log(&session_path, &format!(r#"{{"role":"user","content":{}}}"#, serde_json::to_string(task)?))?;

    let session_path = state::session(cwd)?;
    state::log(&session_path, &format!(r#"{{"role":"user","content":{}}}"#, serde_json::to_string(task)?))?;

    let mut conv = String::new();
    conv.push_str("<|im_start|>system\n");
    conv.push_str("## AGENTS.md\n");
    if let Ok(c) = std::fs::read_to_string(cwd.join("AGENTS.md")) {
        conv.push_str(&c);
        conv.push('\n');
    }
    conv.push_str("## MEMORY.md\n");
    conv.push_str(&state::head(cwd, 4096));
    conv.push_str("\n## Tools\n");
    conv.push_str(TOOL_BLURB);
    if think {
        conv.push_str("\n(Think step-by-step before answering.)\n");
    }
    conv.push_str("<|im_end|>\n");
    conv.push_str(&format!("<|im_start|>user\n{task}<|im_end|>\n"));
    conv.push_str("<|im_start|>assistant\n<function name=\"");

    let mut consecutive_parse_fails = 0;
    let mut last_tool: Option<(String, String)> = None;

    for _step in 0..MAX_STEPS {
        let reply = complete(engine, &conv)?;
        state::log(&session_path, &format!(r#"{{"role":"assistant","content":{}}}"#, serde_json::to_string(&reply)?))?;

        match parse::extract_call(&reply) {
            None => {
                println!("{reply}");
                consecutive_parse_fails += 1;
                state::log(&session_path, &format!(r#"{{"error":"parse_fail","step":{consecutive_parse_fails}}}"#))?;
                if consecutive_parse_fails >= 3 {
                    eprintln!("parse failed 3 times; raw reply above");
                    break;
                }
                conv.push_str(&reply);
                conv.push_str("<|im_end|>\n<|im_start|>user\nSystem: reply with exactly one <function> block.<|im_end|>\n<|im_start|>assistant\n<function name=\"");
            }
            Some(call) => {
                let cmd_key = format!("{}:{}", call.name, call.args);
                if let Some((prev_name, prev_key)) = &last_tool {
                    if prev_name == &call.name && prev_key == &cmd_key {
                        eprintln!("same tool call repeated; stopping");
                        break;
                    }
                }
                last_tool = Some((call.name.clone(), cmd_key));
                consecutive_parse_fails = 0;

                let out = match call.name.as_str() {
                    "shell" => {
                        let cmd = call.args.get("command").and_then(|v| v.as_str()).unwrap_or("");
                        println!("[shell] {cmd}");
                        tools::shell(cwd, cmd)
                    }
                    "read_file" => {
                        let path = call.args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                        println!("[read_file] {path}");
                        tools::read_file(cwd, path)
                    }
                    "write_file" => {
                        let path = call.args.get("path").and_then(|v| v.as_str()).unwrap_or("");
                        let content = call.args.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        println!("[write_file] {path}");
                        tools::write_file(cwd, path, content)
                    }
                    "finish" => {
                        let answer = call.args.get("answer").and_then(|v| v.as_str()).unwrap_or("");
                        println!("{answer}");
                        state::log(&session_path, &format!(r#"{{"tool":"finish","args":{}}}"#, serde_json::to_string(&call.args)?))?;
                        return Ok(());
                    }
                    other => {
                        eprintln!("unknown tool: {other}");
                        "unknown tool".to_string()
                    }
                };
                state::log(&session_path, &format!(r#"{{"tool":{},"args":{},"out":{}}}"#, serde_json::to_string(&call.name)?, serde_json::to_string(&call.args)?, serde_json::to_string(&out)?))?;
                conv.push_str(&reply);
                conv.push_str(&format!("<|im_end|>\n<|im_start|>user\nObservation: {out}<|im_end|>\n<|im_start|>assistant\n<function name=\""));
            }
        }
    }
    Ok(())
}

fn complete(engine: &mut Engine, prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(std::num::NonZeroU32::new(engine.ctx_size).unwrap()));
    let mut ctx = engine.model.new_context(&engine.backend, ctx_params)?;

    let tokens = engine.model.vocab().tokenize(prompt.as_bytes(), true, true);
    let mut batch = LlamaBatch::new(engine.ctx_size as usize, 1);
    let last = (tokens.len() as i32) - 1;
    for (i, tok) in (0_i32..).zip(tokens.into_iter()) {
        batch.add(tok, i, &[0], i == last)?;
    }
    ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::greedy()]);
    let mut cur = batch.n_tokens();
    let mut bytes: Vec<u8> = Vec::new();

    for _ in 0..1024 {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);
        if engine.model.vocab().is_eog(token) {
            break;
        }
        bytes.extend_from_slice(&engine.model.vocab().token_to_piece(token, true, None));
        let s = String::from_utf8_lossy(&bytes);
        if s.contains("</function>") || s.contains("<|im_end|>") {
            break;
        }
        batch.clear();
        batch.add(token, cur, &[0], true)?;
        cur += 1;
        ctx.decode(&mut batch)?;
    }
    let mut out = format!("<function name=\"{}", String::from_utf8_lossy(&bytes));
    if let Some(pos) = out.rfind("</function>") {
        out.truncate(pos + "</function>".len());
    }
    Ok(out)
}
