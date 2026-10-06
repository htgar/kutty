use serde_json::Value;

pub struct Call {
    pub name: String,
    pub args: Value,
}

pub fn extract_call(text: &str) -> Option<Call> {
    let mut last: Option<Call> = None;
    let mut rest = text;
    while let Some(start) = rest.find("<function") {
        let block = &rest[start..];
        let Some(gt) = block.find('>') else { break };
        let head = &block[..gt];
        let parsed = (|| {
            let n0 = head.find("name=\"")?;
            let name_start = n0 + 6;
            let nend = head[name_start..].find('"')?;
            let name = head[name_start..name_start + nend].to_string();
            let end_tag = block.find("</function>")?;
            let body = block[gt + 1..end_tag].trim();
            let args = parse_args_body(body).unwrap_or(Value::Null);
            Some(Call { name, args })
        })();
        if let Some(call) = parsed {
            last = Some(call);
        }
        rest = if let Some(end_tag) = block.find("</function>") {
            &block[end_tag + 11..]
        } else {
            &block[gt + 1..]
        };
    }
    last
}

fn parse_args_body(body: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        return Some(v);
    }
    let mut depth = 0i32;
    let mut started = false;
    let mut end = None;
    for (i, ch) in body.char_indices() {
        match ch {
            '{' => { started = true; depth += 1; }
            '}' if started => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    if let Some(e) = end {
        if let Ok(v) = serde_json::from_str::<Value>(&body[..e]) {
            return Some(v);
        }
    }
    if body.contains("<param") {
        let mut map = serde_json::Map::new();
        let mut cursor = body;
        while let Some(i) = cursor.find("<param") {
            let rest = &cursor[i..];
            let Some(name_q) = rest.find("name=\"") else { break };
            let ns = name_q + 6;
            let Some(ne) = rest[ns..].find('"') else { break };
            let name = rest[ns..ns + ne].to_string();
            let Some(gt) = rest.find('>') else { break };
            let body_start = gt + 1;
            let Some(end) = rest[body_start..].find("</param>") else { break };
            let val = rest[body_start..body_start + end].trim();
            map.insert(name, Value::String(val.to_string()));
            cursor = &rest[body_start + end + 8..];
        }
        if !map.is_empty() {
            return Some(Value::Object(map));
        }
    }
    None
}

#[test]
fn parse_param_style_finish() {
    let s = r#"<function name="finish"><param name="answer">4</param></function>"#;
    let c = extract_call(s).unwrap();
    assert_eq!(c.name, "finish");
    assert_eq!(c.args["answer"], "4");
}
