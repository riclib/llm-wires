//! **What each Anthropic model accepts**, by its id: one function per
//! capability, so the request builder asks this table instead of carrying a
//! model list of its own. An id is read past any provider prefix
//! (`anthropic.`, `us.anthropic.`) to `claude-…`; an id that is not
//! recognisably Claude's — a gateway's own model behind this wire — is
//! answered as the wire always answered it.

/// A Claude id's family and generation: `claude-opus-4-6` is
/// `("opus", (4, 6))`, `claude-opus-4-20250514` `("opus", (4, 0))`, and the
/// old number-first naming (`claude-3-5-sonnet-…`) is `("", (3, 0))`.
/// `None` for an id that is not Claude's.
pub fn generation(model: &str) -> Option<(&str, (u32, u32))> {
    let at = model.find("claude-")?;
    let mut parts = model[at + "claude-".len()..].split(['-', '@', ':', '.']);
    let first = parts.next()?;
    if let Ok(major) = first.parse::<u32>() {
        return Some(("", (major, 0)));
    }
    let major = parts.next()?.parse::<u32>().ok()?;
    // A one- or two-digit part is the minor; a date (`20250514`) is not.
    let minor = parts
        .next()
        .filter(|m| m.len() <= 2)
        .and_then(|m| m.parse::<u32>().ok())
        .unwrap_or(0);
    Some((first, (major, minor)))
}

/// **Whether `model` takes `temperature`.** Anthropic's Messages API refuses
/// the sampling parameters (`temperature`, `top_p`, `top_k`) with a 400 on
/// Claude Opus 4.7 and every model after it — the 5.x line, Fable and Mythos;
/// on Sonnet 5.5 and Haiku 5.5 any value but the default — while 4.6 and
/// earlier take them under the `anthropic-version` this wire sends
/// (`2023-06-01`; from `2026-10-21` every model refuses them). So a Claude id
/// at 4.7 or later is sent no
/// `temperature`; a 4.6 or older one, and an id that is not Claude's, is sent
/// what it was asked. Fable and Mythos are 5.x, so the version says it.
pub fn takes_temperature(model: &str) -> bool {
    match generation(model) {
        None => true,
        Some((_, version)) => version <= (4, 6),
    }
}

/// **Whether `model` can be told to stop thinking** — `thinking: {"type":
/// "disabled"}` (solid #3783). Claude 4 and later take it, except the models
/// that always think and answer it with a 400: Opus 5.5, every Fable and
/// Mythos, and Sonnet 5.5 (whose nearest is `between_tools`, which is not
/// off). An older number-first id and an id that is not Claude's are sent
/// nothing, as before (Anthropic's model table,
/// platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting).
pub fn takes_thinking_off(model: &str) -> bool {
    match generation(model) {
        Some((family, version)) if !family.is_empty() && version.0 >= 4 => !matches!(
            (family, version),
            ("fable" | "mythos", _) | ("opus" | "sonnet", (5, 5))
        ),
        _ => false,
    }
}

/// **Whether `model` takes `output_config.effort`** (#3783): Opus 4.5, and
/// every Claude from 4.6 on. It applies to every output token, with thinking
/// or without (platform.claude.com/docs/en/build-with-claude/effort).
pub fn takes_effort(model: &str) -> bool {
    match generation(model) {
        Some((family, version)) if !family.is_empty() => {
            version >= (4, 6) || (family, version) == ("opus", (4, 5))
        }
        _ => false,
    }
}

/// **Whether `model` thinks adaptively** — `thinking: {"type": "adaptive"}`,
/// its depth set by the effort (#3783): every Claude from 4.6 on. The 4.5
/// line and before think only on a manual `budget_tokens`, which this wire
/// does not send.
pub fn takes_adaptive(model: &str) -> bool {
    matches!(generation(model), Some((family, version)) if !family.is_empty() && version >= (4, 6))
}
