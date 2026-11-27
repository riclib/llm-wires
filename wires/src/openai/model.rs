//! **What an OpenAI-shaped model takes for `reasoning_effort`**, by its id
//! (solid #3783). The OpenAI wire also carries Ollama's `/v1`, gateways and
//! any compatible server, so an id this table does not recognise as one of
//! OpenAI's own is sent the word as asked — `none` for `off`, which an Ollama
//! endpoint reads as no thinking and a model that does not think ignores
//! (0.40.11: qwen3.6 and nimble stop their trace; qwen3.6-nothink answers
//! the same either way).

use crate::Thinking;

/// An id as OpenAI spells it: lower case, past any `provider/` prefix.
fn bare(model: &str) -> String {
    model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .trim()
        .to_ascii_lowercase()
}

/// **The `reasoning_effort` a request on `model` sends for `thinking`**, or
/// `None` for none at all, said once:
/// - `gpt-4…`, `gpt-3…`, `chatgpt-4o…` reason not at all and refuse the field
///   at any value: never sent.
/// - the o-series (`o1`, `o3`, `o4-mini`…) take `low`, `medium`, `high` and
///   have no off: `off` sends nothing.
/// - `gpt-5`, `gpt-5-mini`, `gpt-5-nano` (and their dated ids) take
///   `minimal` as their least, not `none`: `off` sends `minimal`.
/// - `gpt-5.1` on, and every id not OpenAI's own, take all four: `off` is
///   `none` (gpt-5.4 answers it with no reasoning tokens).
pub(crate) fn effort(model: &str, thinking: Thinking) -> Option<&'static str> {
    let id = bare(model);
    let word = match thinking {
        Thinking::Off => "none",
        other => other.as_str(),
    };
    if ["gpt-4", "gpt-3", "chatgpt-4o"]
        .iter()
        .any(|p| id.starts_with(p))
    {
        return crate::said_once(
            model,
            "this model does not reason: `reasoning_effort` is not sent",
        )
        .then_some(word);
    }
    let o_series = ["o1", "o3", "o4"].iter().any(|p| {
        id.strip_prefix(p)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('-'))
    });
    if o_series && thinking == Thinking::Off {
        return crate::said_once(
            model,
            "this model cannot be told not to reason: `thinking: off` is not sent",
        )
        .then_some(word);
    }
    let gpt5_first = id == "gpt-5" || id.starts_with("gpt-5-");
    if gpt5_first && thinking == Thinking::Off {
        crate::said_once(
            model,
            "this model's least reasoning is `minimal`: `thinking: off` is sent as that",
        );
        return Some("minimal");
    }
    Some(word)
}

#[cfg(test)]
mod tests {
    use super::effort;
    use crate::Thinking;

    /// **Each OpenAI-shaped model is sent only what it takes** (#3783's
    /// read, S2): a model that does not reason is sent nothing; the o-series
    /// has no off; the first gpt-5 line says `minimal` for it; gpt-5.1 on,
    /// an Ollama model and a gateway's id say `none`.
    #[test]
    fn each_model_is_sent_only_the_effort_it_takes() {
        for model in ["gpt-4o", "gpt-4.1-mini", "openai/gpt-4o", "gpt-3.5-turbo"] {
            for t in [Thinking::Off, Thinking::High] {
                assert_eq!(effort(model, t), None, "{model} {t:?}");
            }
        }
        for model in ["o3", "o4-mini", "o1-2024-12-17", "o3-mini"] {
            assert_eq!(effort(model, Thinking::Off), None, "{model}");
            assert_eq!(effort(model, Thinking::Low), Some("low"), "{model}");
        }
        for model in ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-2025-08-07"] {
            assert_eq!(effort(model, Thinking::Off), Some("minimal"), "{model}");
            assert_eq!(effort(model, Thinking::Medium), Some("medium"), "{model}");
        }
        for model in [
            "gpt-5.1",
            "gpt-5.4",
            "gpt-5.4-mini",
            "qwen3.6:latest",
            "gemma4:12b-mlx",
            "o3x-local",
        ] {
            assert_eq!(effort(model, Thinking::Off), Some("none"), "{model}");
            assert_eq!(effort(model, Thinking::High), Some("high"), "{model}");
        }
    }
}
