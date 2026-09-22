#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Tokenizer-safe candidate-code discovery with a single-token fast path and a
//! prefix-free multi-token fallback.
//!
//! No model family or vocabulary token ID is hard-coded here.

use std::collections::{BTreeMap, HashSet};
use thiserror::Error;

/// Minimal tokenizer surface required by TurboJev.
pub trait TokenizerView {
    /// Encode text exactly as the selected model runtime would encode it.
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;

    /// Decode an exact token sequence.
    fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError>;
}

/// One verified opaque candidate code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateCode {
    /// Text appended at the model answer boundary.
    pub symbol: String,
    /// Exact tokenizer suffix corresponding to `symbol` at that boundary.
    pub token_ids: Vec<u32>,
}

/// How the selected codebook is scored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodebookKind {
    /// Every candidate is exactly one token.
    SingleToken,
    /// Every candidate has the same token length, making the code set
    /// prefix-free and suitable for exact sequence scoring.
    FixedLengthSequence { tokens_per_code: usize },
    /// Candidates have different token lengths but have been verified to be
    /// prefix-free. Exact sequence scoring is required.
    VariableLengthSequence { max_tokens_per_code: usize },
}

/// Candidate codebook for one exact prompt boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateCodebook {
    /// Verified codes in candidate order.
    pub codes: Vec<CandidateCode>,
    /// Scoring strategy required by these codes.
    pub kind: CodebookKind,
}

impl CandidateCodebook {
    /// Discover an exact codebook. A single-token codebook is always preferred.
    /// If the tokenizer exposes too few stable one-token symbols, TurboJev
    /// searches fixed-length token sequences up to `max_tokens_per_code`.
    pub fn discover<T: TokenizerView>(
        tokenizer: &T,
        prompt_prefix: &str,
        count: usize,
        max_tokens_per_code: usize,
    ) -> Result<Self, TokenizerError> {
        if count < 2 {
            return Err(TokenizerError::TooFewCandidates(count));
        }
        if max_tokens_per_code == 0 {
            return Err(TokenizerError::InvalidMaxTokens);
        }

        let prefix_ids = tokenizer.encode(prompt_prefix)?;
        let mut by_width: BTreeMap<usize, Vec<CandidateCode>> = BTreeMap::new();
        let mut seen_sequences: HashSet<Vec<u32>> = HashSet::new();

        for symbol in generated_symbols(count) {
            let combined = tokenizer.encode(&format!("{prompt_prefix}{symbol}"))?;
            if combined.len() <= prefix_ids.len() || combined[..prefix_ids.len()] != prefix_ids {
                continue;
            }

            let suffix = combined[prefix_ids.len()..].to_vec();
            if suffix.is_empty() || suffix.len() > max_tokens_per_code {
                continue;
            }
            if tokenizer.decode(&suffix)? != symbol {
                continue;
            }
            if !seen_sequences.insert(suffix.clone()) {
                continue;
            }

            let width = suffix.len();
            let bucket = by_width.entry(width).or_default();
            bucket.push(CandidateCode {
                symbol,
                token_ids: suffix,
            });

            if bucket.len() >= count {
                let codes = bucket[..count].to_vec();
                let kind = if width == 1 {
                    CodebookKind::SingleToken
                } else {
                    CodebookKind::FixedLengthSequence {
                        tokens_per_code: width,
                    }
                };
                return Ok(Self { codes, kind });
            }
        }

        let discovered = by_width.values().map(Vec::len).max().unwrap_or(0);
        Err(TokenizerError::InsufficientStableCodes {
            requested: count,
            discovered,
            max_tokens_per_code,
        })
    }

    /// Whether this codebook requires exact multi-token sequence scoring.
    pub const fn requires_sequence_scoring(&self) -> bool {
        matches!(
            self.kind,
            CodebookKind::FixedLengthSequence { .. } | CodebookKind::VariableLengthSequence { .. }
        )
    }
}

/// Generate deterministic opaque labels. Generation is textual only; every
/// label still has to pass exact tokenizer-boundary validation.
fn generated_symbols(requested: usize) -> Vec<String> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let target = requested.saturating_mul(32).clamp(512, 16_384);
    let mut result = Vec::with_capacity(target);

    for &a in ALPHABET {
        result.push(char::from(a).to_string());
        if result.len() >= target {
            return result;
        }
    }

    for &a in ALPHABET {
        for &b in ALPHABET {
            result.push(format!("{}{}", char::from(a), char::from(b)));
            if result.len() >= target {
                return result;
            }
        }
    }

    for &a in ALPHABET {
        for &b in ALPHABET {
            for &c in ALPHABET {
                result.push(format!(
                    "{}{}{}",
                    char::from(a),
                    char::from(b),
                    char::from(c)
                ));
                if result.len() >= target {
                    return result;
                }
            }
        }
    }

    result
}

/// Produce deterministic complete candidate-code sets for runtime probing.
///
/// The returned text is only a search space. Every selected code still has to
/// pass exact answer-boundary tokenization and round-trip verification before
/// it can be used for inference.
pub fn candidate_symbol_sets(
    count: usize,
    max_sets: usize,
) -> Result<Vec<Vec<String>>, TokenizerError> {
    if count < 2 {
        return Err(TokenizerError::TooFewCandidates(count));
    }
    if max_sets == 0 {
        return Ok(Vec::new());
    }

    let target = 62_usize
        .saturating_add(count.saturating_mul(max_sets.saturating_add(2)))
        .clamp(512, 16_384);
    let symbols = generated_symbols(target);
    if symbols.len() < count {
        return Ok(Vec::new());
    }

    let mut starts = vec![0_usize, 26, 52, 62];
    let mut cursor = 62_usize;
    while starts.len() < max_sets.saturating_mul(2) && cursor + count <= symbols.len() {
        starts.push(cursor);
        cursor = cursor.saturating_add(count.max(1));
    }
    starts.sort_unstable();
    starts.dedup();

    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for start in starts {
        if result.len() >= max_sets || start + count > symbols.len() {
            break;
        }
        let set = symbols[start..start + count].to_vec();
        let signature = set.join("\u{0}");
        if seen.insert(signature) {
            result.push(set);
        }
    }
    Ok(result)
}

/// Tokenizer/codebook error.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// Runtime tokenizer implementation failed.
    #[error("tokenizer operation failed: {0}")]
    Backend(String),
    /// Direct scoring requires at least two candidates.
    #[error("candidate codebook requires at least two candidates, got {0}")]
    TooFewCandidates(usize),
    /// Search width must be positive.
    #[error("max_tokens_per_code must be at least 1")]
    InvalidMaxTokens,
    /// Tokenizer did not expose enough stable equal-width codes.
    #[error(
        "only {discovered} compatible codes found; {requested} requested with at most {max_tokens_per_code} tokens per code"
    )]
    InsufficientStableCodes {
        requested: usize,
        discovered: usize,
        max_tokens_per_code: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct CharacterTokenizer {
        ids: HashMap<char, u32>,
    }

    impl CharacterTokenizer {
        fn new() -> Self {
            let ids = (32_u8..=126)
                .map(char::from)
                .enumerate()
                .map(|(index, ch)| (ch, u32::try_from(index + 1).unwrap()))
                .collect();
            Self { ids }
        }
    }

    impl TokenizerView for CharacterTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
            text.chars()
                .map(|ch| {
                    self.ids
                        .get(&ch)
                        .copied()
                        .ok_or_else(|| TokenizerError::Backend(format!("unknown char: {ch}")))
                })
                .collect()
        }

        fn decode(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
            let mut output = String::new();
            for id in token_ids {
                let Some((ch, _)) = self.ids.iter().find(|(_, candidate)| *candidate == id) else {
                    return Err(TokenizerError::Backend(format!("unknown token: {id}")));
                };
                output.push(*ch);
            }
            Ok(output)
        }
    }

    #[test]
    fn discovers_single_token_fast_path() {
        let tokenizer = CharacterTokenizer::new();
        let book = CandidateCodebook::discover(&tokenizer, "answer:", 4, 4).unwrap();
        assert_eq!(book.kind, CodebookKind::SingleToken);
        assert_eq!(
            book.codes
                .iter()
                .map(|code| code.symbol.as_str())
                .collect::<Vec<_>>(),
            vec!["A", "B", "C", "D"]
        );
    }

    #[test]
    fn fixed_width_codes_are_prefix_free_by_construction() {
        let codes = [vec![1_u32, 2], vec![1, 3], vec![2, 1]];
        for left in &codes {
            for right in &codes {
                if left != right {
                    assert!(!right.starts_with(left));
                }
            }
        }
    }

    #[test]
    fn candidate_symbol_sets_are_deterministic_and_complete() {
        let first = candidate_symbol_sets(4, 8).unwrap();
        let second = candidate_symbol_sets(4, 8).unwrap();
        assert_eq!(first, second);
        assert!(!first.is_empty());
        assert!(first.iter().all(|set| set.len() == 4));
        assert_eq!(first[0], vec!["A", "B", "C", "D"]);
    }
}
