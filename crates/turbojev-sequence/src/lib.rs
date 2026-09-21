#![forbid(unsafe_code)]
#![deny(rust_2018_idioms)]

//! Exact multi-token candidate scoring utilities.
//!
//! Candidate codes are represented as a prefix-free trie. Each unique prefix
//! is evaluated once and every edge contributes an exact conditional
//! log-probability computed against the full vocabulary normalizer.

use std::collections::BTreeMap;
use thiserror::Error;

pub type TokenId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SequenceWork {
    pub naive_token_steps: usize,
    pub unique_prefix_evaluations: usize,
    pub shared_token_steps_saved: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Node {
    children: BTreeMap<TokenId, usize>,
    terminal: Option<usize>,
}

impl Node {
    fn new() -> Self {
        Self { children: BTreeMap::new(), terminal: None }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateTrie {
    nodes: Vec<Node>,
    candidate_count: usize,
    work: SequenceWork,
}

impl CandidateTrie {
    pub fn build(candidates: &[Vec<TokenId>]) -> Result<Self, SequenceError> {
        if candidates.len() < 2 {
            return Err(SequenceError::TooFewCandidates(candidates.len()));
        }
        let mut nodes = vec![Node::new()];
        let mut naive = 0usize;
        for (candidate_index, tokens) in candidates.iter().enumerate() {
            if tokens.is_empty() {
                return Err(SequenceError::EmptyCandidate(candidate_index));
            }
            naive = naive.saturating_add(tokens.len());
            let mut node_index = 0usize;
            for (depth, token) in tokens.iter().copied().enumerate() {
                if nodes[node_index].terminal.is_some() {
                    return Err(SequenceError::NotPrefixFree(candidate_index));
                }
                let child = match nodes[node_index].children.get(&token).copied() {
                    Some(index) => index,
                    None => {
                        let index = nodes.len();
                        nodes.push(Node::new());
                        nodes[node_index].children.insert(token, index);
                        index
                    }
                };
                node_index = child;
                if depth + 1 == tokens.len() {
                    if nodes[node_index].terminal.is_some() {
                        return Err(SequenceError::DuplicateCandidate(candidate_index));
                    }
                    if !nodes[node_index].children.is_empty() {
                        return Err(SequenceError::NotPrefixFree(candidate_index));
                    }
                    nodes[node_index].terminal = Some(candidate_index);
                }
            }
        }
        let edges = nodes.len().saturating_sub(1);
        let prefix_evals = nodes.iter().filter(|node| !node.children.is_empty()).count();
        let work = SequenceWork {
            naive_token_steps: naive,
            unique_prefix_evaluations: prefix_evals,
            shared_token_steps_saved: naive.saturating_sub(edges),
        };
        Ok(Self { nodes, candidate_count: candidates.len(), work })
    }

    pub const fn candidate_count(&self) -> usize { self.candidate_count }
    pub const fn work(&self) -> SequenceWork { self.work }

    /// Score every candidate with exact conditional log-likelihood.
    /// `next_logits(prefix)` must return the full-vocabulary next-token logits
    /// after the supplied candidate-prefix tokens have been appended to the
    /// already prepared prompt.
    pub fn score_with<E, F>(&self, mut next_logits: F) -> Result<Vec<f32>, SequenceScoreError<E>>
    where
        F: FnMut(&[TokenId]) -> Result<Vec<f32>, E>,
    {
        let mut scores = vec![f32::NEG_INFINITY; self.candidate_count];
        let mut prefix = Vec::new();
        self.visit(0, 0.0, &mut prefix, &mut scores, &mut next_logits)?;
        if scores.iter().any(|score| !score.is_finite()) {
            return Err(SequenceScoreError::IncompleteScores);
        }
        Ok(scores)
    }

    fn visit<E, F>(
        &self,
        node_index: usize,
        cumulative: f32,
        prefix: &mut Vec<TokenId>,
        scores: &mut [f32],
        next_logits: &mut F,
    ) -> Result<(), SequenceScoreError<E>>
    where
        F: FnMut(&[TokenId]) -> Result<Vec<f32>, E>,
    {
        let node = &self.nodes[node_index];
        if node.children.is_empty() {
            return Ok(());
        }
        let logits = next_logits(prefix).map_err(SequenceScoreError::Backend)?;
        let log_z = logsumexp(&logits).ok_or(SequenceScoreError::InvalidLogits)?;
        for (&token, &child_index) in &node.children {
            let token_index = usize::try_from(token).map_err(|_| SequenceScoreError::TokenOutOfRange(token))?;
            let token_logit = *logits.get(token_index).ok_or(SequenceScoreError::TokenOutOfRange(token))?;
            if !token_logit.is_finite() {
                return Err(SequenceScoreError::InvalidLogits);
            }
            let score = cumulative + token_logit - log_z;
            let child = &self.nodes[child_index];
            if let Some(candidate_index) = child.terminal {
                scores[candidate_index] = score;
            }
            if !child.children.is_empty() {
                prefix.push(token);
                self.visit(child_index, score, prefix, scores, next_logits)?;
                prefix.pop();
            }
        }
        Ok(())
    }
}

pub fn logsumexp(logits: &[f32]) -> Option<f32> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|value| f64::from(*value - maximum).exp()).sum();
    if !sum.is_finite() || sum <= 0.0 { return None; }
    Some(maximum + sum.ln() as f32)
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SequenceError {
    #[error("sequence scoring requires at least two candidates, got {0}")]
    TooFewCandidates(usize),
    #[error("candidate {0} is empty")]
    EmptyCandidate(usize),
    #[error("candidate {0} duplicates another token sequence")]
    DuplicateCandidate(usize),
    #[error("candidate {0} makes the codebook non-prefix-free")]
    NotPrefixFree(usize),
}

#[derive(Debug, Error)]
pub enum SequenceScoreError<E> {
    #[error("sequence scoring backend failed")]
    Backend(E),
    #[error("sequence scoring received invalid full-vocabulary logits")]
    InvalidLogits,
    #[error("candidate token {0} is outside the returned vocabulary")]
    TokenOutOfRange(TokenId),
    #[error("not every candidate received a finite score")]
    IncompleteScores,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trie_reuses_candidate_prefixes() {
        let trie = CandidateTrie::build(&[vec![1, 2], vec![1, 3], vec![4, 5]]).unwrap();
        assert_eq!(trie.work().naive_token_steps, 6);
        assert_eq!(trie.work().shared_token_steps_saved, 1);
    }

    #[test]
    fn exact_scores_use_full_vocabulary_normalizer() {
        let trie = CandidateTrie::build(&[vec![0, 1], vec![0, 2], vec![3, 1]]).unwrap();
        let mut calls = 0usize;
        let scores = trie.score_with::<(), _>(|prefix| {
            calls += 1;
            let logits = match prefix {
                [] => vec![3.0, 0.0, 0.0, 2.0],
                [0] => vec![0.0, 2.5, 1.0, 0.0],
                [3] => vec![0.0, 1.5, 0.0, 0.0],
                _ => unreachable!(),
            };
            Ok(logits)
        }).unwrap();
        assert_eq!(calls, 3);
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[1]);
        assert!(scores[0] > scores[2]);
    }

    #[test]
    fn rejects_prefix_codebooks() {
        let error = CandidateTrie::build(&[vec![1], vec![1, 2]]).unwrap_err();
        assert!(matches!(error, SequenceError::NotPrefixFree(_)));
    }
}
