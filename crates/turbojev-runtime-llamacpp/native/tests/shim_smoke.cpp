#include "turbojev_llama_shim.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace {

bool close_enough(float left, float right) {
    return std::fabs(left - right) < 1e-6f;
}

void assert_row_eq(const float * left, const float * right, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        assert(close_enough(left[i], right[i]));
    }
}

std::vector<float> stub_logits(const std::vector<uint32_t> & tokens) {
    long long checksum = 0;
    for (uint32_t token : tokens) checksum += token;
    const float offset = static_cast<float>(checksum % 997) / 997.0f;
    const float length_term = static_cast<float>(tokens.size() % 31) / 31.0f;
    std::vector<float> logits(4096, -10.0f + offset);
    logits[100 + static_cast<int>('A')] = 1.0f + offset + 0.25f * length_term;
    logits[100 + static_cast<int>('B')] = 4.0f + 1.5f * offset - 0.10f * length_term;
    logits[100 + static_cast<int>('C')] = 0.5f - 0.75f * offset + 0.20f * length_term;
    return logits;
}

float token_logprob(const std::vector<uint32_t> & prefix, uint32_t token) {
    const auto logits = stub_logits(prefix);
    const float maximum = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (float value : logits) sum += std::exp(static_cast<double>(value - maximum));
    return logits[token] - (maximum + static_cast<float>(std::log(sum)));
}

float sequence_logprob(std::vector<uint32_t> prefix, const std::vector<uint32_t> & sequence) {
    float total = 0.0f;
    for (uint32_t token : sequence) {
        total += token_logprob(prefix, token);
        prefix.push_back(token);
    }
    return total;
}

} // namespace

int main() {
    char error[1024] = {};
    tj_llama_config config{};
    config.n_ctx = 4096;
    config.n_batch = 1024;
    config.n_seq_max = 8; // sequence 0 + up to seven parallel branches
    config.n_gpu_layers = 0;

    tj_llama_runtime * runtime = tj_llama_open("fixture.gguf", &config, error, sizeof(error));
    assert(runtime != nullptr);
    assert(tj_llama_abi_version() == TURBOJEV_LLAMA_SHIM_ABI_VERSION);
    assert(tj_llama_shared_prefix_supported(runtime) == 1);
    assert(tj_llama_parallel_suffix_supported(runtime) == 1);
    assert(tj_llama_multi_root_supported(runtime) == 1);
    assert(tj_llama_max_multi_root_tasks(runtime) == 8);
    assert(tj_llama_max_multi_root_tokens(runtime) == 1024);
    assert(tj_llama_max_parallel_tasks(runtime) == 7);
    assert(tj_llama_max_parallel_suffix_tokens(runtime) == 0);

    char desc[128] = {};
    const int64_t desc_len = tj_llama_model_desc(runtime, desc, sizeof(desc), error, sizeof(error));
    assert(desc_len > 0);
    assert(std::string(desc) == "stub-llama-model");

    const int64_t base_count = tj_llama_encode_chat(
        runtime, "system", "user", nullptr, nullptr, 0, error, sizeof(error));
    const int64_t answer_count = tj_llama_encode_chat(
        runtime, "system", "user", "A", nullptr, 0, error, sizeof(error));
    assert(base_count > 4);
    assert(answer_count == base_count + 1);

    std::vector<uint32_t> base(static_cast<size_t>(base_count));
    std::vector<uint32_t> answered(static_cast<size_t>(answer_count));
    assert(tj_llama_encode_chat(runtime, "system", "user", nullptr, base.data(), base.size(), error, sizeof(error)) == base_count);
    assert(tj_llama_encode_chat(runtime, "system", "user", "A", answered.data(), answered.size(), error, sizeof(error)) == answer_count);
    assert(std::equal(base.begin(), base.end(), answered.begin()));

    const uint32_t a = answered.back();
    const uint32_t b = 100u + static_cast<uint32_t>('B');
    const uint32_t c = 100u + static_cast<uint32_t>('C');
    char decoded[8] = {};
    assert(tj_llama_decode_tokens(runtime, &a, 1, decoded, sizeof(decoded), error, sizeof(error)) == 1);
    assert(std::string(decoded) == "A");

    const uint32_t candidates_ab[2] = {a, b};
    float reference_logits[2] = {};
    assert(tj_llama_score_next(runtime, base.data(), base.size(), candidates_ab, 2, reference_logits, error, sizeof(error)) == 0);
    assert(reference_logits[1] > reference_logits[0]);

    // For one-token codes, exact sequence log-probabilities differ from raw
    // next-token logits only by the same full-vocabulary log-normalizer. The
    // candidate distribution is therefore identical.
    const uint32_t single_sequence_tokens[2] = {a, b};
    const size_t single_sequence_offsets[3] = {0, 1, 2};
    float single_sequence_scores[2] = {};
    assert(tj_llama_score_sequences(
        runtime,
        base.data(),
        base.size(),
        single_sequence_tokens,
        single_sequence_offsets,
        2,
        single_sequence_scores,
        error,
        sizeof(error)) == 0);
    assert(close_enough(
        single_sequence_scores[1] - single_sequence_scores[0],
        reference_logits[1] - reference_logits[0]));

    assert(tj_llama_sequence_scores_supported(runtime) == 1);
    const std::vector<uint32_t> seq_aa = {a, a};
    const std::vector<uint32_t> seq_ab = {a, b};
    const std::vector<uint32_t> seq_ba = {b, a};
    const uint32_t sequence_tokens[6] = {a, a, a, b, b, a};
    const size_t sequence_offsets[4] = {0, 2, 4, 6};
    float independent_sequences[3] = {};
    assert(tj_llama_score_sequences(
        runtime,
        base.data(),
        base.size(),
        sequence_tokens,
        sequence_offsets,
        3,
        independent_sequences,
        error,
        sizeof(error)) == 0);
    assert(close_enough(independent_sequences[0], sequence_logprob(base, seq_aa)));
    assert(close_enough(independent_sequences[1], sequence_logprob(base, seq_ab)));
    assert(close_enough(independent_sequences[2], sequence_logprob(base, seq_ba)));

    const size_t prefix_count = base.size() - 2;
    assert(tj_llama_prefill_shared(runtime, base.data(), prefix_count, error, sizeof(error)) == 0);
    assert(tj_llama_max_parallel_suffix_tokens(runtime) == 1024);

    float shared_logits[2] = {};
    assert(tj_llama_score_suffix(
        runtime,
        base.data() + prefix_count,
        base.size() - prefix_count,
        candidates_ab,
        2,
        shared_logits,
        error,
        sizeof(error)) == 0);
    assert_row_eq(shared_logits, reference_logits, 2);

    float shared_sequences[3] = {};
    assert(tj_llama_score_suffix_sequences(
        runtime,
        base.data() + prefix_count,
        base.size() - prefix_count,
        sequence_tokens,
        sequence_offsets,
        3,
        shared_sequences,
        error,
        sizeof(error)) == 0);
    assert_row_eq(shared_sequences, independent_sequences, 3);

    // Empty suffixes are legal when two tasks share their complete prompt. The
    // shim retains final-prefix logits explicitly for that edge case.
    assert(tj_llama_prefill_shared(runtime, base.data(), base.size(), error, sizeof(error)) == 0);
    float empty_suffix_logits[2] = {};
    assert(tj_llama_score_suffix(
        runtime, nullptr, 0, candidates_ab, 2, empty_suffix_logits, error, sizeof(error)) == 0);
    assert_row_eq(empty_suffix_logits, reference_logits, 2);

    // Two different branches with different candidate sets must match serial
    // shared-prefix scoring exactly when evaluated together.
    assert(tj_llama_prefill_shared(runtime, base.data(), prefix_count, error, sizeof(error)) == 0);
    const std::vector<uint32_t> suffix_a(base.begin() + static_cast<std::ptrdiff_t>(prefix_count), base.end());
    const std::vector<uint32_t> suffix_b = {
        base[prefix_count],
        100u + static_cast<uint32_t>('X'),
        100u + static_cast<uint32_t>('Y'),
    };
    const uint32_t candidates_ac[2] = {a, c};
    float serial_a[2] = {};
    float serial_b[2] = {};
    assert(tj_llama_score_suffix(runtime, suffix_a.data(), suffix_a.size(), candidates_ab, 2, serial_a, error, sizeof(error)) == 0);
    assert(tj_llama_score_suffix(runtime, suffix_b.data(), suffix_b.size(), candidates_ac, 2, serial_b, error, sizeof(error)) == 0);

    std::vector<uint32_t> suffix_flat;
    suffix_flat.insert(suffix_flat.end(), suffix_a.begin(), suffix_a.end());
    suffix_flat.insert(suffix_flat.end(), suffix_b.begin(), suffix_b.end());
    const size_t suffix_offsets[3] = {0, suffix_a.size(), suffix_flat.size()};
    const uint32_t candidate_flat[4] = {a, b, a, c};
    const size_t candidate_offsets[3] = {0, 2, 4};
    float parallel_logits[4] = {};
    assert(tj_llama_score_suffix_batch(
        runtime,
        suffix_flat.data(),
        suffix_offsets,
        candidate_flat,
        candidate_offsets,
        2,
        parallel_logits,
        error,
        sizeof(error)) == 0);
    assert_row_eq(parallel_logits, serial_a, 2);
    assert_row_eq(parallel_logits + 2, serial_b, 2);

    // Mixed empty/non-empty suffixes preserve row order: empty branches read
    // retained prefix logits while non-empty branches read decode output rows.
    assert(tj_llama_prefill_shared(runtime, base.data(), base.size(), error, sizeof(error)) == 0);
    const std::vector<uint32_t> mixed_suffix = {100u + static_cast<uint32_t>('Z')};
    const size_t mixed_suffix_offsets[3] = {0, 0, 1};
    const uint32_t mixed_candidates[4] = {a, b, a, b};
    const size_t mixed_candidate_offsets[3] = {0, 2, 4};
    float mixed_parallel[4] = {};
    float mixed_serial[2] = {};
    assert(tj_llama_score_suffix(runtime, mixed_suffix.data(), 1, candidates_ab, 2, mixed_serial, error, sizeof(error)) == 0);
    assert(tj_llama_score_suffix_batch(
        runtime,
        mixed_suffix.data(),
        mixed_suffix_offsets,
        mixed_candidates,
        mixed_candidate_offsets,
        2,
        mixed_parallel,
        error,
        sizeof(error)) == 0);
    assert_row_eq(mixed_parallel, empty_suffix_logits, 2);
    assert_row_eq(mixed_parallel + 2, mixed_serial, 2);

    // Independent full prompts are evaluated as isolated sequence roots in one
    // llama_decode call. This is intentionally different from shared-prefix
    // suffix batching: each root starts at position zero and owns its own KV.
    const std::vector<uint32_t> root_a = base;
    std::vector<uint32_t> root_b = base;
    root_b.push_back(100u + static_cast<uint32_t>('Q'));
    float root_serial_a[2] = {};
    float root_serial_b[2] = {};
    assert(tj_llama_score_next(runtime, root_a.data(), root_a.size(), candidates_ab, 2, root_serial_a, error, sizeof(error)) == 0);
    assert(tj_llama_score_next(runtime, root_b.data(), root_b.size(), candidates_ac, 2, root_serial_b, error, sizeof(error)) == 0);

    std::vector<uint32_t> root_flat;
    root_flat.insert(root_flat.end(), root_a.begin(), root_a.end());
    root_flat.insert(root_flat.end(), root_b.begin(), root_b.end());
    const size_t root_offsets[3] = {0, root_a.size(), root_flat.size()};
    const uint32_t root_candidates[4] = {a, b, a, c};
    const size_t root_candidate_offsets[3] = {0, 2, 4};
    float root_fused[4] = {};
    assert(tj_llama_score_root_batch(
        runtime,
        root_flat.data(),
        root_offsets,
        root_candidates,
        root_candidate_offsets,
        2,
        root_fused,
        error,
        sizeof(error)) == 0);
    assert_row_eq(root_fused, root_serial_a, 2);
    assert_row_eq(root_fused + 2, root_serial_b, 2);

    // Multi-root stress varies root count, prompt lengths, token content and
    // ordering. Every row must match completely independent reference scoring.
    for (size_t round = 0; round < 48; ++round) {
        const size_t root_count = 2 + (round % 7); // 2..8
        std::vector<std::vector<uint32_t>> roots(root_count);
        std::vector<float> serial(root_count * 2, 0.0f);
        for (size_t root = 0; root < root_count; ++root) {
            roots[root] = base;
            const size_t extra = (round + root * 3) % 5;
            for (size_t i = 0; i < extra; ++i) {
                roots[root].push_back(
                    100u + static_cast<uint32_t>('k')
                    + static_cast<uint32_t>((round * 5 + root * 7 + i) % 12));
            }
            assert(tj_llama_score_next(
                runtime,
                roots[root].data(),
                roots[root].size(),
                candidates_ab,
                2,
                serial.data() + root * 2,
                error,
                sizeof(error)) == 0);
        }

        std::vector<uint32_t> flat_roots;
        std::vector<size_t> starts{0};
        std::vector<uint32_t> flat_candidates;
        std::vector<size_t> candidate_starts{0};
        for (const auto & root : roots) {
            flat_roots.insert(flat_roots.end(), root.begin(), root.end());
            starts.push_back(flat_roots.size());
            flat_candidates.push_back(a);
            flat_candidates.push_back(b);
            candidate_starts.push_back(flat_candidates.size());
        }
        std::vector<float> fused(root_count * 2, 0.0f);
        assert(tj_llama_score_root_batch(
            runtime,
            flat_roots.data(),
            starts.data(),
            flat_candidates.data(),
            candidate_starts.data(),
            root_count,
            fused.data(),
            error,
            sizeof(error)) == 0);
        assert_row_eq(fused.data(), serial.data(), fused.size());

        // Fused roots are cleared after the call; an independent request must
        // still match its original baseline exactly.
        float post_fused[2] = {};
        assert(tj_llama_score_next(
            runtime,
            roots[0].data(),
            roots[0].size(),
            candidates_ab,
            2,
            post_fused,
            error,
            sizeof(error)) == 0);
        assert_row_eq(post_fused, serial.data(), 2);
    }

    // Cache-isolation stress: repeatedly vary branch count, order, lengths and
    // token content. Every batched row must match the cached-serial baseline.
    for (size_t round = 0; round < 48; ++round) {
        const size_t task_count = 2 + (round % 6); // 2..7, within n_seq_max-1
        assert(tj_llama_prefill_shared(runtime, base.data(), prefix_count, error, sizeof(error)) == 0);

        std::vector<std::vector<uint32_t>> suffixes(task_count);
        std::vector<float> serial(task_count * 2, 0.0f);
        for (size_t task = 0; task < task_count; ++task) {
            const size_t width = (task + round) % 4; // includes empty branches
            for (size_t i = 0; i < width; ++i) {
                suffixes[task].push_back(
                    100u + static_cast<uint32_t>('a')
                    + static_cast<uint32_t>((round * 7 + task * 3 + i) % 20));
            }
            const uint32_t * ptr = suffixes[task].empty() ? nullptr : suffixes[task].data();
            assert(tj_llama_score_suffix(
                runtime,
                ptr,
                suffixes[task].size(),
                candidates_ab,
                2,
                serial.data() + task * 2,
                error,
                sizeof(error)) == 0);
        }

        std::vector<uint32_t> flat;
        std::vector<size_t> offsets;
        offsets.push_back(0);
        std::vector<uint32_t> candidates;
        std::vector<size_t> candidate_starts;
        candidate_starts.push_back(0);
        for (const auto & suffix : suffixes) {
            flat.insert(flat.end(), suffix.begin(), suffix.end());
            offsets.push_back(flat.size());
            candidates.push_back(a);
            candidates.push_back(b);
            candidate_starts.push_back(candidates.size());
        }
        std::vector<float> batched(task_count * 2, 0.0f);
        assert(tj_llama_score_suffix_batch(
            runtime,
            flat.empty() ? nullptr : flat.data(),
            offsets.data(),
            candidates.data(),
            candidate_starts.data(),
            task_count,
            batched.data(),
            error,
            sizeof(error)) == 0);
        assert_row_eq(batched.data(), serial.data(), batched.size());

        // A batch must leave sequence 0 immutable for subsequent serial work.
        float post_batch[2] = {};
        const uint32_t * first_ptr = suffixes[0].empty() ? nullptr : suffixes[0].data();
        assert(tj_llama_score_suffix(
            runtime,
            first_ptr,
            suffixes[0].size(),
            candidates_ab,
            2,
            post_batch,
            error,
            sizeof(error)) == 0);
        assert_row_eq(post_batch, serial.data(), 2);
    }

    tj_llama_close(runtime);

    // Multimodal prefill requests one sequence so its frames share the full
    // context window. Do not silently clamp that request back to two slots.
    config.n_seq_max = 1;
    runtime = tj_llama_open("fixture.gguf", &config, error, sizeof(error));
    assert(runtime != nullptr);
    assert(tj_llama_max_parallel_tasks(runtime) == 0);
    assert(tj_llama_multi_root_supported(runtime) == 0);
    tj_llama_close(runtime);
    return 0;
}
