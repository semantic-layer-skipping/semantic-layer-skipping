# Meeting Log

This document contains a log of notes for meetings throughout the project, sorted by date (most recent first).

## 2026-05-04

**Progress so far**
1. K-nn strategies.
    - Default (top-1).
    - Safe-knn.
    - Consensus-decay.
    - semantic-boundary.
    - softmax-expected-skip: (crashed for some reason)
    ```
    slurm-28299181.out
    torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable
    Search for `cudaErrorDevicesUnavailable' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
    CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
    For debugging consider passing CUDA_LAUNCH_BLOCKING=1
    Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
    ```

2. Injection strategies.
   - (Default) Copy state into residual stream directly.
   - Scale: scale state by average vector norms for destination and source layers.
   - RMS: scale by RMS of the destination and source layers.
   - Affine: ((states - s_means) / s_stds) * t_std + t_mean
    ```
    Copy:   20185, 19814, 19080, 15468, 13179, 10036 vectors saved. (strict matching)
    Scalar: 20237, 19782, 19082, 15510, 13762, 10036 vectors saved. (strict matching)
    RMS:    20274, 19934, 19213, 15647, 13479, 10036 vectors saved. (strict matching)
    Affine: 20113, 19794, 19134, 15582, 13838, 10036 vectors saved. (strict matching)
    ```

3. KL Divergence population (threshold 2.0)
    ```
    Copy:   30043, 29973, 29604, 26262, 21604, 8834 vectors saved. (KL div 2.0)
    Scalar: 30093, 29992, 29624, 26172, 22000, 8834 vectors saved. (KL div 2.0)
    RMS:    29923, 29855, 29425, 25988, 21328, 8834 vectors saved. (KL div 2.0)
    Affine: 29889, 29712, 29437, 26057, 22711, 8834 vectors saved. (KL div 2.0)
    ```

4. Calibration by next-token accuracy thresholds.
    ```
    Precision 60.0%
    Checkpoint 0 (L4): Threshold 0.9249 | Keeps 72038/386104 (18.7%)
    Checkpoint 1 (L8): Threshold 0.2412 | Keeps 386104/386104 (100.0%)
    Checkpoint 2 (L12): Threshold 0.2959 | Keeps 386104/386104 (100.0%)
    Checkpoint 3 (L16): Threshold 0.8989 | Keeps 38181/386104 (9.9%)
    Checkpoint 4 (L20): Threshold 0.8774 | Keeps 41155/386104 (10.7%)
    Checkpoint 5 (L24): Threshold 0.7561 | Keeps 290520/386104 (75.2%)

    Precision 70.0%
    Checkpoint 0 (L4): Threshold 0.9768 | Keeps 6757/386104 (1.8%)
    Checkpoint 1 (L8): Threshold 0.9906 | Keeps 2115/386104 (0.5%)
    Checkpoint 2 (L12): Threshold 0.9515 | Keeps 10507/386104 (2.7%)
    Checkpoint 3 (L16): Threshold 0.9472 | Keeps 6244/386104 (1.6%)
    Checkpoint 4 (L20): Threshold 0.9170 | Keeps 10937/386104 (2.8%)
    Checkpoint 5 (L24): Threshold 0.8365 | Keeps 117798/386104 (30.5%)

    Precision 80.0%
    Checkpoint 0 (L4): Threshold 0.9868 | Keeps 1742/386104 (0.5%)
    Checkpoint 1 (L8): Threshold 0.9992 | Keeps 206/386104 (0.1%)
    Checkpoint 2 (L12): Threshold 0.9723 | Keeps 3402/386104 (0.9%)
    Checkpoint 3 (L16): Threshold 0.9622 | Keeps 3143/386104 (0.8%)
    Checkpoint 4 (L20): Threshold 0.9353 | Keeps 5162/386104 (1.3%)
    Checkpoint 5 (L24): Threshold 0.8731 | Keeps 57862/386104 (15.0%)

    Precision 90.0%
    Checkpoint 0 (L4): Threshold 0.9913 | Keeps 811/386104 (0.2%)
    Checkpoint 1 (L8): Threshold 1.0069 | Keeps 30/386104 (0.0%)
    Checkpoint 2 (L12): Threshold 0.9797 | Keeps 2351/386104 (0.6%)
    Checkpoint 3 (L16): Threshold 0.9775 | Keeps 1981/386104 (0.5%)
    Checkpoint 4 (L20): Threshold 0.9486 | Keeps 2875/386104 (0.7%)
    Checkpoint 5 (L24): Threshold 0.9073 | Keeps 29348/386104 (7.6%)

    Precision 95.0%
    Checkpoint 0 (L4): Threshold 0.9932 | Keeps 603/386104 (0.2%)
    Checkpoint 1 (L8): Threshold 1.0079 | Keeps 23/386104 (0.0%)
    Checkpoint 2 (L12): Threshold 0.9837 | Keeps 1991/386104 (0.5%)
    Checkpoint 3 (L16): Threshold 0.9960 | Keeps 985/386104 (0.3%)
    Checkpoint 4 (L20): Threshold 0.9565 | Keeps 2166/386104 (0.6%)
    Checkpoint 5 (L24): Threshold 0.9302 | Keeps 17995/386104 (4.7%)
    ```

5. Calibration by hit rate.
    ```
        Hit rate 0.05  Keeps 19305/386104 (5.0%)
        Checkpoint 0 (L4): Threshold 0.9618
        Checkpoint 1 (L8): Threshold 0.9610
        Checkpoint 2 (L12): Threshold 0.9376
        Checkpoint 3 (L16): Threshold 0.9193
        Checkpoint 4 (L20): Threshold 0.9011
        Checkpoint 5 (L24): Threshold 0.9274

        Hit rate 0.1 ( Keeps 38610/386104 (10.0%)    )

        Checkpoint 0 (L4): Threshold 0.9457
        Checkpoint 1 (L8): Threshold 0.9457
        Checkpoint 2 (L12): Threshold 0.9168
        Checkpoint 3 (L16): Threshold 0.8986
        Checkpoint 4 (L20): Threshold 0.8795
        Checkpoint 5 (L24): Threshold 0.8929

        Hit rate 0.15 (Keeps 57915/386104 (15.0%))
        Checkpoint 0 (L4): Threshold 0.9330
        Checkpoint 1 (L8): Threshold 0.9322
        Checkpoint 2 (L12): Threshold 0.9014
        Checkpoint 3 (L16): Threshold 0.8846
        Checkpoint 4 (L20): Threshold 0.8649
        Checkpoint 5 (L24): Threshold 0.8731
    ```

6. KV strategies.
   - Full compute
   - Copy
   - Project-only.

    ```
    Full compute
                    "avg_rouge_l": 0.16710126734649197,
                    "avg_token_accuracy": 0.07663007285637627,
                    "theoretical_speedup": 1.071357117891871,

    Project-only
                    "avg_rouge_l": 0.1586763247749071,
                    "theoretical_speedup": 1.0559588216380178,

    COPY
                    "avg_rouge_l": 0.1270323947414138,
                    "theoretical_speedup": 1.0393117582634945,
    ```

## 2026-04-23

**Progress so far**
1. Figures on the doc:
   - 4.8 and 4.9: most skips actually occur in the first checkpoint only, and then very few for the rest of the checkpoints. I think this is because of the fixed threshold used across checkpoints. (See point 3 below, calibration).
   - 4.10: while most tokens skip 0 layers, the rest is more diverse (e.g. skipping 3 blocks=12 layers is most common for others).
   - 4.11: longer generations usually mean more layers skipped (but not always).
   - 4.12 and 4.13: vector hit distributions in DB. There is somewhat of a Zipf-like pattern with some vectors being most popular, but the signal is still not very strong: e.g. 20k vectors hit in total, top 5 hit vectors form 1k (5%) of these hits.
2. Running experiments on 3B model (4 block size) and 1.5B model (2 block size), and the above patterns are basically still the same. Skipping-quality tradeoff is very similar as well, i.e. similar results to figures 4.4, 4.5, 4.6, 4.7.
3. Calibration: instead of having uniform thresholds, my calibration phase simulates skips that would be done by a vector DB (top-1 neighbour), and records which similarity-levels result in correct next-token. Early experiments show the initial checkpoint needs a very high threshold compared to middle layers (U-shaped pattern).
4. Vector DB size doesn't impact that much: 10% subsampled vectors gives similar results to 100% (i.e. full 20k generations).
5. K-nn approaches - introduced alternative approaches: top-1 neighbour, safe-knn, consensus-decay, semantic-boundary, softmax-expected-skip

**Additional designs and experiments**
- Threshold calibration/hyperparameter tuning with validation dataset.
- Can include checkpoint 0: just after position encoding. Since we see early checkpoints allow to skip more. But also, skipping early means no skipping is performed later, which might be problematic.
- We can experiment with GPTCache dataset as well.
- KV cache verification.
- PyTorch metrics.
- Other offline population approaches: use KL-divergence instead of strict token match.

**Systems-level virtual pipelining**
- nanoVLLM - doesn't support online serving, and no pipeline parallelism. They use flash attention library.
- miniSGLang - supports online serving and pipeline parallelism. 4000 lines: we would focus on (1) scheduler, (2) KV-cache methods (different kernels), (3) pipeline parallelism, and (4) benchmark, and how to adapt to our design.
  - This could impact radix caching performance as well.
  - We can copy the miniSGLang repo to the current repo (if it is small), or fork it.

### 2026-04-16

**Progress made up to meeting**
- Added frequency penalty, tracking of per-token skipping metrics and vector DB utilisation.
- Checkpoint Architecture plots: Figure 4.8 and Figure 4.9
- Token distribution plots: Figure 4.10 and Figure 4.11
- Vector DB utilisation plots: Figure 4.12 and Figure 4.13


**Title change** - can be changed to "Retrieval-based Semantic Layer Skipping for LLM Serving"

**Extra storage and GPU hours** - requests sent for 3 additional TB

**Next steps**
- Look at GPTCache dataset.
- Implement additional K-nn decision making, and run search evaluation.
- Improve calibration. Run calibration and thresholding experiments.
- Evaluate 2 checkpoints on larger model. (after storage increases)

### 2026-04-08

**Progress made up to meeting**
- Added repetition penalty, BERT-score evaluation, and evaluation against ShareGPT labels.
- Figure 4.4 (Label comparison for IVFPQ) shows skipped layer percentage and quality metric evaluated against the label.
The dashed yellow line indicates the baseline generation (no skipping) score against the label. Figure 4.5 is the same, but uses exact search instead.
- Figure 4.6 shows the scale factor (skipping generation score divided by baseline evaluation score) for IVFPQ. For skipping around 1%, it shows the token accuracy and BERT scores actually improve upon the baseline, and 10% skipping maintains around 0.9 of the baseline token accuracy and BERT score. Figure 4.7 shows the same scale factor plot for exact search.
- Also ran block-size 2 raw population (Qwen 1.5B, 10k samples to maintain the ~200GB size) and block-size 4 on Qwen 3B (10k samples). But paused for now due to storage limits.

**Storage limits** - we are currently limited by the 1TB storage limit on HPC, which is around 400GB for single index data (raw chunks 200GB, merged chunks 200GB, IVFPQ <5GB). We can ask to increase HPC storage. We can also store in cloud services, but 10Mbps for 1TB of data would take around 10 days - regenerating data (20 hours) might be faster.

**Incremental IVFPQ** - We can bypass the flat storage stage, adding to index after each mega-batch, but this would discard intermediate raw chunks. That said, this idea can be extended to an "online learning pattern" where you add selected online prompts/vectors to the index.

**Frequency penalty or Repetition Penalty** - it should be fine to use a repetition penalty only, although frequency penalty can be considered as well.

**IVFPQ tradeoff** - if storage is an issue, we can keep 1 exact index and experiment with many different IVFPQ hyperparameters, to get a better idea of the tradeoff curve.

**Training with different dataset sizes** - how much do tradeoff curves change as we change offline profiling dataset size?

**Next evaluation-based steps** - consider frequency penalty, keep track of per-block skipping distribution, vector DB items hit rate, other K-NN decision approaches.

### 2026-03-11

**Progress made up to meeting**
1. PyTorch runner - population phase optimisations:
    - Profiled each stage. 3 phases. Went down about 50x faster.
    - Cross-checkpoint batching (O(N))
    - Optimisations: (1) dynamic logit computation to reduce memory usage, (2) immutable KV caching, (3) vectorisation, (4) buffering database writes and compilation did not really help

2. Large-scale population:
    - ShareGPT filtered with vLLM approach: prompt length < 1024, with generation length < 2048.
    - After this filtering, 90k goes down to 35k samples (22522, 14965, 3712 items for training, calibration and evaluation).
    - Populating with 20k training samples results in 200GB (of dimension 1,536) across 6 checkpoints.
    - Since 200GB is too large, we perform chunked saving of results (every 2k samples, 20GB).
    - Script was run on slurm (took 14hr).

3. Merging databases with subsampling.
   - Since merging directly would need greater than 200GB memory, we merge random subsamples from each chunk.
   - Subsampling: e.g., get 10% from each. Roughly 600k vectors per checkpoint.

4. IVFPQ:
   - Flat - is too slow. Also, needs thresholding above 0.97 roughly, otherwise token accuracy is much lower.
   - IVFPQ: quite fast, but has almost no benefit for above 0.97 - inaccurate top-1?
   - Index construction: 160k training vectors, nlist = 4096, m = 64, nbits = 8.
   - Index search: n_probes = 128.
   - HPC: How to use multiple cpu threads for IVFPQ, set openmp threads but still ?

5. Threshold experiments:
   - Uniform checkpointing experiments run to get accuracy vs efficiency tradeoff.
   - Token accuracy diverges quickly.
   - Sometimes we get repeated behaviour. One example also shows switching to Chinese after an early-exit skip ("Amazon.com" in threshold 0.92)

**Training/Calibration/Indexing Next Steps**
   - **Flat index** - run small flat index experiments to get ideas for the trade-off curve for different thresholds, to isolate impact of IVFPQ.
   - **IVFPQ** - experiment with different hyperparameters, e.g., nlist, m, nbits, number of training vectors, nprobes. Also, investigate CPU usage (locally and on HPC) to increase number of used CPUs.
   - **Finer-blocks** - currently, we have 6 checkpoints of block width 4. We can experiment with finer blocks, e.g., 2 layers per block, to see if we can skip more layers while maintaining accuracy.
   - **Other models** - current model has 28 layers. We can experiment with other models: e.g., Qwen2.5-3B (36 layers, 3,072 hidden state), or MoE models.
   - **Calibration** - implement PyTorch calibration and optimise it to be batched. This should allow us to get non-uniform thresholds.

**Evaluation next steps**
   - **Comparison against ShareGPT ground truth** - currently, we compare skipping vs non-skipping output. But we should also compare against ShareGPT baseline, maybe scores improve with skipping - we expect both to be quite low overall.
   - **Embedding-based evaluation** - Instead of BLEU/Rouge, we can perform BERT embedding-based eval with cosine. Save outputs for now so we can perform this later if needed.
   - **Distribution of requests and blocks skipped** - instead of aggregate results, let's consider number of blocks skipped distribution. We can consider the length of prompts as well.
   - **Repetition penalty** - using temperature=0 can cause repeated behaviour in generation (not just specific to skipping). We can set repetition penalty to 1.2 or something to mitigate this ([Paper](https://arxiv.org/abs/2512.04419)). We can also ignore examples with such repetition.
   - **Vector DB items hit-rate** - gather statistics on how many times certain vector DB entries are being hit. Is it uniform? Is it skewed, e.g. spatial or temporal localities? These will inform how we can produce hierarchy of indexes. Hits can be determined in calibration phase as well.
   - **Other k-nn decision approaches** - currently, we just get top-1 neighbours decision. Can we improve on this? E.g., get k=5 and get worst decision of them?

### 2026-02-19

**Write-ups** - draft of initial sections of dissertation for next week,
    as well as a shorter experimental report with initial results from stage 1 experiments (using ShareGPT dataset from below) for 2 weeks time, to be written.

**Initial end-to-end results** - using the toy Isaac Newton dataset, we see that calibration works well: we can skip 17% layers while maintaining 95% token generation accuracy.
    Also, even with same prompting, vLLM has been shown to provide different outputs across runs, due to batching and float precision fluctuations, so some sort of inaccuracy is inherent anyways - we can compare against this.

**Improving experiment speed** - while Stage 1 is not focused on systems-level metrics and improvements, running large datasets (batch size of 1, repeated exact token match generation), take a long amount of time.
This can be improved by adding a form of batching to the repeated-population phase. PyTorch could be explored to see if it is better than TransformerLens for this purpose.
PyTorch metrics can also be considered. CPU nodes in HPC can be used as well (especially if using a low batch size): a lot more hours are provided compared to GPUs

**Other datasets** - while other task-based datasets (e.g., GSM8K, BOOLQ, MMLU) can be used for task-based accuracy, the initial focus is on unlabelled datasets (i.e., ShareGPT) for building offline banks. We can consider these datasets later.
Also note that some of these tasks expect only 1 token as output, which makes them less suitable for decode-only layer skipping. [PrefillOnly](https://arxiv.org/pdf/2505.07203) optimises LLM serving for these workloads.
For our case, we can also prompt the LLM to not generate 1 token, but think or reason before the final answer.

### 2026-02-06

**Checkpoint selection** - early results from [hidden state analysis PR](https://github.com/semantic-layer-skipping/semantic-layer-skipping/pull/9)
suggest that generally speaking, the early layers have more similar, lexical changes to the prompts in similar ways, whereas final layers have the greatest differences across diverse prompts.
  - Manual design: this motivates a manual checkpoint design of: 8 layers for first block, then 4 for the rest, maybe even 2 for the final layers (e.g., 8, 4, 4, 4, 2, 2 for a 24-layer model).
  - Automatic: can we use e.g., greedy/dynamic programming/integer linear programming to decide checkpoints in an online manner (see papers from last meeting). Similarly, we can use cosine analysis from PR above to automatically choose.
  - Dynamic design: additional complexity can be having different blocks for different prompts, determined dynamically at runtime. This must also be done very fast (ms-level), so it is difficult.

**Calibration** - early results from [calibration PR](https://github.com/semantic-layer-skipping/semantic-layer-skipping/pull/10).
We should extend this to auto-regressive calibration. Also, larger datasets (e.g., vLLM script) should give better insights.
vLLM benchmarking script can be directly imported (via import vllm) and use their dataset preprocessing code, as well as adding own preprocessing. This allows to use larger datasets. We can also use ShareGPT dataset, but it is quite out-of-date/deprecated.

**Systems papers** - DREX and Laser are the main papers to consider for stage 2, online serving/scheduling. Laser also performs layer-level scheduling;
however, they focus on Goodput, which involves minimising tail latency, and managing multi-SLO requirements. Here, we are considering more general reduction of latency, so minimising average latency.
Laser is also implemented on vLLM, like DREX, and it allows arbitrary number of layer-scheduling (e.g., 5 layers for one prompt, and 2 for another prompt). But they do not consider early-exit or layer skipping at all.

### 2026-01-23

**SGLang** - could also be considered for stage 2 integration, as an alternative to vLLM.
    It also has a mini version (around 4k lines), released recently (a couple of months ago), so unlike nanoVLLM,
    it is built on top on a more recent version so supports more up-to-date features, such as online serving,
    chunked prefills and writing your own kernels (CUDA or Triton).

**Adaptive Layer Importance** - [Initial results](https://github.com/AKafakA/semantic-layer-skipping/pull/2#issuecomment-3741078839)
show that Importance of layers can be token-dependent, so can we find importance of layers, and decide whether boundary checkpoints can be decided.
Also, it shows you need online table lookup since this is adaptive.

**Interleaved execution** - Within block, we can interleave and execute only a small set of these, e.g., skip every other layer (see PR results from above, indicating we can skip 1 layer fairly frequently, across layers)
   - Similar ideas to partitioned pipeline parallelism: Megatron-LM papers [first](https://arxiv.org/pdf/1909.08053) and the more-relevant [second](https://arxiv.org/pdf/2104.04473).
      Also, the micro-batch pipeline parallelism paper [GPipe](https://arxiv.org/pdf/1811.06965)
   - Interleaved partitioning or continuous partitioning.
   - However:
     - Interleaved might make the single matrix multiplication computations more difficult: as each compute might perform on another layer
     - The amount of needed KV re-computation is unclear/need to experiment with.

**Offline profiling** - deciding which layers to be the checkpoints for block skipping.
   - Can initially decide manually, based on profiling results, to determine most important layers.
   - Or, can use automatic algorithms, e.g., based on clustering of layer importance scores across many tokens/requests. Some ideas in [Alpa paper](https://www.usenix.org/system/files/osdi22-zheng-lianmin.pdf)

**Vector DB** - exact GPU, IVF-PQ, HNSW, GPU-based: these can be experimented with as an ablation studies.
   - Considerations: top-k accuracy, throughput limitations (CPU-GPU transfer), memory usage in GPU.

**Interpretability** - can we understand why certain layers are more important than others, linking to Transformer architecture and algorithms, rather than empirical results.
   - Can we connect to existing literature on layer importance, e.g., [ShortGPT](https://arxiv.org/abs/2403.03853).
   - Considering Transformer interpretability, feature visualisations etc., for example [layer importance paper](https://aclanthology.org/2024.blackboxnlp-1.29/)
   - These insights are based on initial experiments where first layers perform minimal updates beyond token embedding.

### 2026-01-16

- Block skipping, e.g., block size = 4 or 5 layers, to skip as opposed to arbitrary skipping
- Additional [layer-skipping paper](https://arxiv.org/abs/2601.02569) LoRA-finetuned layers instead of skipping
- System components: router (given hidden state decide which skipping decision), scheduler (decide which requests/block-level computations to serve next), model executor (within each GPU, how to compute), KV recomputation (runs recomputation kernels)
    - For pipelined multi-GPU setup, we might need multiple queues for lightweight kernels. Note: KV caches don't need to be shared across GPUs
    - PyTorch vs vLLM implementations.
    - Diminishing returns from larger KV caches, this memory can be better used for layer skipping caches (marginal utility gain).
    - PCIe between CPU-GPU can also be a bottleneck.

### 2026-01-07

**Initial profiling discussion** (based on [this PR](https://github.com/AKafakA/semantic-layer-skipping/pull/2)):
  - Initial approach is for early exiting, by measuring:
      - Option A - soft metric: Measure the KL-divergence between the true final token distribution, and the token distribution that would arise using the layer hidden state.
      - Option B - strict metric: the token with the highest probability/logit should match the true predicted token.
  - Next steps for layer skipping insights:
      - Similar to KL-divergence, measure how the hidden state embeddings change over layers. If, for a given request, certain layers have low change, then they can be skipped.
      - Or similarly, across many requests, average the change in hidden state embeddings per layer. Layers with low average change can be skipped (non-dynamic).
      - This can help to identify U-shaped patterns in layer importance, as seen in [ShortGPT](https://arxiv.org/abs/2403.03853).
      - Simulate layer skipping based on these metrics, and measure the impact on KL-divergence/top-1 accuracy. For example, put layer 10's output to layer 16 (skipping layers 11-15) and compute rest of the layers.
      - Decoding: greedy decoding initially, but can explore sampling-based decoding later.

**Models**:
  - Locally can run on MAC M4 GPU up to models like `Qwen/Qwen2.5-1.5B-Instruct`.
  - For larger models, can use Cambridge HPC cluster, departmental GPUs, or Google Colab.
  - For `instruct` models, need to ensure that the prompt formatting is correct (e.g., with system/user/assistant tags).

**Datasets**:
  - Unlabelled datasets (like ShareGPT) can be used for initial profiling and building offline banks.
      - This provides a lot more data, millions of examples. But we need to be careful about the size of the indexes.
      - Concrete datasets include ShareGPT subsets with cleaned conservations, e.g., [ShareGPT with 60k conversations](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json), and  [LMSYS Chat 1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
  - Task-specific datasets (like MMLU) can be used for measuring task-based accuracy. For example, (for early exit) earlier layers might perform well (or even better than later layers) on easier questions, while harder questions might need more layers.
  - Dataset sizes:
       - Typical conversational prompts produce 300 tokens. So, 10k training examples would produce up to 3 million per-layer token representations for indexing. However, not all internal embeddings will be cached, e.g., if we find we shouldn't skip certain layers.
       - More complex reasoning datasets (like [MMLU-pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) with 12k questions) can have longer responses (~1000 tokens), leading to more embeddings.
  - Can also look at robustness: offline index formed on ShareGPT, but test on out-of-distribution data (e.g., MMLU, or other datasets).

**Virtual Pipelining Proposal Discussion**:
  - See comments on the proposal overleaf doc.
  - High-level discussion points:
    - Project-only, repair and full computation kernels: can change the design/number of these kernels. CUDA/Triton implementation.
    - Multi-GPU setup: similar to pipeline parallelism. Single-GPU setup: concept of virtual pipelining and virtual queues.
    - CPU vs GPU communication: Scheduler is on CPU - so CPU-side vector index might not be too long. But you need to load GPU vectors from GPU to CPU. [Retrieval Attention](https://arxiv.org/abs/2409.10516) performs similar GPU-CPU co-execution, demonstrating reduced GPU memory footprint, although not discussing impact on end-to-end latency. [RAGCache](https://arxiv.org/abs/2404.12457) performs speculative retrieval of results from CPU to start RAG early.
    - Potential extension: pre-compute certain layers' outputs, and store them in the index. Precompute, or real-time compute of intermediate KV caches?
    - Protector formulation: similar to CacheBlend. Goal is to find weights to decide at runtime whether current tokens should be protected with higher accuracy KV cache estimation.
    - Other KV cache pruning methods, based on token importance as opposed to general compression/quantisation methods, can be explored. This would help to determine dynamically the KV computation kernel to be used for tokens being decoded. A lot of work has been done on this front, e.g., [LazyLLM](https://machinelearning.apple.com/research/dynamic-token-pruning).
    - For single-GPU setup, we can have a *single* scheduler managing all virtual queues, e.g., with deepest-first scheduling, serving from the deepest virtual queues first (which helps prevent starvation although this might not be the best global scheduler). Or, we can serve from each virtual queue in parallel, multiplexing the GPU (see MuxServe). A simpler scheduling approach can also be used, such as extending the vLLM scheduler to support non-priority based (e.g., FCFS) within each pipeline stage/queue

**vLLM integration**
  - Can initially prototype on [nanoVLLM](https://github.com/GeeeekExplorer/nano-vllm)
    - Advantage: small codebase (1200 lines, with 70 for scheduler), easier to understand and modify.
    - Disadvantages: lacks many features of vLLM, e.g, pipeline parallelism, which we would need to implement. Similarly, it doesn't support online inference, so performance profiling would be limited to offline batch inference without continuous batching/requests, or we would need to implement these features.
  - Following this, can port to vLLM itself.

### 2025-12-15

**TransformerLens** - for initial profiling and insights: e.g. do we see huge attention variance spikes.
- If results are highly dynamic, then a model-free skipping approach may need to be rethought.
- Models to test on: Qwen families (both MoE and non-MoE), e.g. [Qwen1.5-MoE-A2-7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B), and larger [Qwen3](https://arxiv.org/pdf/2505.09388) models (30B or 235B for MoE)

**PyTorch Prototype** - following insights, prototype layer skipping approach can be implemented in PyTorch, potentially with a model-free approach.

**Second Stage** - not necessarily coupled with the initial design and approach from above results. More systems-focused and integration with vLLM - will be refined further once stage 1 is complete.

**GPU Resources** - for initial experiments and models, e.g., smaller Qwen-0.5B models, Google Colab GPU/TPU is sufficient. Department GPU cluster is not necessary immediately for initial experiments, although would be useful for later. Other GPU resources exist as well.

**Literature review** - good to draft at these stages (on Overleaf), to make it easier to get feedback and write up later.
- Currently, general LLM serving survey is complete (see pdf in docs), and is sufficient to proceed with initial stage 1 experiments.
- Further reviews can be done on: (i) algorithmic side of layer skipping (see reference papers in references.bib), (ii) (for stage 2) systems side of layer skipping (e.g., is there an analysis for the tradeoff between KV cache memory and memory allocated for layer skipping), and (iii) additional related papers like LLM caching (see references.bib)

**Datasets** - additional notes on primary datasets to consider:
- vLLM benchmarks - high-quality benchmark suite, focused on system side, but will give a good idea on how benchmarks are structured.
- MoE-CAP - new dataset, with focus on MoE models, and systems tradeoffs. Good to explore for both stages.

**Current goal:** by end of January, have initial profiling with TransformerLens and design of layer skipping approach. If model-free approach is feasible, then prototype implementation can be started as well. Insights can be done on larger models later. Draft of approach can be started as well.


### 2025-12-08

The project can be divided into two main stages, with differing scope and goals.

Stage 1 – initial design, implementation and experiments focused on FLOPs reduction
- Using [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), implement layer skipping based on semantic caching.
  - Focus is on insights into flops-accuracy tradeoff, rather than end-to-end latency/throughput improvements (which depends on indexing method, cache size, replacement policy etc.)
- Models: Qwen2-0.5B (non-MoE), Llama3 (non-MoE). MoE ones as well (see project proposal).
  - Potentially different insights to be gained from different models.
- For initial experiments, use a few examples from datasets.
  - Potential datasets include: [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered), [LMSYS](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), [vLLM benchmarks](https://github.com/vllm-project/vllm/tree/main/benchmarks) (with prefix-cache aware dataset through `--dataset`), [MoE-CAP](https://openreview.net/pdf?id=k2fWVhG0u5), [Routing Arena](https://arxiv.org/abs/2310.04140), [GPTCache](https://github.com/zilliztech/GPTCache/blob/main/examples/benchmark/benchmark_sqlite_faiss_onnx.py), [SemanticRouter](https://github.com/vllm-project/semantic-router/tree/main/bench)
- Key metrics: FLOPs, accuracy.

Stage 2 – systems-level focus and integration with vLLM
- Refine approach, based on initial experiments from Stage 1, for end-to-end latency/throughput improvements
  - Key issue: synchronisation of layer-skipped tokens with non-skipped tokens within a batch.
- Memory profiling experiments to determine how much VRAM should be allocated to KV cache vs layer-skipping cache/model.
  - Measure inference efficiency against KV cache size: do we get diminishing returns after a certain size?
  - Based on results, design memory allocation strategy between the KV cache and layer-skipping cache/model.
- Introduce layer skipping to the vLLM scheduling/orchestration layer.
  - Can be semantic caching based skipping, or other adaptive skipping approaches.
- Key metrics: end-to-end latency, throughput/goodput, accuracy.

Other notes:
- Cambridge HPC: should be sufficient in terms of profiling and experiments (i.e., we won't need sudo access)
- Vector indexing: GPU Faiss vs CPU-based methods.
    - GPU Faiss avoids data transfer overheads, but more limited in terms of index types (e.g., MatMuls resulting in similarity but increasing FLOPs).
- Adding support to vLLM may be complex, depending on the design of vLLM's scheduling/orchestration layer.
    - May need to understand vLLM's internals better before proceeding with this part.

Updates to the workplan (from the project proposal):
- Up to 5 Jan: same as in proposal
- 5 Jan - 18 Jan:  set-up vector libraries and initial experiments with TransformerLens
- 19 Jan - 1 Feb:  implement simple semantic caching based layer skipping in TransformerLens, identify simple queries
- 2 Feb - 15 Feb:  prepare small workloads, experiments with different models, analyse FLOPs-accuracy tradeoffs
- 16 Feb - 1 Mar:  design and implement systems-aware caching strategies, understand vLLM internals
- 2 Mar onwards: same as in proposal, with testing and optimisation being for vLLM integration
