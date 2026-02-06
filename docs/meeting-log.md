# Meeting Log

This document contains a log of notes for meetings throughout the project, sorted by date (most recent first).

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
