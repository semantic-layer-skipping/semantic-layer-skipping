# Meeting Log

This document contains a log of notes for meetings throughout the project, sorted by date (most recent first).

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
