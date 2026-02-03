import logging
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import utils
from transformer_lens import HookedTransformer
from utils import PLOTS_DIR, get_device, question_to_prompt


def analyse_embedding_variance(model_name: str, prompts: list[str]):
    device = get_device()
    logging.info(f"Loading {model_name} on {device}...")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()

    # 1. collect hidden states for the last token of each prompt
    n_layers = model.cfg.n_layers
    n_prompts = len(prompts)
    layer_embeddings = []
    for prompt in prompts:
        _, cache = model.run_with_cache(prompt, return_type=None)

        # extract last token state for every layer
        prompt_layer_states = []  # shape: (n_layers, hidden_dim)
        for i in range(n_layers):
            state = cache[f"blocks.{i}.hook_resid_pre"][0, -1, :].cpu()
            prompt_layer_states.append(state)
        layer_embeddings.append(torch.stack(prompt_layer_states))

    # shape: (num_prompts, layers, hidden_dim)
    all_states = torch.stack(layer_embeddings)
    # shape: (layers, num_prompts, hidden_dim)
    all_states = all_states.permute(1, 0, 2)

    # compute metrics
    avg_inter_prompt_sim = []
    avg_drift = []
    avg_norms = []
    for i in range(n_layers):
        layer_data = all_states[i]  # shape: (num_prompts, hidden_dim)

        # metric 1: inter-prompt cosine similarity
        norm_data = F.normalize(layer_data, p=2, dim=1)
        sim_matrix = torch.mm(norm_data, norm_data.t())

        # exclude diagonal self-similarities
        mask = ~torch.eye(n_prompts, dtype=torch.bool)
        mean_sim = sim_matrix[mask].mean().item()
        avg_inter_prompt_sim.append(mean_sim)

        # metric 2: embedding drift (for same prompt)
        if i < n_layers - 1:
            next_layer_data = all_states[i + 1]
            drifts = F.cosine_similarity(layer_data, next_layer_data, dim=1)
            avg_drift.append(drifts.mean().item())

        # metric 3: average norm of embeddings (signal magnitude)
        norms = torch.norm(layer_data, p=2, dim=1)
        avg_norms.append(norms.mean().item())

    plt.figure(figsize=(18, 6))

    # plot 1: inter-prompt similarity
    plt.subplot(1, 3, 1)
    plt.plot(range(n_layers), avg_inter_prompt_sim, marker="o", color="purple")
    plt.title("Inter-Prompt Similarity")
    plt.xlabel("Layer Index")
    plt.ylabel("Avg Cosine Sim (Across Prompts)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.05)

    # plot 2: layer drift
    plt.subplot(1, 3, 2)
    plt.plot(range(n_layers - 1), avg_drift, marker="s", color="teal")
    plt.title("Layer Drift")
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Sim (Layer L vs Layer L+1)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    # plot 3: L2 norm analysis
    plt.subplot(1, 3, 3)
    plt.plot(range(n_layers), avg_norms, marker="^", color="orange")
    plt.title("Signal Magnitude (L2 Norm)")
    plt.xlabel("Layer Index")
    plt.ylabel("Avg Norm")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_dir = os.path.join(PLOTS_DIR, "hidden_state_analysis/")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "result.png"))
    plt.show()


if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    prompts = [question_to_prompt(q) for q in utils.PROMPTS]
    analyse_embedding_variance(MODEL_NAME, prompts)
