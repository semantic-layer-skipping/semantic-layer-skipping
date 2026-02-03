import logging
import os

import matplotlib.pyplot as plt
import numpy as np
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

    # find the index of the last real token for each prompt
    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token = model.tokenizer.eos_token  # ensure pad token exists
    # tokenise with enforced left padding
    tokens = model.to_tokens(prompts, prepend_bos=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type=None)

    layer_embeddings = []
    for i in range(n_layers):
        # shape: [batch, max_seq_len, hidden_dim]
        layer_states = cache[f"blocks.{i}.hook_resid_pre"]

        # since we forced left-padding, all the last tokens are at index -1
        last_token_states = layer_states[:, -1, :].cpu()
        layer_embeddings.append(last_token_states)

    # shape: (layers, num_prompts, hidden_dim)
    all_states = torch.stack(layer_embeddings)

    # compute metrics
    avg_inter_prompt_sim = []
    std_inter_prompt_sim = []
    avg_drift = []
    std_drift = []
    avg_norms = []
    std_norms = []
    for i in range(n_layers):
        layer_data = all_states[i]  # shape: (num_prompts, hidden_dim)

        # metric 1: inter-prompt cosine similarity
        norm_data = F.normalize(layer_data, p=2, dim=1)
        sim_matrix = torch.mm(norm_data, norm_data.t())

        # exclude diagonal self-similarities
        mask = ~torch.eye(n_prompts, dtype=torch.bool)
        sim_values = sim_matrix[mask]
        mean_sim = sim_values.mean().item()
        std_sim = sim_values.std().item()

        avg_inter_prompt_sim.append(mean_sim)
        std_inter_prompt_sim.append(std_sim)

        # metric 2: embedding drift (for same prompt)
        if i < n_layers - 1:
            next_layer_data = all_states[i + 1]
            drifts = F.cosine_similarity(layer_data, next_layer_data, dim=1)
            avg_drift.append(drifts.mean().item())
            std_drift.append(drifts.std().item())

        # metric 3: average norm of embeddings (signal magnitude)
        norms = torch.norm(layer_data, p=2, dim=1)
        avg_norms.append(norms.mean().item())
        std_norms.append(norms.std().item())

    plt.figure(figsize=(18, 6))

    layers_range = np.arange(n_layers)
    drift_range = np.arange(n_layers - 1)

    # plot 1: inter-prompt similarity
    plt.subplot(1, 3, 1)
    lower_sim = np.array(avg_inter_prompt_sim) - 2 * np.array(std_inter_prompt_sim)
    upper_sim = np.array(avg_inter_prompt_sim) + 2 * np.array(std_inter_prompt_sim)
    plt.plot(
        layers_range, avg_inter_prompt_sim, marker="o", color="purple", label="Mean"
    )
    plt.fill_between(
        layers_range,
        lower_sim,
        upper_sim,
        color="purple",
        alpha=0.2,
        label="2-sigma CI",
    )

    plt.title("Inter-Prompt Similarity")
    plt.xlabel("Layer Index")
    plt.ylabel("Avg Cosine Sim (Across Prompts)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 1.05)
    plt.legend()

    # plot 2: layer drift
    plt.subplot(1, 3, 2)
    plt.plot(drift_range, avg_drift, marker="s", color="teal", label="Mean")
    lower_drift = np.array(avg_drift) - 2 * np.array(std_drift)
    upper_drift = np.array(avg_drift) + 2 * np.array(std_drift)
    plt.fill_between(
        drift_range,
        lower_drift,
        upper_drift,
        color="teal",
        alpha=0.2,
        label="2-sigma CI",
    )
    plt.title("Layer Drift")
    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Sim (Layer L vs Layer L+1)")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    plt.legend()

    # plot 3: L2 norm analysis
    plt.subplot(1, 3, 3)
    plt.plot(layers_range, avg_norms, marker="^", color="orange", label="Mean")
    lower_norm = np.array(avg_norms) - 2 * np.array(std_norms)
    upper_norm = np.array(avg_norms) + 2 * np.array(std_norms)
    plt.fill_between(
        layers_range,
        lower_norm,
        upper_norm,
        color="orange",
        alpha=0.2,
        label="2-sigma CI",
    )
    plt.title("Signal Magnitude (L2 Norm)")
    plt.xlabel("Layer Index")
    plt.ylabel("Avg Norm")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plot_dir = os.path.join(PLOTS_DIR, "hidden_state_analysis/")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "result.png"))
    plt.show()


if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

    prompts = [question_to_prompt(q) for q in utils.ISAAC_NEWTON_QUESTIONS]
    analyse_embedding_variance(MODEL_NAME, prompts)
