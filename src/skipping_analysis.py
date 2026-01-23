import torch
import torch.nn.functional as F
import logging
import sys
from typing import Optional, Dict, List, Any
from transformer_lens import HookedTransformer
from utils import get_device, PLOTS_DIR
import matplotlib.pyplot as plt
import numpy as np
import os


class SkipAnalyser:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: Optional[str] = None):
        """
        Initialises the model and puts it in evaluation mode.
        """
        self.device = get_device() if device is None else device
        logging.info(f"Loading model '{model_name}' on device: {self.device}...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
            # no gradients for inference analysis
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        self.model.eval()
        logging.info("Model loaded successfully.")

    def analyse_skips(self,
                      prompt: str,
                      similarity_threshold: float = 0.9,
                      max_skip_size: int = 8) -> List[Dict[str, Any]]:
        """
        Analyses a single prompt to find valid layer skips using both:
        1. Cosine Similarity (Heuristic/Soft)
        2. Strict Match Simulation (Strict)
        """
        # get original logits and full cache
        original_logits, full_cache = self.model.run_with_cache(prompt, return_type="logits")

        # focus on last token (next token prediction)
        target_final_logits = original_logits[0, -1, :]
        target_token_id = torch.argmax(target_final_logits).item()
        target_token_str = self.model.to_string(target_token_id)
        logging.info(f"Baseline Target Token: '{target_token_str}' (ID: {target_token_id})")

        # prepare input tokens for re-runs
        tokens = self.model.to_tokens(prompt)

        n_layers = self.model.cfg.n_layers
        all_results = []

        # iterate through possible start layers (0 to n_layers - 2)
        for start_layer in range(n_layers - 1):
            # extract the residual state at the output of start_layer
            start_state = full_cache[f"blocks.{start_layer}.hook_resid_post"][0, -1, :] # shape: (hidden_size,)

            # iterate through possible skip sizes
            for n in range(1, max_skip_size + 1):
                dest_layer = start_layer + n + 1
                if dest_layer >= n_layers:
                    break

                # --- Strategy A: Cosine Similarity Heuristic ---
                # compare distance of Output(Start) vs Output(Start + N)
                last_skipped_layer_idx = dest_layer - 1
                end_state_hook = f"blocks.{last_skipped_layer_idx}.hook_resid_post"
                end_state = full_cache[end_state_hook][0, -1, :]

                cos_similarity = F.cosine_similarity(start_state, end_state, dim=0).item()
                passes_similarity = cos_similarity >= similarity_threshold

                # --- Strategy B: Strict Match Simulation ---
                # we inject start_state directly into 'dest_layer', bypassing the middle.
                def injection_hook(resid_pre, hook):
                    # resid_pre shape: (Batch, Seq, Hidden)
                    # we only overwrite the last token's state
                    resid_pre[:, -1, :] = start_state
                    return resid_pre

                # run inference with the hook applied only to dest_layer
                with torch.no_grad():
                    hook_point = f"blocks.{dest_layer}.hook_resid_pre"
                    # note: we re-run the entire model with the hook
                    # TODO: slice model and only run from dest_layer onwards for efficiency
                    sim_logits = self.model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(hook_point, injection_hook)]
                    )
                sim_token_id = torch.argmax(sim_logits[0, -1, :]).item()
                passes_strict = (sim_token_id == target_token_id)

                # Store result
                res = {
                    "start_layer": start_layer,
                    "skip_count": n,
                    "land_layer": dest_layer,
                    "similarity": cos_similarity,
                    "strict_match": passes_strict,
                    "passes_similarity_threshold": passes_similarity
                }
                all_results.append(res)

                # Logging (Keep it concise)
                if passes_strict:
                    logging.info(f"✅ FOUND: L{start_layer}->Skip{n} (Sim: {cos_similarity:.4f})")
                elif passes_similarity:
                    logging.info(f"⚠️ High Sim ({cos_similarity:.4f}) but Strict Failed L{start_layer}->Skip{n}")
                else:
                    # did not pass either check
                    logging.info(f"❌ No Skip Valid at L{start_layer}->Skip {n} (Sim: {cos_similarity:.4f})")

        return all_results


def plot_skip_heatmap(results: List[Dict[str, Any]], prompt: str, max_skip_size: int, n_layers: int):
    """
    Visualises the skip analysis as a heatmap.
    X-Axis: Start Layer
    Y-Axis: Skip Count
    Color: Cosine Similarity
    Markers: Strict Match Success/Failure
    """
    if not results:
        print("No results to plot.")
        return

    # rows = skip size (1 to max_skip), cols = start layer (0 to n_layers-1)
    similarity_grid = np.zeros((max_skip_size, n_layers))
    strict_grid = np.zeros((max_skip_size, n_layers))  # 1 for Pass, 0 for Fail

    # mask for invalid combinations (e.g. going out of bounds)
    mask = np.ones((max_skip_size, n_layers))
    for res in results:
        r = res['skip_count'] - 1  # 0-indexed for array
        c = res['start_layer']
        # fill in grids
        similarity_grid[r, c] = res['similarity']
        strict_grid[r, c] = 1 if res['strict_match'] else 0
        mask[r, c] = 0  # mark as valid

    fig, ax = plt.subplots(figsize=(14, 6))
    # heatmap of cosine similarity, with invalid areas marked as white
    masked_similarity = np.ma.masked_where(mask == 1, similarity_grid)
    cax = ax.imshow(masked_similarity, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0.4, vmax=1.0)
    cbar = fig.colorbar(cax)
    cbar.set_label('Cosine Similarity (Residual Stream)')

    # add markers for strict match results
    for res in results:
        r = res['skip_count'] - 1
        c = res['start_layer']
        if res['strict_match']:
            # star for success
            ax.text(c, r, '★', ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        elif res['passes_similarity_threshold'] and not res['strict_match']:
            # red X for "False Positive" (High Sim, Bad Result)
            ax.text(c, r, 'x', ha='center', va='center', color='red', fontsize=10, fontweight='bold')

    ax.set_title(f"Skipping Analysis: '{prompt[:50]}...'", fontsize=14)
    ax.set_xlabel("Start Layer (Where we skip FROM)")
    ax.set_ylabel("Skip Count (N layers skipped)")
    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(max_skip_size))
    ax.set_yticklabels(range(1, max_skip_size + 1))

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/skip_analysis_{prompt[12:20].replace(' ', '_')}.png")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    analyser = SkipAnalyser(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    prompts = [
        "[QUESTION]: What is the capital of France?\n[ANSWER]:",
        "[QUESTION]: Which university did Isaac Newton go to?\n[ANSWER]:",
        "[QUESTION]: Who proposed the fundamental theory of relativity?\n[ANSWER]:",
    ]

    max_skips = 10
    for prompt in prompts:
        logging.info(f"\nAnalysing Prompt: '{prompt}'")
        results = analyser.analyse_skips(prompt, similarity_threshold=0.95, max_skip_size=max_skips)
        plot_skip_heatmap(results, prompt, max_skip_size=max_skips, n_layers=analyser.model.cfg.n_layers)

        # summary of best skips
        print(f"\n--- Best Skips for '{prompt[:20]}...' ---")
        successful_skips = [r for r in results if r['strict_match']]
        if successful_skips:
            # sort by skip count (descending) to show biggest wins first
            successful_skips.sort(key=lambda x: x['skip_count'], reverse=True)
            for res in successful_skips[:10]:
                print(
                    f"Start L{res['start_layer']} -> Skip {res['skip_count']} -> Land L{res['land_layer']} (Sim: {res['similarity']:.4f})")
        else:
            print("No strict skips found.")
        print("=" * 50)
