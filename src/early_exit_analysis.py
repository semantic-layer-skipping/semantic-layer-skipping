import logging
import os
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer

from utils import get_device, PLOTS_DIR
from store import SkippingVectorDB
from strategies import SkipDecision, Action, EarlyExitStrategy, StrictMatchStrategy


class EarlyExitAnalyser:
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

    def _compute_early_exit_logits(self, resid_state: torch.Tensor) -> torch.Tensor:
        """
        Simulates an 'Early Exit' from the residual stream.
        1. Takes the residual state (Batch, Seq, Hidden)
        2. Applies the Model's Final Normalisation (RMSNorm/LayerNorm)
        3. Projects using the Unembedding Head (Vocab)
        """
        # `ln_final` stores the final layer norm
        normalised_state = self.model.ln_final(resid_state)
        logits = self.model.unembed(normalised_state)
        return logits

    def analyse_prompt(self, prompt: str,
                       vector_db: Optional[SkippingVectorDB] = None,
                       early_exit_strategy: Optional[EarlyExitStrategy] = None
                       ) -> Dict[str, Any]:
        """
        Runs a single prompt and calculates exit metrics for EVERY layer,
        specifically for the LAST token (next-token prediction).
        """
        # 1. run model and cache all intermediate states
        # final_logits shape: (batch=1, seq_len, vocab_size)
        final_logits, cache = self.model.run_with_cache(prompt, return_type="logits")

        # get the final token's logits, probs, and predicted token id
        target_final_logits = final_logits[0, -1, :] # shape: (vocab_size,)
        target_probs = F.softmax(target_final_logits, dim=-1)
        target_token_id = torch.argmax(target_final_logits).item()

        n_layers = self.model.cfg.n_layers
        results = {
            "kl_divergence": [],
            "strict_match": [],
            "top_token_id": [],
            "layer_indices": list(range(n_layers)),
            "prompt": prompt,
            "final_predicted_token": self.model.to_string(target_token_id),
            "db_actions": []
        }

        # 2. iterate through every layer's output
        for i in range(n_layers):
            # hook name for the residual stream after layer i
            hook_name = f"blocks.{i}.hook_resid_post"

            # extract residual stream at the last token
            # shape: (batch, seq, hidden) -> (hidden)
            resid_pre = cache[hook_name][0, -1, :]

            # 3. simulate early exit
            early_logits = self._compute_early_exit_logits(resid_pre)
            early_token_id = torch.argmax(early_logits).item()

            # 4. metric: Strict Match (did we get the same token?)
            is_match = (early_token_id == target_token_id)

            # 5. metric: KL Divergence
            kl = F.kl_div(
                F.log_softmax(early_logits, dim=-1), # convert logits to log_probs
                target_probs,
                reduction='sum'
            ).item()

            # store data
            results["kl_divergence"].append(kl)
            results["strict_match"].append(is_match)
            results["top_token_id"].append(early_token_id)

            # 6. vector DB population logic
            if vector_db and early_exit_strategy:
                should_exit = early_exit_strategy.should_exit(early_logits, target_final_logits)
                if should_exit:
                    logging.info(f"    [DB] Layer {i}: Early exit condition met by strategy.")

                    # extract vector for DB storage (convert to numpy)
                    vector_np = resid_pre.detach().cpu().numpy().reshape(1, -1)  # shape: (1, hidden)

                    decision = SkipDecision(action=Action.EXIT)
                    results["db_actions"].append(decision)

                    vector_db.add_vector(layer_idx=i, vector=vector_np, decision=decision)

                else:
                    # we could also store CONTINUE decisions if desired
                    results["db_actions"].append(Action.CONTINUE)

            # debug print the top 5 predictions at every 4th layer + last layer
            if i % 4 == 0 or i == n_layers - 1:
                top_vals, top_indices = torch.topk(early_logits, k=5)
                current_top_strings = [self.model.to_string(idx) for idx in top_indices]
                logging.info(f"    Layer {i} (KL: {kl:.4f}) Top Predictions: ")
                logging.info(f"        {current_top_strings}")

        return results


def apply_chat_template(prompt: str, model_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def plot_layer_divergence(analysis_results: List[Dict[str, Any]], kl_cap: float = 10.0):
    """
    Plots KL Divergence and Accuracy logic across layers for multiple prompts.
    """
    plt.figure(figsize=(12, 6))

    # 1. plot KL divergence
    plt.subplot(1, 2, 1)
    for res in analysis_results:
        layers = res["layer_indices"]
        kl = res["kl_divergence"]
        # cut off extremely high KL values for readability (first few layers are random)
        kl = [min(k, kl_cap) for k in kl]
        label_text = f"Prompt: '{res['prompt'][:15]}...'"
        plt.plot(layers, kl, marker='o', label=label_text)

    plt.title("Soft Metric: KL Divergence vs. Depth")
    plt.xlabel("Transformer Layer")
    plt.ylabel("KL Divergence")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. plot strict matching
    plt.subplot(1, 2, 2)
    for res in analysis_results:
        layers = res["layer_indices"]
        matches = [1 if x else 0 for x in res["strict_match"]]
        plt.plot(layers, matches, marker='s', linestyle='--', alpha=0.7)

    plt.title("Strict Metric: Token Match")
    plt.xlabel("Transformer Layer")
    plt.yticks([0, 1], ["Mismatch", "Match"])
    plt.grid(True, alpha=0.3)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(f"{PLOTS_DIR}/early_exit_analysis.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # example models: Qwen3 (trust_remote_code warning): Qwen/Qwen3-0.6B.
    # Qwen/Qwen2-1.5B
    # Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B-Instruct,  Qwen/Qwen2.5-3B-Instruct
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    analyser = EarlyExitAnalyser(model_name=model_name)

    prompts = [
        #"[QUESTION]: What is the capital of France?\n[ANSWER]:",
        "[QUESTION]: Which university did Isaac Newton go to?\n[ANSWER]:",
        #"[QUESTION]: Who proposed the fundamental theory of relativity?\n[ANSWER]:",
    ]
    test_prompts = prompts


    vector_db = SkippingVectorDB(n_checkpoints=analyser.model.cfg.n_layers,
                                 vector_dim=analyser.model.cfg.d_model)
    strategy = StrictMatchStrategy()

    all_results = []
    logging.info("\nStarting Analysis...")
    for prompt in test_prompts:
        logging.info(f"Analysing: '{prompt}'")
        res = analyser.analyse_prompt(prompt, vector_db=vector_db, early_exit_strategy=strategy)
        all_results.append(res)

        try:
            first_match_layer = res["strict_match"].index(True)
            logging.info(f" -> First accurate prediction at Layer: {first_match_layer}")
            logging.info(f" -> Final Token: {res['final_predicted_token']}")
        except ValueError:
            logging.info(" -> No layer matched the final output (!!)")

    # visualise
    plot_layer_divergence(all_results)
