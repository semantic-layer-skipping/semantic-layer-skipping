import logging
import os
from pathlib import Path

import torch

PLOTS_DIR = "plots"


# ISAAC_NEWTON_QUESTIONS = """
# Isaac Newton earned his Bachelor of Arts degree from which university?
# Who is Einstein?
# """.strip().split("\n")

ISAAC_NEWTON_QUESTIONS = """
At which university was Isaac Newton a student?
To which university did Isaac Newton gain admission in 1661?
Isaac Newton earned his Bachelor of Arts degree from which university?
Which university is famous for being Isaac Newton's alma mater?
Where did Isaac Newton study before becoming a fellow of Trinity College?
Isaac Newton served as the Lucasian Professor of Mathematics at which university?
Which university counts Isaac Newton among its most historically significant alumni?
At what institution did Isaac Newton complete the majority of his academic work?
Isaac Newton's academic career is most closely associated with which British university?
Which university is home to the college where Isaac Newton lived and worked?
""".strip().split("\n")

ISAAC_NEWTON_QUESTIONS_TRAIN = ISAAC_NEWTON_QUESTIONS[:3]
ISAAC_NEWTON_QUESTIONS_CALIBRATION = ISAAC_NEWTON_QUESTIONS[3:7]
ISAAC_NEWTON_QUESTIONS_TEST = ISAAC_NEWTON_QUESTIONS[7:]

PROMPTS = [
    "What is the capital of France?",
    "Which university did Isaac Newton go to?",
    "Who proposed the fundamental theory of relativity?",
]

HPC_USER = "yff23"

PERSONAL_HPC_EXPERIMENTS_PATH = "~/rds/hpc-work/semantic-layer-skipping/experiments"
HPC_EXPERIMENTS_PATH = (
    "~/rds/rds-cl-acs-yff23-cjlENNKY3so/semantic-layer-skipping/experiments/"
)


def question_to_prompt(question: str) -> str:
    """Converts a question into a standard prompt format."""
    return f"[QUESTION]: {question}\n[ANSWER]: "


def get_device():
    """Utility function to get the available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_experiment_output_dir(loc: str = None):
    current_path = Path.cwd()

    if loc == "repo":
        return "experiments"
    elif loc == "hpc-work":
        return os.path.join(os.path.expanduser(PERSONAL_HPC_EXPERIMENTS_PATH))
    elif loc == "rds-cl":
        return os.path.join(os.path.expanduser(HPC_EXPERIMENTS_PATH))

    if get_device().type == "cuda" or HPC_USER in current_path.parts:
        # we are likely running on HPC, so save to special directory
        return os.path.join(os.path.expanduser(PERSONAL_HPC_EXPERIMENTS_PATH))
    else:
        # for local runs, save to a local directory
        return "experiments"


def set_logging_config():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def compute_truncated_kl_divergence(
    true_logits: torch.Tensor, sim_logits: torch.Tensor, top_k: int = 50
) -> torch.Tensor:
    """
    Computes a truncated, mass-aware forward KL Divergence.
    Includes FP16 underflow protections and automatic checkpoint broadcasting.
    """
    # calculate full distributions
    true_probs_full = torch.nn.functional.softmax(true_logits, dim=-1)
    sim_log_probs_full = torch.nn.functional.log_softmax(sim_logits, dim=-1)

    # isolate the top-K absolute probabilities and indices
    true_top_k_probs, top_k_indices = torch.topk(true_probs_full, k=top_k, dim=-1)

    # handle broadcasting if sim_logits has an extra dimension
    # (e.g., [checkpoints, batch, vocab])
    if sim_logits.dim() > true_logits.dim():
        num_checkpoints = sim_logits.shape[0]
        # dynamically expand to match the extra checkpoint dimension
        expanded_indices = top_k_indices.unsqueeze(0).expand(
            num_checkpoints, *([-1] * top_k_indices.dim())
        )
        true_top_k_probs = true_top_k_probs.unsqueeze(0)
    else:
        expanded_indices = top_k_indices

    # gather the absolute log-probabilities for the identical tokens
    sim_top_k_log_probs = torch.gather(
        sim_log_probs_full, dim=-1, index=expanded_indices
    )

    # upcast to float32 to prevent 16-bit underflow during the log calculation
    true_top_k_probs_fp32 = true_top_k_probs.to(torch.float32)
    sim_top_k_log_probs_fp32 = sim_top_k_log_probs.to(torch.float32)

    # clamp to avoid NaN from log(0)
    safe_true_probs = true_top_k_probs_fp32.clamp(min=1e-10)

    # 7. calculate truncated forward KL divergence
    kl_divs = (
        true_top_k_probs_fp32 * (safe_true_probs.log() - sim_top_k_log_probs_fp32)
    ).sum(dim=-1)

    return kl_divs
