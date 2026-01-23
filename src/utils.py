import torch


PLOTS_DIR = "plots"


PROMPTS = [
    "[QUESTION]: What is the capital of France?\n[ANSWER]:",
    "[QUESTION]: Which university did Isaac Newton go to?\n[ANSWER]:",
    "[QUESTION]: Who proposed the fundamental theory of relativity?\n[ANSWER]:",
]

def get_device():
    """Utility function to get the available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
