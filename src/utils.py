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

PROMPTS = [
    "[QUESTION]: What is the capital of France?\n[ANSWER]:",
    "[QUESTION]: Which university did Isaac Newton go to?\n[ANSWER]:",
    "[QUESTION]: Who proposed the fundamental theory of relativity?\n[ANSWER]:",
]


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
