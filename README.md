# semantic-layer-skipping

This project is developed as part of a dissertation for Part III Computer Science at the University of Cambridge.



## Prerequisites

To set up the project, follow these steps:

1. Ensure `uv` is installed.
2. Run:
    ```bash
    uv sync
    ```
3. Enable git hooks:
    ```bash
    uv run pre-commit install
    ```


## Running

Scripts can be run using:
   ```
   uv run python -m inference.runner
   ```
Or, if the virtual environment is activated:
   ```
   python3 -m inference.runner
   ```

Initial analysis scripts can be found in the `initial_analysis` folder.

## Result inspection

We can analyse sample generated text using:
```
jq '.' results/sharegpt_test_100s_2048t_top1_strict_full_generation.json | grep "text" | less -S
```
This takes in json input, decodes it and outputs the generated text (1 line per entry, no matter how long it is).

Viewing squeue full job names
```
squeue -u yff23 -O JobID,Partition,Name:70
```
