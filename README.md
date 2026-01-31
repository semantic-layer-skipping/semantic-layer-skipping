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
