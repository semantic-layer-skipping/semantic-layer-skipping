import json
import logging
import os
from dataclasses import asdict
from typing import Any

from experiment.config import CalibrationConfig, PopulationConfig, TestConfig
from store import SkippingVectorDB


class ExperimentManager:
    def __init__(self, population_config: PopulationConfig):
        self.population_config = population_config
        os.makedirs(self.population_config.base_path, exist_ok=True)

    def initialise_db(self, force_new: bool = False) -> SkippingVectorDB:
        """Loads existing DB or creates new one."""
        if force_new or not self.db_exists():
            return self._create_new_db()

        try:
            self._validate_population_config()
            return SkippingVectorDB.load(
                self.population_config.db_path,
                len(self.population_config.checkpoints),
                self.population_config.vector_dim,
            )
        except ValueError as e:
            logging.warning(f"DB Mismatch ({e}). Creating fresh.")
            return self._create_new_db()

    def save_population_state(self, db: SkippingVectorDB):
        """Saves DB and the Population Config."""
        # save config
        with open(
            f"{self.population_config.base_path}/population_config.json", "w"
        ) as f:
            json.dump(asdict(self.population_config), f, indent=4)

        # save DB
        db.save(self.population_config.db_path)
        logging.info(f"Population state saved to {self.population_config.base_path}")

    def db_exists(self):
        return os.path.exists(self.population_config.db_path)

    def _create_new_db(self):
        return SkippingVectorDB(
            len(self.population_config.checkpoints), self.population_config.vector_dim
        )

    def _validate_population_config(self):
        path = f"{self.population_config.base_path}/population_config.json"
        if not os.path.exists(path):
            raise ValueError("Missing config file")

        with open(path) as f:
            saved = json.load(f)

        # Check critical fields
        if saved["model_name"] != self.population_config.model_name:
            raise ValueError("Model Name mismatch")
        if saved["checkpoints"] != self.population_config.checkpoints:
            raise ValueError("Checkpoints mismatch")

    # CALIBRATION

    def get_calibration_path(self, run_name: str) -> str:
        """Helper to get the path for a specific calibration run."""

        return os.path.join(self.population_config.base_path, "calibration", run_name)

    def save_calibration_state(
        self, calibration_config: CalibrationConfig, thresholds: dict[int, float]
    ):
        """Saves thresholds and calibration config into a subfolder."""
        calibration_dir = (
            f"{self.population_config.base_path}/calibration/"
            f"{calibration_config.run_name}"
        )
        results_dir = os.path.join(calibration_dir, "results")

        os.makedirs(calibration_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # save config
        with open(f"{calibration_dir}/config.json", "w") as f:
            json.dump(asdict(calibration_config), f, indent=4)

        # save thresholds
        with open(f"{calibration_dir}/thresholds.json", "w") as f:
            # convert int keys to string for JSON
            json.dump({str(k): v for k, v in thresholds.items()}, f, indent=4)

        logging.info(f"Calibration results saved to {calibration_dir}")

    def load_thresholds(self, run_name: str) -> dict[int, float]:
        """Loads thresholds from a specific calibration run."""
        path = (
            f"{self.population_config.base_path}/calibration/{run_name}/thresholds.json"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration run '{run_name}' not found.")

        with open(path) as f:
            data = json.load(f)
            return {int(k): v for k, v in data.items()}

    # EVALUATION RESULTS

    def save_test_results(self, test_config: TestConfig, metrics: dict[str, Any]):
        """
        Saves test results INSIDE the parent calibration folder.
        """
        # get parent calibration folder
        cal_dir = self.get_calibration_path(test_config.calibration_run)
        results_dir = os.path.join(cal_dir, "results")

        if not os.path.exists(results_dir):
            logging.warning(
                f"Results directory not found at {results_dir}. Creating it."
            )
            os.makedirs(results_dir, exist_ok=True)

        # construct output
        output_data = {
            "meta": {
                "experiment": self.population_config.experiment_name,
                "calibration_run": test_config.calibration_run,
                "test_run": test_config.run_name,
            },
            "config": asdict(test_config),
            "metrics": metrics,
        }

        # save
        filename = f"{test_config.run_name}.json"
        save_path = os.path.join(results_dir, filename)

        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=4)

        logging.info(f"Test results saved to {save_path}")
