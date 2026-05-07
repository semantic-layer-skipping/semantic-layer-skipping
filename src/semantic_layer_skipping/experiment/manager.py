import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any

from experiment.config import CalibrationConfig, EvalConfig, PopulationConfig
from store import SkippingVectorDB


class ExperimentManager:
    def __init__(self, population_config: PopulationConfig):
        self.population_config = population_config
        os.makedirs(self.population_config.base_path, exist_ok=True)

    def initialise_db(
        self, force_new: bool = False, ensure_exists=False
    ) -> SkippingVectorDB:
        """Loads existing DB or creates new one."""
        if ensure_exists and not self.db_exists():
            raise FileNotFoundError(
                f"DB not found at {self.population_config.db_path}. "
            )

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

        # check critical fields
        if saved["model_name"] != self.population_config.model_name:
            raise ValueError("Model Name mismatch")
        if saved["checkpoints"] != self.population_config.checkpoints:
            raise ValueError("Checkpoints mismatch")

    # MERGED DBs
    def get_merged_db_path(self, keep_fraction: float) -> str:
        """
        Generates a standard path for a merged DB based on its subsampling fraction.
        """
        folder_name = f"db_merged_subsampled_{int(keep_fraction * 100)}pct"
        return os.path.join(self.population_config.base_path, folder_name)

    def merged_db_exists(self, keep_fraction: float) -> bool:
        path = self.get_merged_db_path(keep_fraction)
        return os.path.exists(path)

    def load_merged_db(self, keep_fraction: float) -> SkippingVectorDB:
        """Loads a specific merged database."""
        path = self.get_merged_db_path(keep_fraction)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Merged DB not found at {path}")

        return SkippingVectorDB.load(
            path,
            len(self.population_config.checkpoints),
            self.population_config.vector_dim,
        )

    def get_ivfpq_db_path(self, keep_fraction: float) -> str:
        folder_name = f"db_ivfpq_subsampled_{int(keep_fraction * 100)}pct"
        return os.path.join(self.population_config.base_path, folder_name)

    def ivfpq_db_exists(self, keep_fraction: float) -> bool:
        path = self.get_ivfpq_db_path(keep_fraction)
        return os.path.exists(path)

    def load_ivfpq_db(self, keep_fraction: float) -> SkippingVectorDB:
        path = self.get_ivfpq_db_path(keep_fraction)
        if not os.path.exists(path):
            raise FileNotFoundError(f"IVFPQ DB not found at {path}")

        return SkippingVectorDB.load(
            path,
            len(self.population_config.checkpoints),
            self.population_config.vector_dim,
        )

    # CALIBRATION

    def get_raw_calibration_path(self, data_run_name: str) -> str:
        """Helper to get the path for the raw simulated data."""
        return os.path.join(
            self.population_config.base_path, "calibration", data_run_name
        )

    def get_calibration_path(self, run_name: str) -> str:
        """Helper to get the path for a specific calibration run."""
        return os.path.join(self.population_config.base_path, "calibration", run_name)

    def raw_calibration_exists(self, data_run_name: str) -> bool:
        """
        Checks if the heavy simulation loop has already been
        run for these parameters.
        """
        path = self.get_raw_calibration_path(data_run_name)
        return os.path.exists(os.path.join(path, "raw_results.json"))

    def calibration_exists(self, run_name: str) -> bool:
        """Checks if a valid calibration result exists."""
        path = self.get_calibration_path(run_name)
        return os.path.exists(os.path.join(path, "thresholds.json"))

    def save_raw_calibration_results(self, cal_cfg: CalibrationConfig, calibrator):
        """Saves the raw simulation results before precision thresholds are applied."""
        data_dir = self.get_raw_calibration_path(cal_cfg.data_run_name)
        os.makedirs(data_dir, exist_ok=True)

        results_file = os.path.join(data_dir, "raw_results.json")
        logging.info(f"Saving raw simulation results to {results_file}")

        with open(results_file, "w") as f:
            json.dump(calibrator.get_serialised_results(), f, indent=4)

        # save the raw data config to document how it was generated
        with open(os.path.join(data_dir, "data_config.json"), "w") as f:
            json.dump(asdict(cal_cfg), f, indent=4)

    def load_raw_calibration_results(self, cal_cfg: CalibrationConfig, calibrator):
        """loads raw simulation results into the calibrator to skip gpu execution."""
        results_file = os.path.join(
            self.get_raw_calibration_path(cal_cfg.data_run_name), "raw_results.json"
        )
        logging.info(f"Loading cached simulation results from {results_file}")

        with open(results_file) as f:
            data = json.load(f)
            calibrator.load_serialised_results(data)

    def save_calibration_state(
        self,
        calibration_config: CalibrationConfig,
        thresholds: dict[int, float],
        calibrator=None,
    ):
        """Saves thresholds and calibration config into a subfolder."""
        calibration_dir = self.get_calibration_path(calibration_config.run_name)
        results_dir = os.path.join(calibration_dir, "results")

        os.makedirs(calibration_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        if calibrator is not None:
            # save calibrator results
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            calibration_results_file = (
                f"{calibration_dir}/calibrator_results_{current_time}.json"
            )
            serialisable_results = calibrator.get_serialised_results()
            logging.info(f"Saving full results to {calibration_results_file}")
            with open(calibration_results_file, "w") as f:
                json.dump(serialisable_results, f, indent=4)

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

    def get_test_results_path(
        self, test_config: EvalConfig, db_path: str | None
    ) -> str:
        """Determines the file path where the test results JSON will be saved."""

        if test_config.random_skip_prob is not None:
            results_dir = os.path.join(
                self.population_config.base_path, "baselines", "random_skip"
            )
        elif test_config.calibration_run == "manual_thresholds":
            # dedicated folder for manual experiments
            db_name = (
                os.path.basename(os.path.normpath(db_path)) if db_path else "default_db"
            )
            results_dir = os.path.join(
                self.population_config.base_path, f"manual_eval_results_{db_name}"
            )
        else:
            # standard folder inside the calibration run
            cal_dir = self.get_calibration_path(test_config.calibration_run)
            results_dir = os.path.join(cal_dir, "results")

        filename = f"{test_config.run_name}.json"
        return os.path.join(results_dir, filename)

    def test_results_exist(self, test_config: EvalConfig, db_path: str | None) -> bool:
        """Checks if the evaluation has already been run and saved."""
        return os.path.exists(self.get_test_results_path(test_config, db_path))

    def save_test_results(
        self, test_config: EvalConfig, metrics: dict[str, Any], db_path: str | None
    ):
        """
        Saves test results. If manual thresholds are used, saves them to a
        dedicated 'manual_eval' folder instead of a calibration run folder.
        """
        save_path = self.get_test_results_path(test_config, db_path)

        # ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        config_dict = asdict(test_config)
        if config_dict.get("thresholds"):
            config_dict["thresholds"] = {
                str(k): v for k, v in config_dict["thresholds"].items()
            }

        # construct output
        output_data = {
            "meta": {
                "experiment": self.population_config.experiment_name,
                "calibration_run": test_config.calibration_run,
                "test_run": test_config.run_name,
                # record db_path only if it wasn't a random_skip_prob baseline
                "db_path": db_path if test_config.random_skip_prob is None else None,
            },
            "config": config_dict,
            "metrics": metrics,
        }

        # save
        with open(save_path, "w") as f:
            json.dump(output_data, f, indent=4)

        logging.info(f"Test results saved to {save_path}")
