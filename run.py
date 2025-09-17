import os
import sys
import subprocess

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MLPROJECT_DIR = PROJECT_ROOT / "MLProject"

def run(cmd: list[str]):
    print(f"\n[CMD] {' '.join(cmd)}\n", flush=True)
    res = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if res.returncode != 0:
        sys.exit(res.returncode)

def main():
    (PROJECT_ROOT / "artifacts").mkdir(exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)

    os.environ.pop("MLFLOW_RUN_ID", None)
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    os.environ["MLFLOW_TRACKING_URI"] = "file:mlruns"

    # TUNE (Optuna)
    run([
        sys.executable, str(MLPROJECT_DIR / "modelling_tuning.py"),
        "--experiment_name", "mlp_tuning_local",
        "--tracking_dir", "mlruns",
        "--n_trials", os.getenv("OPTUNA_N_TRIALS", "30"),
        "--study_db", "optuna_study_mlp.db",
    ])

    # TRAIN FINAL (best_params.json)
    run([
        sys.executable, str(MLPROJECT_DIR / "modelling.py"),
        "--experiment_name", "mlp_local",
        "--tracking_dir", "mlruns",
    ])

    print("\n[OK] Tuning & Training. Artifacts > ./artifacts, model > ./models\n")

if __name__ == "__main__":
    main()