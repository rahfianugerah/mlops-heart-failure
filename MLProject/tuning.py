import optuna
import numpy as np
import pandas as pd
import os, json, argparse
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

def load_processed_data():
    repo_root = Path(__file__).resolve().parent.parent
    base_dir  = repo_root / "preprocessing" / "preprocess_dataset"
    train_p   = base_dir / "train_preprocessed.csv"
    test_p    = base_dir / "test_preprocessed.csv"

    if not train_p.exists() or not test_p.exists():
        raise FileNotFoundError(f"[ERROR] train/test_preprocessed.csv Not found in {base_dir}")

    train_df = pd.read_csv(train_p)
    test_df  = pd.read_csv(test_p)
    target_col = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_col]).values.astype("float32")
    y_train = train_df[target_col].values
    X_test  = test_df.drop(columns=[target_col]).values.astype("float32")
    y_test  = test_df[target_col].values
    n_classes = len(np.unique(y_train))
    print(f"[DATA] Loaded from {base_dir}")
    return X_train, X_test, y_train, y_test, n_classes

def _clear_mlflow_env():
    for k in [
        "MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID",
        "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_S3_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"
    ]:
        os.environ.pop(k, None)

def plot_confusion(cm, title, path):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar()
    ticks = np.arange(cm.shape[0]); plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_roc_and_auc(y_true, y_score, n_classes, path_png, title="ROC [VAL]"):
    fig = plt.figure()
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC={roc_val:.3f}")
    else:
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        vals = []
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            vals.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, label=f"class {c} AUC={vals[-1]:.3f}")
        roc_val = float(np.mean(vals))
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title); plt.legend(); plt.tight_layout()
    fig.savefig(path_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    return roc_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="mlp-tuning-models")
    parser.add_argument("--tracking_dir", type=str, default="mlruns")
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--study_db", type=str, default="optuna-study-mlp.db")
    args = parser.parse_args()

    repo_root     = Path(__file__).resolve().parent.parent
    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    _clear_mlflow_env()
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{args.tracking_dir}"

    mlflow.set_experiment(args.experiment_name)
    mlflow.sklearn.autolog(log_models=False)

    X_train, X_test, y_train, y_test, n_classes = load_processed_data()
    avg = "binary" if n_classes == 2 else "macro"
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "units1":        trial.suggest_categorical("units1", [32, 64, 128, 256]),
            "units2":        trial.suggest_categorical("units2", [16, 32, 64, 128]),
            "activation":    trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha":         trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "batch_size":    trial.suggest_categorical("batch_size", [32, 64, 128]),
            "epochs":        trial.suggest_int("epochs", 100, 500, step=50),
        }

        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
            mlflow.log_params(params)

            est = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(int(params["units1"]), int(params["units2"])),
                    activation=params["activation"],
                    alpha=float(params["alpha"]),
                    learning_rate_init=float(params["learning_rate"]),
                    batch_size=int(params["batch_size"]),
                    max_iter=int(params["epochs"]),
                    early_stopping=True, n_iter_no_change=10,
                    random_state=42
                ))
            ])

            est.fit(X_tr, y_tr)
            y_pred  = est.predict(X_val)
            y_proba = est.predict_proba(X_val)

            f1  = f1_score(y_val, y_pred, average=avg, zero_division=0)
            pre = precision_score(y_val, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_val, y_pred, average=avg, zero_division=0)
            val_acc = float((y_pred == y_val).mean())

            if n_classes == 2:
                auc_val = roc_auc_score(y_val, y_proba[:, 1])
            else:
                auc_val = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")

            mlflow.log_metric("f1_valid", float(f1))
            mlflow.log_metric("precision_valid", float(pre))
            mlflow.log_metric("recall_valid", float(rec))
            mlflow.log_metric("val_accuracy", float(val_acc))
            mlflow.log_metric("roc_auc_valid", float(auc_val))

            return f1

    with mlflow.start_run(run_name="optuna-study"):
        storage = f"sqlite:///{args.study_db}"
        study = optuna.create_study(
            study_name="mlp-tuning-f1",
            direction="maximize",
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1, gc_after_trial=True)

    best = study.best_params.copy()
    best.setdefault("units1", 64); best.setdefault("units2", 32)
    best.setdefault("activation", "relu"); best.setdefault("alpha", 1e-4)
    best.setdefault("learning_rate", 1e-3); best.setdefault("batch_size", 64)
    best.setdefault("epochs", 200)

    out_root      = repo_root / "best_params.json"
    out_mlproject = Path(__file__).resolve().parent / "best_params.json"
    json.dump(best, open(out_root, "w"), indent=2)
    json.dump(best, open(out_mlproject, "w"), indent=2)
    print("[INFO] Saved Best Params:", best)
    print("[INFO] Best F1:", study.best_value)

    final_est = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(int(best["units1"]), int(best["units2"])),
            activation=best["activation"],
            alpha=float(best["alpha"]),
            learning_rate_init=float(best["learning_rate"]),
            batch_size=int(best["batch_size"]),
            max_iter=int(best["epochs"]),
            early_stopping=True, n_iter_no_change=10,
            random_state=42
        ))
    ])

    final_est.fit(X_tr, y_tr)
    y_pred  = final_est.predict(X_val)
    y_proba = final_est.predict_proba(X_val)

    cm = confusion_matrix(y_val, y_pred)
    plot_confusion(cm, "Confusion Matrix [VAL]", artifacts_dir / "confusion_matrix_val.png")
    np.savetxt(artifacts_dir / "confusion_matrix_val.csv", cm, delimiter=",", fmt="%d")

    auc_val = plot_roc_and_auc(y_val, y_proba, n_classes, artifacts_dir / "roc_curve_val.png", title="ROC [VAL]")
    with open(artifacts_dir / "classification_report_val.txt", "w") as f:
        f.write(classification_report(y_val, y_pred, zero_division=0))
    with open(artifacts_dir / "metrics_val.json", "w") as f:
        json.dump({
            "f1_valid": f1_score(y_val, y_pred, average=("binary" if n_classes == 2 else "macro"), zero_division=0),
            "precision_valid": precision_score(y_val, y_pred, average=("binary" if n_classes == 2 else "macro"), zero_division=0),
            "recall_valid": recall_score(y_val, y_pred, average=("binary" if n_classes == 2 else "macro"), zero_division=0),
            "val_accuracy": float((y_pred == y_val).mean()),
            "roc_auc_valid": float(auc_val)
        }, f, indent=2)

    print(f"[ARTIFACTS] Validation Artifacts Saved: {artifacts_dir}")

if __name__ == "__main__":
    main()