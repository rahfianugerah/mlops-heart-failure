import joblib
import numpy as np
import pandas as pd
import os, json, argparse, shutil
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
)

def load_processed_data():
    repo_root = Path(__file__).resolve().parent.parent
    base_dir = repo_root / "preprocessing" / "preprocess_dataset"

    train_p = base_dir / "train_preprocessed.csv"
    test_p  = base_dir / "test_preprocessed.csv"

    if not train_p.exists() or not test_p.exists():
        raise FileNotFoundError(f"train/test_preprocessed.csv not found in {base_dir}")

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

def plot_confusion(cm, title, path):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar()
    ticks = np.arange(cm.shape[0]); plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_roc_and_auc(y_true, y_score, n_classes, path_png):
    fig = plt.figure()
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        plt.plot([0,1],[0,1],"--", color="gray")
    else:
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        roc_aucs = []
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_aucs.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, label=f"class {c} (AUC={roc_aucs[-1]:.3f})")
        roc_auc = float(np.mean(roc_aucs))
        plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    fig.savefig(path_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    return roc_auc

def _clear_mlflow_env():
    for k in ["MLFLOW_RUN_ID", "MLFLOW_EXPERIMENT_ID"]:
        os.environ.pop(k, None)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="mlp_local")
    parser.add_argument("--tracking_dir", type=str, default="mlruns")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    artifacts_dir = repo_root / "artifacts"
    models_dir = repo_root / "models"
    artifacts_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    _clear_mlflow_env()
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{args.tracking_dir}"

    X_train, X_test, y_train, y_test, n_classes = load_processed_data()
    avg = "binary" if n_classes == 2 else "macro"

    h = dict(units1=64, units2=32, activation="relu", alpha=1e-4,
             learning_rate=1e-3, batch_size=64, epochs=200)
    best_params_path = repo_root / "best_params.json"
    if best_params_path.exists():
        loaded = json.load(open(best_params_path))
        alias = {"lr": "learning_rate", "max_iter": "epochs", "hidden1": "units1", "hidden2": "units2"}
        for k, v in loaded.items():
            h[alias.get(k, k)] = v
        print(f"[INFO] Using best_params.json: {h}")

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(int(h["units1"]), int(h["units2"])),
            activation=h["activation"],
            alpha=float(h["alpha"]),
            learning_rate_init=float(h["learning_rate"]),
            batch_size=int(h["batch_size"]),
            max_iter=int(h["epochs"]),
            early_stopping=True, n_iter_no_change=10,
            random_state=42
        ))
    ])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="mlp-final") as run:
        run_id = run.info.run_id
        with open(artifacts_dir / "run_id.txt", "w") as f:
            f.write(run_id)
        print(f"[RUN-ID] {run_id}")

        mlflow.log_params(h)

        mlp.fit(X_tr, y_tr)

        train_acc = mlp.score(X_tr, y_tr)
        val_acc   = mlp.score(X_val, y_val)
        y_pred  = mlp.predict(X_test)
        y_proba = mlp.predict_proba(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)
        auc_val = (roc_auc_score(y_test, y_proba[:, 1]) if n_classes == 2
                   else roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))

        mlflow.log_metric("train_accuracy", float(train_acc))
        mlflow.log_metric("val_accuracy",   float(val_acc))
        mlflow.log_metric("accuracy_test",  float(acc))
        mlflow.log_metric("precision_test", float(prec))
        mlflow.log_metric("recall_test",    float(rec))
        mlflow.log_metric("f1_test",        float(f1))
        mlflow.log_metric("roc_auc_test",   float(auc_val))

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion(cm, "Confusion Matrix (Test)", artifacts_dir / "confusion_matrix_test.png")
        pd.DataFrame(cm).to_csv(artifacts_dir / "confusion_matrix_test.csv", index=False)

        auc_plot = plot_roc_and_auc(y_test, y_proba, n_classes, artifacts_dir / "roc_curve_test.png")

        with open(artifacts_dir / "classification_report.txt", "w") as f:
            f.write(classification_report(y_test, y_pred, zero_division=0))

        with open(artifacts_dir / "metrics.json", "w") as f:
            json.dump({
                "train_accuracy": float(train_acc),
                "val_accuracy":   float(val_acc),
                "accuracy_test":  float(acc),
                "precision_test": float(prec),
                "recall_test":    float(rec),
                "f1_test":        float(f1),
                "roc_auc_test":   float(auc_val),
                "roc_auc_plot":   float(auc_plot)
            }, f, indent=2)

        target_path = models_dir / "mlp-models"
        if target_path.exists():
            shutil.rmtree(target_path)

        mlflow.sklearn.save_model(mlp, path=target_path)
        joblib.dump(mlp, models_dir / "mlp-pipeline.joblib")

        mlflow.sklearn.log_model(mlp, artifact_path="model")

if __name__ == "__main__":
    main()