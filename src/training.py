
"""
Physics-Informed Neural Network Training Script (Refactored).

This script trains a Multi-Layer Perceptron (MLP) to classify material
stability and saves a complete checkpoint bundle (model weights, hyperparameters,
feature list, and the data scaler) for future inference.
"""

import argparse
import json
import logging
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import BalancedMLP


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return json.load(f)

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.json"
CONFIG = load_config(CONFIG_PATH)

# Convert path strings to Path objects
CONFIG["paths"]["logs_dir"] = Path(CONFIG["paths"]["logs_dir"])
CONFIG["paths"]["models_dir"] = Path(CONFIG["paths"]["models_dir"])
CONFIG["paths"]["plots_dir"] = Path(CONFIG["paths"]["plots_dir"])
if "results_dir" in CONFIG["paths"]:
    CONFIG["paths"]["results_dir"] = Path(CONFIG["paths"]["results_dir"])

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path):
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = log_dir / f"{Path(__file__).stem}_{timestamp}.log"
    file_handler = logging.FileHandler(log_filename)
    stream_handler = logging.StreamHandler(sys.stdout)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(file_formatter)
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed for torch, numpy, and random set to {seed}")


def load_and_preprocess_data(
    csv_path: str, balance_strategy: Optional[str], random_state: int
) -> Optional[Tuple[StandardScaler, List[str], Tuple]]:
    """
    Loads and preprocesses data using the exact logic from the original notebook.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Error: The file at {csv_path} was not found.")
        return None

    df = df.dropna(axis=1, how="all")

    # Normalize column names
    if "born_criteria" in df.columns:
        df = df.rename(columns={"born_criteria": "Born_Criteria"})
    if "state" in df.columns:
        df = df.rename(columns={"state": "State"})

    df["Born_Criteria"] = pd.to_numeric(df["Born_Criteria"], errors="coerce")
    df["State"] = pd.to_numeric(df["State"], errors="coerce")
    df = df.dropna(subset=["Born_Criteria", "State"])
    df = df[df["Born_Criteria"].isin([0.0, 1.0])]

    metadata_cols = ["material_id", "Composition", "band_gap", "crystal_system"]
    existing_metadata_cols = [c for c in metadata_cols if c in df.columns]
    metadata_orig = df[existing_metadata_cols].copy()

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.difference(
        ["State", "Born_Criteria", "composition", "Composition", "material_id", "crystal_system"]
    )
    df = df.drop(columns=non_numeric_cols)
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = df.columns.difference(
        ["State", "Born_Criteria", "composition", "Composition", "material_id", "crystal_system"]
    ).tolist()
    CONFIG["model"]["feature_names"] = feature_cols

    df[feature_cols] = df[feature_cols].apply(lambda x: x.fillna(x.median()))
    df = df.dropna(subset=feature_cols + ["State", "Born_Criteria"])

    if df.empty:
        logger.error(
            "DataFrame is empty after cleaning. Please check the input CSV and cleaning steps."
        )
        return None

    X = df[feature_cols].values
    y = df["State"].astype(np.float32).values
    born_criteria_orig = df["Born_Criteria"].astype(np.float32).values

    logger.info(
        f"Original cleaned dataset: {X.shape[0]} samples, {X.shape[1]} features"
    )
    logger.info(
        f"Original class distribution - Stable: {(y == 1).sum()}, Unstable: {(y == 0).sum()}"
    )

    born_criteria = born_criteria_orig
    if balance_strategy == "smote":
        logger.info(f"\nApplying SMOTE (random_state={random_state})...")
        smote = SMOTE(random_state=random_state)
        X, y = smote.fit_resample(X, y)

        n_original = len(born_criteria_orig)
        n_new = len(y) - n_original
        if n_new > 0:
            new_born_criteria = np.random.choice(
                born_criteria_orig, size=n_new, replace=True
            )
            born_criteria = np.concatenate([born_criteria_orig, new_born_criteria])

    logger.info(f"\nFinal dataset size after balancing: {X.shape[0]} samples")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    CONFIG["model"]["input_dim"] = X_scaled.shape[1]


    n_synthetic = len(y) - len(born_criteria_orig)
    if n_synthetic > 0:
        synthetic_metadata = pd.DataFrame(index=range(n_synthetic), columns=existing_metadata_cols)
        for col in existing_metadata_cols:
            synthetic_metadata[col] = "Synthetic"
            if col == "band_gap":
                 synthetic_metadata[col] = -1.0 # Placeholder
        
        metadata_combined = pd.concat([metadata_orig, synthetic_metadata], ignore_index=True)
    else:
        metadata_combined = metadata_orig.reset_index(drop=True)

    X_train, X_test, y_train, y_test, born_train, born_test, meta_train, meta_test = train_test_split(
        X_scaled,
        y,
        born_criteria,
        metadata_combined,
        test_size=CONFIG["evaluation"]["test_size"],
        random_state=random_state,
        stratify=y,
    )
    
    X_train, X_val, y_train, y_val, born_train, born_val, meta_train, meta_val = train_test_split(
        X_train,
        y_train,
        born_train,
        meta_train,
        test_size=CONFIG["evaluation"]["validation_size"],
        random_state=random_state,
        stratify=y_train,
    )

    split_data = (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        born_train, born_val, born_test,
        meta_train, meta_val, meta_test
    )

    return scaler, feature_cols, split_data



def focal_loss(
    pred_logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float
) -> torch.Tensor:
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_logits.squeeze(), targets, reduction="none"
    )
    pt = torch.exp(-bce_loss)
    f_loss = alpha * (1 - pt) ** gamma * bce_loss
    return f_loss.mean()


def physics_informed_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    born_criteria: torch.Tensor,
    lambda_penalty: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch.isnan(pred_logits).any():
        return (
            torch.tensor(float("nan")),
            torch.tensor(float("nan")),
            torch.tensor(0.0),
        )
    prediction_loss = focal_loss(
        pred_logits,
        targets,
        alpha=CONFIG["training"]["focal_loss_alpha"],
        gamma=CONFIG["training"]["focal_loss_gamma"],
    )
    probs = torch.sigmoid(pred_logits.squeeze())
    inconsistent_mask = (probs >= 0.5) & (born_criteria == 0.0)
    physics_penalty = inconsistent_mask.float().sum() / targets.shape[0]
    total_loss = prediction_loss + lambda_penalty * physics_penalty
    return total_loss, prediction_loss, physics_penalty


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float, restore_best_weights: bool):
        self.patience, self.min_delta, self.restore_best_weights = (
            patience,
            min_delta,
            restore_best_weights,
        )
        self.best_loss, self.counter, self.best_weights, self.best_epoch = (
            float("inf"),
            0,
            None,
            0,
        )

    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.counter, self.best_epoch = val_loss, 0, epoch
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def train_with_validation(
    model: nn.Module, data_tensors: Dict[str, torch.Tensor]
) -> Dict[str, List[float]]:
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=CONFIG["training"]["scheduler_factor"],
        patience=CONFIG["training"]["scheduler_patience"],
        verbose=False,
    )
    early_stopping = EarlyStopping(
        patience=CONFIG["training"]["patience"],
        min_delta=1e-5,
        restore_best_weights=True,
    )
    history_keys = [
        "train_loss",
        "val_loss",
        "train_pred_loss",
        "val_pred_loss",
        "train_reward",
        "val_reward",
        "train_acc",
        "val_acc",
        "train_stable_acc",
        "val_stable_acc",
        "train_unstable_acc",
        "val_unstable_acc",
        "learning_rate",
    ]
    history = {k: [] for k in history_keys}
    logger.info(
        f"\n🚀 Starting Training (patience={CONFIG['training']['patience']})..."
    )
    for epoch in range(CONFIG["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        train_outputs = model(data_tensors["X_train"])
        train_loss, train_pred_loss, train_reward = physics_informed_loss(
            train_outputs,
            data_tensors["y_train"],
            data_tensors["born_train"],
            lambda_penalty=CONFIG["training"]["lambda_penalty"],
        )
        if torch.isnan(train_loss):
            logger.error(f"❌ NaN loss at epoch {epoch + 1}! Stopping.")
            break
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=CONFIG["training"]["clip_grad_norm"]
        )
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(data_tensors["X_val"])
            val_loss, val_pred_loss, val_reward = physics_informed_loss(
                val_outputs,
                data_tensors["y_val"],
                data_tensors["born_val"],
                lambda_penalty=CONFIG["training"]["lambda_penalty"],
            )
            train_preds = (torch.sigmoid(train_outputs.squeeze()) >= 0.5).float()
            val_preds = (torch.sigmoid(val_outputs.squeeze()) >= 0.5).float()
            history["train_loss"].append(train_loss.item())
            history["val_loss"].append(val_loss.item())
            history["train_pred_loss"].append(train_pred_loss.item())
            history["val_pred_loss"].append(val_pred_loss.item())
            history["train_reward"].append(train_reward.item())
            history["val_reward"].append(val_reward.item())
            history["learning_rate"].append(optimizer.param_groups[0]["lr"])
            history["train_acc"].append(
                (train_preds == data_tensors["y_train"]).float().mean().item()
            )
            history["train_stable_acc"].append(
                (
                    train_preds[data_tensors["y_train"] == 1]
                    == data_tensors["y_train"][data_tensors["y_train"] == 1]
                )
                .float()
                .mean()
                .item()
                if (data_tensors["y_train"] == 1).any()
                else 0.0
            )
            history["train_unstable_acc"].append(
                (
                    train_preds[data_tensors["y_train"] == 0]
                    == data_tensors["y_train"][data_tensors["y_train"] == 0]
                )
                .float()
                .mean()
                .item()
                if (data_tensors["y_train"] == 0).any()
                else 0.0
            )
            history["val_acc"].append(
                (val_preds == data_tensors["y_val"]).float().mean().item()
            )
            history["val_stable_acc"].append(
                (
                    val_preds[data_tensors["y_val"] == 1]
                    == data_tensors["y_val"][data_tensors["y_val"] == 1]
                )
                .float()
                .mean()
                .item()
                if (data_tensors["y_val"] == 1).any()
                else 0.0
            )
            history["val_unstable_acc"].append(
                (
                    val_preds[data_tensors["y_val"] == 0]
                    == data_tensors["y_val"][data_tensors["y_val"] == 0]
                )
                .float()
                .mean()
                .item()
                if (data_tensors["y_val"] == 0).any()
                else 0.0
            )
        scheduler.step(val_loss.item())
        if early_stopping(val_loss.item(), model, epoch):
            logger.info(
                f"🛑 Early stopping at epoch {epoch + 1}. Best validation loss {early_stopping.best_loss:.4f} at epoch {early_stopping.best_epoch + 1}."
            )
            break
        if (epoch + 1) % 100 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{CONFIG['training']['epochs']} | Val Loss: {history['val_loss'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f} | Val Unstable Acc: {history['val_unstable_acc'][-1]:.4f}"
            )
    return history


def evaluate_model(
    model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    born_test: torch.Tensor,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_probs = torch.sigmoid(test_outputs.squeeze())
        test_preds = (test_probs >= 0.5).float()
        y_test_np, test_preds_np = y_test.cpu().numpy(), test_preds.cpu().numpy()
        stable_mask, unstable_mask = (y_test_np == 1), (y_test_np == 0)
        physics_matches = ((test_preds == 1) & (born_test == 1)).sum()
        physics_opportunities = (born_test == 1).sum()
        return {
            "overall_accuracy": (test_preds == y_test).float().mean().item(),
            "stable_accuracy": (
                test_preds_np[stable_mask] == y_test_np[stable_mask]
            ).mean()
            if stable_mask.any()
            else 0.0,
            "unstable_accuracy": (
                test_preds_np[unstable_mask] == y_test_np[unstable_mask]
            ).mean()
            if unstable_mask.any()
            else 0.0,
            "physics_consistency": (physics_matches / physics_opportunities).item()
            if physics_opportunities > 0
            else 0.0,
            "predictions": test_preds_np,
            "true_labels": y_test_np,
        }


def plot_training_history(history: Dict, save_path: Path):
    """
    Plots the training and validation history and saves it as a PDF file.

    This function is updated to conform to the specified plotting guidelines:
    - Saves the plot in PDF format for vector quality.
    - Uses a single, consistent font size for all text elements.
    - Removes plot borders (spines) for a cleaner appearance.
    - Sets a specific linewidth for better visibility.
    """
    # Set a consistent font size for all plot elements.
    plt.rcParams.update({"font.size": 10})

    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Define a consistent linewidth
    linewidth = 0.6

    def plot(ax, key1, key2, title):
        ax.plot(
            epochs,
            history[key1],
            "b-",
            label=key1.replace("_", " ").title(),
            linewidth=linewidth,
        )
        ax.plot(
            epochs,
            history[key2],
            "r-",
            label=key2.replace("_", " ").title(),
            linewidth=linewidth,
        )
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        # Remove top and right borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plot(axes[0, 0], "train_loss", "val_loss", "Total Loss")
    plot(axes[0, 1], "train_pred_loss", "val_pred_loss", "Prediction Loss (Focal)")
    plot(axes[0, 2], "train_reward", "val_reward", "Physics Reward")
    plot(axes[1, 0], "train_acc", "val_acc", "Overall Accuracy")

    axes[1, 1].plot(
        epochs,
        history["train_stable_acc"],
        "c-",
        label="Train Stable Acc",
        linewidth=linewidth,
    )
    axes[1, 1].plot(
        epochs,
        history["val_stable_acc"],
        "g-",
        label="Val Stable Acc",
        linewidth=linewidth,
    )
    axes[1, 1].plot(
        epochs,
        history["train_unstable_acc"],
        "m-",
        label="Train Unstable Acc",
        linewidth=linewidth,
    )
    axes[1, 1].plot(
        epochs,
        history["val_unstable_acc"],
        "y-",
        label="Val Unstable Acc",
        linewidth=linewidth,
    )
    axes[1, 1].set_title("Class-wise Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    axes[1, 2].plot(
        epochs, history.get("learning_rate", []), "purple", linewidth=linewidth
    )
    axes[1, 2].set_title("Learning Rate")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_yscale("log")
    axes[1, 2].grid(True)
    axes[1, 2].spines["top"].set_visible(False)
    axes[1, 2].spines["right"].set_visible(False)

    plt.tight_layout()
    # Save the figure in PDF format
    plt.savefig(save_path.with_suffix(".pdf"), format="pdf")
    plt.close(fig)
    logger.info(f"Training history plot saved to {save_path.with_suffix('.pdf')}")


def save_checkpoint(
    model: nn.Module, scaler: StandardScaler, feature_names: List[str], path: Path
):
    """Saves a complete checkpoint bundle for inference."""
    checkpoint = {
        "input_dim": CONFIG["model"]["input_dim"],
        "dropout_rate": CONFIG["model"]["dropout_rate"],
        "state_dict": model.state_dict(),
        "scaler": scaler,
        "feature_names": feature_names,
    }
    torch.save(checkpoint, path)
    logger.info(f"Complete checkpoint saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Physics-Informed Neural Network")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to the configuration JSON file",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Path to the dataset CSV file (overrides config)",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=None,
        help="Directory to save logs (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed (overrides config)",
    )
    args = parser.parse_args()

    # Load config from argument if provided
    if args.config:
        CONFIG = load_config(Path(args.config))
        # Re-convert paths
        CONFIG["paths"]["logs_dir"] = Path(CONFIG["paths"]["logs_dir"])
        CONFIG["paths"]["models_dir"] = Path(CONFIG["paths"]["models_dir"])
        CONFIG["paths"]["plots_dir"] = Path(CONFIG["paths"]["plots_dir"])

    # Override with command line arguments if provided
    if args.csv_path:
         CONFIG["data"]["csv_path"] = args.csv_path
    if args.logs_dir:
         CONFIG["paths"]["logs_dir"] = Path(args.logs_dir)
    if args.seed is not None:
         CONFIG["random_states"]["global_seed"] = args.seed

    warnings.filterwarnings("ignore")
    setup_logging(CONFIG["paths"]["logs_dir"])
    logger.info("Starting new experiment run.")

    for dir_path in ["models_dir", "plots_dir"]:
        CONFIG["paths"][dir_path].mkdir(exist_ok=True)

    set_seed(CONFIG["random_states"]["global_seed"])

    result = load_and_preprocess_data(
        CONFIG["data"]["csv_path"],
        CONFIG["data"]["balance_strategy"],
        random_state=CONFIG["random_states"]["data_split_seed"],
    )
    if result is None:
        sys.exit(1)

    scaler, feature_names, processed_data = result

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        born_train, born_val, born_test,
        meta_train, meta_val, meta_test
    ) = processed_data

    logger.info("\n✅ Data loaded and preprocessed successfully!")

    tensors = [
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        born_train,
        born_val,
        born_test,
    ]
    tensor_names = [
        "X_train",
        "X_val",
        "X_test",
        "y_train",
        "y_val",
        "y_test",
        "born_train",
        "born_val",
        "born_test",
    ]
    data_tensors = {
        name: torch.tensor(arr, dtype=torch.float32)
        for name, arr in zip(tensor_names, tensors)
    }

    logger.info("\n📊 Dataset splits:")
    logger.info(
        f"Train samples: {data_tensors['X_train'].shape[0]}, Validation: {data_tensors['X_val'].shape[0]}, Test: {data_tensors['X_test'].shape[0]}"
    )

    model = BalancedMLP(
        input_dim=CONFIG["model"]["input_dim"],
        dropout_rate=CONFIG["model"]["dropout_rate"],
    )
    history = train_with_validation(model, data_tensors)
    logger.info("\n✅ Training completed!")

    plot_filename = (
        f"training_history_seed_{CONFIG['random_states']['global_seed']}.png"
    )
    plot_path = CONFIG["paths"]["plots_dir"] / plot_filename
    plot_training_history(history, plot_path)

    logger.info("\n📊 Evaluating final model on the test set...")
    results = evaluate_model(
        model, data_tensors["X_test"], data_tensors["y_test"], data_tensors["born_test"]
    )
    sep = "=" * 60
    logger.info("\n" + sep)
    logger.info("FINAL EVALUATION RESULTS")
    logger.info(sep)
    logger.info(f"Overall Accuracy:        {results['overall_accuracy']:.4f}")
    logger.info(f"Stable Class Accuracy:   {results['stable_accuracy']:.4f}")
    logger.info(f"Unstable Class Accuracy: {results['unstable_accuracy']:.4f}")
    logger.info(f"Physics Consistency:     {results['physics_consistency']:.4f}")
    logger.info(sep)
    seed = CONFIG["random_states"]["global_seed"]
    report = classification_report(
        results["true_labels"],
        results["predictions"],
        target_names=["Unstable (0)", "Stable (1)"],
        digits=4,
    )
    logger.info(f"\nDetailed Classification Report [seed: {seed}]:\n{report}")

    model_filename = f"physics_informed_model_checkpoint_seed_{seed}.pth"
    model_path = CONFIG["paths"]["models_dir"] / model_filename
    save_checkpoint(model, scaler, feature_names, model_path)

    # Save detailed results
    if "results_dir" in CONFIG["paths"]:
        results_dir = Path(CONFIG["paths"]["results_dir"])
        results_dir.mkdir(exist_ok=True)
        
        def save_results(X, y, meta, filename):
            model.eval()
            with torch.no_grad():
                outputs = model(torch.tensor(X, dtype=torch.float32))
                preds = (torch.sigmoid(outputs.squeeze()) >= 0.5).float().numpy()
            
            df_res = meta.copy()
            df_res["State"] = y
            df_res["predicted_state"] = preds
            df_res.to_csv(results_dir / filename, index=False)
            logger.info(f"Saved results to {results_dir / filename}")

        save_results(X_train, y_train, meta_train, "train_results.csv")
        save_results(X_val, y_val, meta_val, "val_results.csv")
        save_results(X_test, y_test, meta_test, "test_results.csv")
