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

# --- Configuration ---
CONFIG: Dict[str, Any] = {
    "data": {
        "csv_path": "./datasets/new_combined_dataset_with_born_criteria.csv",
        "balance_strategy": "smote",
    },
    "model": {
        "input_dim": None,
        "dropout_rate": 0.4,
    },
    "training": {
        "epochs": 2000,
        "patience": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "clip_grad_norm": 1.0,
        # "lambda_penalty" is removed from the configuration.
        "focal_loss_alpha": 0.6,
        "focal_loss_gamma": 2.0,
        "scheduler_patience": 25,
        "scheduler_factor": 0.5,
    },
    "evaluation": {
        "test_size": 0.2,
        "validation_size": 0.25,
    },
    "random_states": {
        "global_seed": 42,
        "data_split_seed": 42,
    },
    "paths": {
        "logs_dir": Path("./logs"),
        "models_dir": Path("./models"),
        "plots_dir": Path("./plots"),
    },
}

# --- Logger Setup ---
logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path):
    """Configure logging to write to a file and the console."""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = Path(__file__).stem
    log_filename = log_dir / f"{script_name}_{timestamp}.log"
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
    """Set random seeds for reproducibility."""
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


# --- Data Handling ---
def load_and_preprocess_data(
    csv_path: str, balance_strategy: Optional[str], random_state: int
) -> Optional[Tuple[np.ndarray, ...]]:
    """
    Load data from CSV and preprocess. This logic is identical to the
    working physics-informed script to ensure the same data is used.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"Error: The file at {csv_path} was not found.")
        return None

    df = df.dropna(axis=1, how="all")
    df["Born_Criteria"] = pd.to_numeric(df["Born_Criteria"], errors="coerce")
    df["State"] = pd.to_numeric(df["State"], errors="coerce")
    df = df.dropna(subset=["Born_Criteria", "State"])
    df = df[df["Born_Criteria"].isin([0.0, 1.0])]

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.difference(
        ["State", "Born_Criteria"]
    )
    df = df.drop(columns=non_numeric_cols).replace([np.inf, -np.inf], np.nan)

    feature_cols = df.columns.difference(["State", "Born_Criteria"])
    df[feature_cols] = df[feature_cols].apply(lambda x: x.fillna(x.median()))
    df = df.dropna()

    X = df[feature_cols].values
    y = df["State"].astype(np.float32).values
    born_criteria_orig = df["Born_Criteria"].astype(np.float32).values

    logger.info(f"Original dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(
        f"Original class distribution - Stable: {(y == 1).sum()}, Unstable: {(y == 0).sum()}"
    )

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
        else:
            born_criteria = born_criteria_orig
        logger.info(f"Dataset balanced. New size: {X.shape[0]} samples.")
    else:
        born_criteria = born_criteria_orig

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    CONFIG["model"]["input_dim"] = X_scaled.shape[1]

    return train_test_split(
        X_scaled,
        y,
        born_criteria,
        test_size=CONFIG["evaluation"]["test_size"],
        random_state=random_state,
        stratify=y,
    )


# --- Model and Loss ---
class BalancedMLP(nn.Module):
    """The same powerful MLP architecture."""

    def __init__(self, input_dim: int, dropout_rate: float):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def focal_loss(
    pred_logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float
) -> torch.Tensor:
    """Compute the Focal Loss."""
    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        pred_logits.squeeze(), targets, reduction="none"
    )
    pt = torch.exp(-bce_loss)
    f_loss = alpha * (1 - pt) ** gamma * bce_loss
    return f_loss.mean()


# --- Training and Evaluation ---
class EarlyStopping:
    """Early stops training if validation loss doesn't improve."""

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
    """Train the model using Focal Loss."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
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
        "train_acc",
        "val_acc",
        "train_stable_acc",
        "val_stable_acc",
        "train_unstable_acc",
        "val_unstable_acc",
        "learning_rate",
    ]
    history = {k: [] for k in history_keys}

    logger.info(f"\n🚀 Starting Focal Loss Baseline Training...")

    for epoch in range(CONFIG["training"]["epochs"]):
        model.train()
        optimizer.zero_grad()

        train_outputs = model(data_tensors["X_train"])
        train_loss = focal_loss(
            train_outputs,
            data_tensors["y_train"],
            alpha=CONFIG["training"]["focal_loss_alpha"],
            gamma=CONFIG["training"]["focal_loss_gamma"],
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
            val_loss = focal_loss(
                val_outputs,
                data_tensors["y_val"],
                alpha=CONFIG["training"]["focal_loss_alpha"],
                gamma=CONFIG["training"]["focal_loss_gamma"],
            )

            train_preds = (torch.sigmoid(train_outputs.squeeze()) >= 0.5).float()
            val_preds = (torch.sigmoid(val_outputs.squeeze()) >= 0.5).float()

            history["train_loss"].append(train_loss.item())
            history["val_loss"].append(val_loss.item())
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
    model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor
) -> Dict[str, Any]:
    """Evaluate the final model on the test set. No physics consistency metric."""
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_probs = torch.sigmoid(test_outputs.squeeze())
        test_preds = (test_probs >= 0.5).float()
        y_test_np, test_preds_np = y_test.cpu().numpy(), test_preds.cpu().numpy()
        stable_mask, unstable_mask = (y_test_np == 1), (y_test_np == 0)
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
            "predictions": test_preds_np,
            "true_labels": y_test_np,
        }


def plot_training_history(history: Dict, save_path: Path):
    """
    Plots the training and validation history and saves it as a PDF file,
    adhering to specified graphical guidelines.
    """
    # Use a single font size for all plot elements for consistency.
    plt.rcParams.update({"font.size": 10})

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Define a consistent linewidth for all plots.
    linewidth = 0.6

    # Plot 1: Training and Validation Loss
    axes[0, 0].plot(
        epochs, history["train_loss"], "b-", label="Train Loss", linewidth=linewidth
    )
    axes[0, 0].plot(
        epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=linewidth
    )
    axes[0, 0].set_title("Training and Validation Loss (Focal)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    # Plot 2: Overall Accuracy
    axes[0, 1].plot(
        epochs, history["train_acc"], "b-", label="Train Accuracy", linewidth=linewidth
    )
    axes[0, 1].plot(
        epochs,
        history["val_acc"],
        "r-",
        label="Validation Accuracy",
        linewidth=linewidth,
    )
    axes[0, 1].set_title("Overall Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    # Plot 3: Class-wise Validation Accuracy
    axes[1, 0].plot(
        epochs,
        history["val_stable_acc"],
        "g-",
        label="Val Stable Acc",
        linewidth=linewidth,
    )
    axes[1, 0].plot(
        epochs,
        history["val_unstable_acc"],
        "y-",
        label="Val Unstable Acc",
        linewidth=linewidth,
    )
    axes[1, 0].set_title("Class-wise Validation Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    # Plot 4: Learning Rate
    axes[1, 1].plot(
        epochs, history.get("learning_rate", []), "purple", linewidth=linewidth
    )
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    plt.tight_layout()

    # Save the figure in PDF vector format for high quality.
    pdf_save_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_save_path, format="pdf")
    plt.close(fig)

    logger.info(f"Training history plot saved to {pdf_save_path}")


def save_model(model: nn.Module, path: Path):
    """Save the model's state dictionary."""
    torch.save(model.state_dict(), path)
    logger.info(f"Model saved to {path}")


# --- Main Execution ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    setup_logging(CONFIG["paths"]["logs_dir"])
    logger.info("Starting new FOCAL LOSS BASELINE training run.")

    for dir_path in ["models_dir", "plots_dir"]:
        CONFIG["paths"][dir_path].mkdir(exist_ok=True)

    set_seed(CONFIG["random_states"]["global_seed"])

    processed_data = load_and_preprocess_data(
        CONFIG["data"]["csv_path"],
        CONFIG["data"]["balance_strategy"],
        CONFIG["random_states"]["data_split_seed"],
    )
    if processed_data is None:
        sys.exit(1)

    X_train_full, X_test, y_train_full, y_test, born_train_full, born_test = (
        processed_data
    )
    logger.info("\n✅ Data loaded and preprocessed successfully!")

    # Perform validation split on all data
    X_train, X_val, y_train, y_val, _, _ = train_test_split(
        X_train_full,
        y_train_full,
        born_train_full,
        test_size=CONFIG["evaluation"]["validation_size"],
        random_state=CONFIG["random_states"]["data_split_seed"],
        stratify=y_train_full,
    )

    tensors = [X_train, X_val, X_test, y_train, y_val, y_test]
    tensor_names = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
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
        f"focal_loss_baseline_history_seed_{CONFIG['random_states']['global_seed']}.png"
    )
    plot_path = CONFIG["paths"]["plots_dir"] / plot_filename
    plot_training_history(history, plot_path)

    logger.info("\n📊 Evaluating final model on the test set...")
    # Note: evaluate_model is called without born_test
    results = evaluate_model(model, data_tensors["X_test"], data_tensors["y_test"])

    sep = "=" * 60
    logger.info("\n" + sep)
    logger.info("FINAL EVALUATION RESULTS (FOCAL LOSS BASELINE)")
    logger.info(sep)
    logger.info(f"Overall Accuracy:        {results['overall_accuracy']:.4f}")
    logger.info(f"Stable Class Accuracy:   {results['stable_accuracy']:.4f}")
    logger.info(f"Unstable Class Accuracy: {results['unstable_accuracy']:.4f}")
    logger.info(sep)

    seed = CONFIG["random_states"]["global_seed"]
    report = classification_report(
        results["true_labels"],
        results["predictions"],
        target_names=["Unstable (0)", "Stable (1)"],
        digits=4,
    )
    logger.info(f"\nDetailed Classification Report [seed: {seed}]:\n{report}")

    model_filename = f"focal_loss_baseline_model_{seed}.pth"
    model_path = CONFIG["paths"]["models_dir"] / model_filename
    save_model(model, model_path)
