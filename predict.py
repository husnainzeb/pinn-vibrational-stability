# predict.py

"""
Script to load a trained model checkpoint, evaluate its performance on a
benchmark dataset, and save the results with predictions to a new CSV file.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Import the model architecture from the model.py file
from model import BalancedMLP

# --- Configuration ---
CONFIG: Dict[str, Any] = {
    "paths": {
        "model_checkpoint": Path(
            "./models/physics_informed_model_checkpoint_seed_3141.pth"
        ),
        "benchmark_data": Path("./datasets/benchmark_dataset.csv"),
        "output_results_csv": Path("./datasets/benchmark_results.csv"),
    }
}

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_checkpoint(filepath: Path) -> Tuple[BalancedMLP, StandardScaler, List[str]]:
    """
    Loads a checkpoint bundle containing the model, scaler, and feature names.

    Args:
        filepath (Path): Path to the .pth checkpoint file.

    Returns:
        A tuple containing the loaded model, the fitted scaler, and the feature list.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Checkpoint file not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))

    if "feature_names" not in checkpoint:
        raise KeyError(
            "Checkpoint is outdated. Please regenerate it with the latest training.py script."
        )

    model = BalancedMLP(
        input_dim=checkpoint["input_dim"], dropout_rate=checkpoint["dropout_rate"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    scaler = checkpoint["scaler"]
    feature_names = checkpoint["feature_names"]

    logger.info(f"✅ Checkpoint loaded successfully from {filepath}")
    logger.info(f"   - Model was trained on {checkpoint['input_dim']} features.")

    return model, scaler, feature_names


def evaluate_on_benchmark(
    model: BalancedMLP,
    scaler: StandardScaler,
    feature_names: List[str],
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preprocesses the benchmark data, makes predictions, prints a classification report,
    and returns the DataFrame with an added prediction column.

    Returns:
        pd.DataFrame: The original benchmark DataFrame with a new 'predicted_state' column.
    """
    # 1. Prepare the data for prediction
    required_cols = feature_names + ["State"]
    if not all(col in benchmark_df.columns for col in required_cols):
        raise ValueError(
            "Benchmark CSV is missing required feature or 'State' columns."
        )

    X_raw = benchmark_df[feature_names].copy()
    X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_raw = X_raw.apply(lambda x: x.fillna(x.median()))

    y_true = benchmark_df["State"].values

    # 2. Scale the features using the loaded scaler
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # 3. Make predictions
    logger.info(
        f"Making predictions on {len(X_tensor)} samples from the benchmark dataset..."
    )
    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits.squeeze())
        y_pred = (probabilities >= 0.5).long().numpy()

    # 4. Print the final evaluation report to the console
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON BENCHMARK DATASET")
    print("=" * 60)
    print(
        classification_report(
            y_true, y_pred, target_names=["Unstable (0)", "Stable (1)"], digits=4
        )
    )
    print("=" * 60)

    # 5. Add predictions to the DataFrame and return it
    results_df = benchmark_df.copy()
    results_df["predicted_state"] = y_pred

    return results_df


if __name__ == "__main__":
    try:
        # 1. Load the complete model checkpoint
        loaded_model, loaded_scaler, feature_names_list = load_checkpoint(
            CONFIG["paths"]["model_checkpoint"]
        )

        # 2. Load the benchmark dataset
        benchmark_csv_path = CONFIG["paths"]["benchmark_data"]
        if not benchmark_csv_path.is_file():
            raise FileNotFoundError(
                f"Benchmark data file not found at: {benchmark_csv_path}"
            )

        df_benchmark = pd.read_csv(benchmark_csv_path)
        logger.info(
            f"✅ Benchmark dataset loaded successfully from {benchmark_csv_path}"
        )

        # 3. Run the evaluation and get the results DataFrame
        df_results = evaluate_on_benchmark(
            loaded_model, loaded_scaler, feature_names_list, df_benchmark
        )

        # 4. Save the DataFrame with predictions to a new CSV file
        output_path = CONFIG["paths"]["output_results_csv"]
        df_results.to_csv(output_path, index=False)
        logger.info(
            f"✅ Benchmark results with predictions saved successfully to: {output_path}"
        )

    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"\nAn error occurred: {e}")
        logger.error(
            "Please ensure you have run training.py and create_benchmark_dataset.py successfully."
        )
        sys.exit(1)
