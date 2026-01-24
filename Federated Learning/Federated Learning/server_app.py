"""
server_app.py
Server-side orchestration for MNIST Flower simulation.
"""

import os
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_LOG_TO_STDERR"] = "0"

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from task import (
    SimpleModel,
    load_test_sets,
    compute_confusion_matrix,
    plot_confusion_matrix,
)

# -------------------------------------------------
# Flower ServerApp
# -------------------------------------------------
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # -----------------------------
    # Read run config
    # -----------------------------
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # -----------------------------
    # Initialize global model
    # -----------------------------
    model = SimpleModel()
    arrays = ArrayRecord(model.state_dict())

    # -----------------------------
    # FedAvg strategy
    # -----------------------------
    strategy = FedAvg(
        fraction_train=1.0,
        min_available_nodes=3,
    )

    # -----------------------------
    # Run federated training
    # -----------------------------
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # -----------------------------
    # Print final aggregated metrics
    # -----------------------------
    print("\nFinal Global Model Evaluation (aggregated):")

    eval_metrics = result.evaluate_metrics_clientapp

    if eval_metrics:
        final_round = max(eval_metrics.keys())
        metrics = eval_metrics[final_round]

        print(f"Accuracy (all digits): {metrics['acc_all']:.2f}%")
        print(f"Accuracy [1,3,7]:     {metrics['acc_137']:.2f}%")
        print(f"Accuracy [2,5,8]:     {metrics['acc_258']:.2f}%")
        print(f"Accuracy [4,6,9]:     {metrics['acc_469']:.2f}%")
    else:
        print("No client-side evaluation metrics available.")

    # -----------------------------
    # Confusion matrix (ONCE)
    # -----------------------------
    print("\nComputing confusion matrix for final global model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    final_model = SimpleModel()
    final_model.load_state_dict(result.arrays.to_torch_state_dict())
    final_model.to(device)

    testset, _, _, _ = load_test_sets()

    cm = compute_confusion_matrix(final_model, testset, device)
  # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        title="Final Global Model Confusion Matrix",
        filename="confusion_matrix_final.png",
    )

    # -----------------------------
    # Save final model
    # -----------------------------
    torch.save(
        result.arrays.to_torch_state_dict(),
        "final_model.pt",
    )

    print("\nFinal global model saved to final_model.pt")
