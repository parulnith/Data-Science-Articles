"""
client_app.py
Client-side logic for Flower simulation.
"""

import torch
from torch.utils.data import DataLoader

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    from task import load_client_datasets, train as train_fn, SimpleModel

    model = SimpleModel()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_id = context.node_config["partition-id"]
    client_parts = load_client_datasets()
    trainset = client_parts[client_id]

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        device,
    )

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(
                    {
                        "train_loss": train_loss,
                        "num-examples": len(trainset),
                    }
                ),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    from task import load_test_sets, test, SimpleModel
    import torch

    model = SimpleModel()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset, test_137, test_258, test_469 = load_test_sets()

    loss_all, acc_all = test(model, DataLoader(testset, batch_size=64), device)
    _, acc_137 = test(model, DataLoader(test_137, batch_size=64), device)
    _, acc_258 = test(model, DataLoader(test_258, batch_size=64), device)
    _, acc_469 = test(model, DataLoader(test_469, batch_size=64), device)

   
    # print(f"test accuracy on all digits: {acc_all*100:.2f}")
    # print(f"test accuracy on [1,3,7]: {acc_137*100:.2f}")
    # print(f"test accuracy on [2,5,8]: {acc_258*100:.2f}")
    # print(f"test accuracy on [4,6,9]: {acc_469*100:.2f}")

    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "acc_all": acc_all * 100,
                        "acc_137": acc_137 * 100,
                        "acc_258": acc_258 * 100,
                        "acc_469": acc_469 * 100,
                        "num-examples": len(testset),
                    }
                )
            }
        ),
        reply_to=msg,
    )

