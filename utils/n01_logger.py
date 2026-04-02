import random

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
from scipy import stats
from sklearn.metrics import accuracy_score
from torch import nn

## pring factorial function


def norm_plot(curves, title):
    fig, ax = plt.subplots()
    for mu, sigma, label in curves:
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), label=label)

    ax.set_title(title)
    ax.legend()


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * th.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * th.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = th.tensor(best_results)

            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")

    def plot_results(self, run_index=0):
        train_acc = [res[0] for res in self.results[run_index]]
        valid_acc = [res[1] for res in self.results[run_index]]
        test_acc = [res[2] for res in self.results[run_index]]

        plt.plot(train_acc, label='Train')
        plt.plot(valid_acc, label='Valid')
        plt.plot(test_acc, label='Test')
        plt.legend()
        plt.show()


def train_epoch(model, dataloader, optimizer, device):

    model = model.train()
    model = model.to(device)

    logging = dict()

    total_loss = total_examples = 0
    for batched_graph, labels in dataloader:


        optimizer.zero_grad()
        pred = model(batched_graph.to(device))
        labels = labels.view(-1)
        loss = F.cross_entropy(pred, labels.to(device))
        loss.backward()
        optimizer.step()

        logging.update(
            dict(
                bs=pred.shape[0],
                loss=loss.item(),
            )
        )

        # print(logging)
        num_examples = pred.shape[0]
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@th.no_grad()
def test(model, loader, device):
    model = model.to(device)
    model.eval()
    evaluator = Evaluator(name="ogbg-molhiv")

    y_true = []
    y_pred = []

    for batched_graph, labels in loader:
        batched_graph = batched_graph.to(device)

        pred = model(batched_graph)
        y_true.append(labels.view(-1).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = th.cat(y_true, dim=0).numpy()
    y_pred = th.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    y_pred_labels = y_pred.argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred_labels)

    # return evaluator.eval(input_dict)['rocauc'], accuracy
    return accuracy

def train(model, dl_train, dl_val, dl_test, device, logger, args, run):
    """
    A complete model training loop
    """
    optimizer = th.optim.Adam(
        model.parameters(),
        lr=args["lr"],
    )

    for epoch in range(1, 1 + args["epochs"]):
        loss = train_epoch(
            model,
            dl_train,
            optimizer,
            device,
        )

        if epoch % args["eval_steps"] == 0:
            # train_roc, train_acc = test(model, dl_train, device)
            # val_roc, val_acc = test(model, dl_val, device)
            # test_roc, test_acc = test(model, dl_test, device)

            train_acc = test(model, dl_train, device)
            val_acc = test(model, dl_val, device)
            test_acc = test(model, dl_test, device)

            result = [train_acc, val_acc, test_acc]
            logger.add_result(run, result)

            if epoch % args["log_steps"] == 0:
                print(
                    f"Run: {run+1:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {loss:.4f}, "
                    f"Train: {train_acc:.4f} Accuracy, "
                    f"Valid: {val_acc:.4f} Accuracy, "
                    f"Test: {test_acc:.4f} Accuracy"
                )
                print("---")

    return logger


def repeat_experiments(
    model, dl_train, dl_val, dl_test, device, train_args, n_runs
):
    logger = Logger(n_runs, train_args)

    for run in range(n_runs):
        model.reset_parameters()

        logger = train(
            model,
            dl_train,
            dl_val,
            dl_test,
            device,
            logger,
            train_args,
            run,
        )

        logger.print_statistics(run)

    logger.print_statistics()
    return logger
