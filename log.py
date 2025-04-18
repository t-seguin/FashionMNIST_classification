# flake8: noqa E501

import os
import torch
import sys


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
            return log_path
        i = i + 1


if __name__ == "__main__":

    # Example usage :
    # 1- create the directory "./logs" if it does not exist
    top_logdir = "./logs"
    if not os.path.exists(top_logdir):
        os.mkdir(top_logdir)

    # 2- We test the function by calling several times our function
    logdir = generate_unique_logpath(top_logdir, "linear")
    print("Logging to {}".format(logdir))
    # -> Prints out     Logging to   ./logs/linear_0
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    logdir = generate_unique_logpath(top_logdir, "linear")
    print("Logging to {}".format(logdir))
    # -> Prints out     Logging to   ./logs/linear_1
    if not os.path.exists(logdir):
        os.mkdir(logdir)


class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.filepath)
            # torch.save(self.model, self.filepath)
            self.min_loss = loss


def write_summary(
    log_dir: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    summary_file = open(log_dir + "/summary.txt", "w")
    summary_text = f"""
Executed command
================
{"python " + " ".join(sys.argv)}

Dataset
=======
FashionMNIST

Model summary
=============
{model}

{sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters

{"Regularization params : " + "|".join([f"{param}={value}" for param, value in model.regularization_params.items()]) if vars(model).get("regularization_params") else ""}

Optimizer
========
{optimizer}
"""
    summary_file.write(summary_text)
    summary_file.close()

    return summary_text
