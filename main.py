# flake8: noqa F401

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models import (
    LinearNet,
    FullyConnected,
    FullyConnectedRegularized,
    FullyConnectedDropout,
    ConvNet,
    load_model,
)
from data import get_data_loader
from tqdm import tqdm
from train import train
from test import test
from utils import progress, redirect_to_tqdm
from log import ModelCheckpoint, generate_unique_logpath, write_summary
import argparse


EPOCHS = 5
TEST = True
TOP_LOGDIR = "./logs"
BATCH_SIZE = 128
MODEL_NAME = "LinearNet"  # To easier load the model to test
# MODEL_PATH = "logs/LinearNet_0/best_model.pt"
MODEL_PATH = ""

# -----------------------------------------------------

parser = argparse.ArgumentParser()
# Example arguments
parser.add_argument(
    "-e", "--eval", action="store_true", help="if true, evaluation mode: test only"
)
parser.add_argument(
    "-i",
    "--id_model",
    type=int,
    help="If MODEL_NAME precise, load the model MODEL_NAME_{id}",
    default=None,
)
parser.add_argument("-n", "--n_epochs", type=int, help="", default=EPOCHS)
args_dict = vars(parser.parse_args())
print(args_dict)

# -----------------------------------------------------

if args_dict.get("id_model") and MODEL_NAME:
    MODEL_PATH = f"logs/{MODEL_NAME}_{args_dict['id_model']}/best_model.pt"
TRAIN = not args_dict.get("eval")
EPOCHS = args_dict["n_epochs"]
if not TRAIN:
    EPOCHS = 1


train_loader, valid_loader, _ = get_data_loader(batch_size=BATCH_SIZE)

# model = LinearNet(1 * 28 * 28, 10)
# model = FullyConnected(1 * 28 * 28, 10)
# model = FullyConnectedRegularized(1 * 28 * 28, 10)
# model = FullyConnectedDropout(1 * 28 * 28, 10)
model = ConvNet(10)

if TRAIN:
    log_dir = generate_unique_logpath(TOP_LOGDIR, model._get_name())
    model_chekpoint = ModelCheckpoint(log_dir + "/best_model.pt", model)

train_loader = train_loader
f_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters())
use_gpu = torch.cuda.is_available()

if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = load_model(model, MODEL_PATH)

tensorboard_writer = SummaryWriter(log_dir=log_dir)

for t in tqdm(range(EPOCHS)):
    try:
        with redirect_to_tqdm():

            print(f"Epoch : {t+1}/{EPOCHS}")

            if TRAIN:
                train_loss, train_acc = train(
                    model, train_loader, f_loss, optimizer, device
                )
                model_chekpoint.update(train_loss)
                progress(train_loss, train_acc, "Train : ")

            if TEST:
                val_loss, val_acc = test(model, valid_loader, f_loss, device)
                progress(val_loss, val_acc, "Test : ")

            tensorboard_writer.add_scalar("metrics/train_loss", train_loss, t)
            tensorboard_writer.add_scalar("metrics/train_acc", train_acc, t)
            tensorboard_writer.add_scalar("metrics/val_loss", val_loss, t)
            tensorboard_writer.add_scalar("metrics/val_acc", val_acc, t)

    except KeyboardInterrupt:
        break

summary_text = write_summary(log_dir, model, optimizer)
tensorboard_writer.add_text("Experiment summary", summary_text)
