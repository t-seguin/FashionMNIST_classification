import torch
import torch.nn as nn


def train(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    f_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                     used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()
    tot_loss, correct = 0.0, 0.0
    N = 0
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        # Compute some metric for summary
        N += inputs.shape[0]
        tot_loss += inputs.shape[0] * loss.item()
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        try:
            model.penalty().backward()  # If ever the model gets a penalty
        except AttributeError:
            pass
        optimizer.step()

    return tot_loss / N, correct / N


if __name__ == "__main__":

    from models import LinearNet
    from data import get_data_loader
    from tqdm import tqdm

    train_loader, _, _ = get_data_loader()

    model = LinearNet(1 * 28 * 28, 10)
    train_loader = train_loader
    f_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # An example of calling train to learn over 10 epochs of the training set
    for i in tqdm(range(10)):
        train(model, train_loader, f_loss, optimizer, device)
