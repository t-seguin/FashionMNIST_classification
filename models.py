import torch
import torch.nn as nn

L2_REG = 1e-3


class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.classifier = nn.Linear(self.input_size, num_classes)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)

        return y


def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)]


def linear_relu_dropout(dim_in, dim_out, dp_rate):
    return [nn.Dropout(dp_rate), nn.Linear(dim_in, dim_out), nn.ReLU(inplace=True)]


class FullyConnected(nn.Module):

    def __init__(self, input_size, num_classes):
        super(FullyConnected, self).__init__()
        self.classifier = nn.Sequential(
            *linear_relu(input_size, 256),
            *linear_relu(256, 256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


class FullyConnectedDropout(FullyConnected):

    def __init__(self, input_size, num_classes):
        super(FullyConnected, self).__init__()
        self.classifier = nn.Sequential(
            *linear_relu_dropout(input_size, 256, 0.2),
            *linear_relu_dropout(256, 256, 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        y = self.classifier(x)
        return y


class FullyConnectedRegularized(nn.Module):

    def __init__(self, input_size, num_classes, l2_reg=L2_REG):
        super(FullyConnectedRegularized, self).__init__()
        self.regularization_params = {"l2_reg": l2_reg}
        self.lin1 = nn.Linear(input_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, num_classes)

    def penalty(self):
        return self.regularization_params["l2_reg"] * (
            self.lin1.weight.norm(2)
            + self.lin2.weight.norm(2)
            + self.lin3.weight.norm(2)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = nn.functional.relu(self.lin1(x))
        x = nn.functional.relu(self.lin2(x))
        y = self.lin3(x)
        return y


def conv_relu(in_channels, out_chanels, kernel_size=3, stride=1, padding=1, dilation=1):

    return [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_chanels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ),
        # nn.BatchNorm2d(out_chanels),
        nn.ReLU(inplace=True),
    ]


class ConvNet(nn.Module):

    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.convlayers = nn.Sequential(
            *conv_relu(1, 16),
            nn.MaxPool2d(2, stride=2),
            *conv_relu(16, 32),
            nn.MaxPool2d(2, stride=2),
            *conv_relu(32, 64),
            nn.AdaptiveAvgPool2d(7),
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(64 * 49, num_classes), nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor):
        x = self.convlayers(x)
        x = x.view(x.size()[0], -1)
        y = self.linear_layer(x)

        return y


def load_model(model: nn.Module, model_path: str = None) -> nn.Module:

    if model_path:
        model.load_state_dict(torch.load(model_path))

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    return model


if __name__ == "__main__":

    model = LinearNet(1 * 28 * 28, 10)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
