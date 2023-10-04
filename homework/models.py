import torch
import torch.nn.functional as F


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, input):
        x = input - self.mean.view(1, 3, 1, 1).to(self.device)
        x = x / self.std.view(1, 3, 1, 1).to(self.device)
        return x


class Block(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU()
        )
        self.downsample = None
        if stride != 1 or n_input != n_output:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(n_output))

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.net(x) + identity


class TBlock(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=3, padding=1,
                                     output_padding=1, stride=stride),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(n_output, n_output, kernel_size=3, padding=1,
                                     stride=1),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU()
      )
        self.downsample = None
        if stride != 1 or n_input != n_output:
            self.downsample = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=1,
                                         output_padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output))

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.net(x) + identity


class CNNClassifier(torch.nn.Module):

    def __init__(self, layers=[32,64,128], n_input_channels=3, n_classes=6):
        super().__init__()

        # First layers
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
             torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # Add Layers
        c = 32
        for l in layers:
            L.append(Block(c, l, stride=2))
            c = l

        # Put layers together
        self.network = torch.nn.Sequential(*L)

        # Append linear layer to the end
        self.classifier = torch.nn.Linear(layers[-1], n_classes)

    def forward(self, x):

        # Normalize input
        normalize = Normalize(mean=[0.3234, 0.3309, 0.3443], std=[0.4108, 0.3987, 0.4246])
        x = normalize(x)

        # Compute feature maps
        x = self.network(x)

        # Global average pooling (reduces height and width to a single number while retaining channels)
        x = x.mean(dim=[2, 3])

        return self.classifier(x)


class FCN(torch.nn.Module):
    def __init__(self, up_layers=[32], down_layers=[64], n_input_channels=3, n_classes=5):
        super().__init__()
        L = []

        # Add convolutional (downsampling) layers
        c = n_input_channels
        for l in up_layers:
            L.append(Block(c, l, stride=2))
            c = l

        # Add convolutional (upsampling) layers
        c = up_layers[-1]
        for l in down_layers:
            L.append(TBlock(c, l, stride=2))
            c = l

        # Add final layer
        L.append(torch.nn.Conv2d(down_layers[-1], n_classes, kernel_size=3, padding=1))

        # Put layers together
        self.network = torch.nn.Sequential(*L)

    def forward(self, x):

        # Input normalization
        normalize = Normalize(mean=[0.2784, 0.2653, 0.2624], std=[0.3466, 0.3290, 0.3459])
        x = normalize(x)

        # Save initial input
        input = x

        # Compute feature maps
        x = self.network(x)

        # Crop the output to match the input size
        x_cropped = x[:, :, :input.size()[2], :input.size()[3]]

        return x_cropped


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
