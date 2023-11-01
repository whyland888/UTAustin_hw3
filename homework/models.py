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
    def __init__(self, n_input, n_output):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class TBlock(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=2, stride=2),
            torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU())

    def forward(self, x):
        return self.net(x)


class CNNClassifier(torch.nn.Module):

    def __init__(self, layers=[32,64,128], n_input_channels=3, n_classes=6, normalize=False):
        super().__init__()
        self.normalize = normalize

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
        if self.normalize:
            normalize = Normalize(mean=[0.3234, 0.3309, 0.3443], std=[0.4108, 0.3987, 0.4246])
            x = normalize(x)

        # Compute feature maps
        x = self.network(x)

        # Global average pooling (reduces height and width to a single number while retaining channels)
        x = x.mean(dim=[2, 3])

        return self.classifier(x)


class FCN(torch.nn.Module):
    def __init__(self, layers=[32,64,128], n_input_channels=3, n_classes=5, normalize=False):
        super().__init__()

        # Contracting path (Encoder)
        self.down1 = Block(n_input_channels, layers[0])
        self.down2 = Block(layers[0], layers[1])
        self.down3 = Block(layers[1], layers[2])

        # Expansive path (Decoder)
        self.up1 = TBlock(layers[2], layers[1])
        self.conv1 = torch.nn.Conv2d(in_channels=layers[2], out_channels=layers[1], kernel_size=3, stride=1,
                                     padding=1)
        self.up2 = TBlock(layers[1], layers[0])
        self.conv2 = torch.nn.Conv2d(in_channels=layers[1], out_channels=layers[0], kernel_size=3, stride=1,
                                     padding=1)
        self.up3 = TBlock(layers[0], n_classes)

        # Other layers
        # self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.max_pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder (downsampling)
        d1 = self.down1(x)
        d2 = self.max_pool(d1)

        d3 = self.down2(d2)
        d4 = self.max_pool(d3)

        d5 = self.down3(d4)
        d6 = self.max_pool(d5)

        # Decoder (upsampling)
        e1 = self.up1(d6)
        s1 = torch.cat([e1, d4], dim=1)
        c1 = self.conv1(s1)

        e2 = self.up2(c1)
        s2 = torch.cat([e2, d2], dim=1)
        c2 = self.conv2(s2)

        e3 = self.up3(c2)

        return e3


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
