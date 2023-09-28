import torch
import torch.nn.functional as F


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, input):
        x = input - self.mean.view(1, 3, 1, 1)
        x = x / self.std.view(1, 3, 1, 1)
        return x


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, stride=1, activation='relu'):
            super().__init__()
            if activation == 'relu':
                activation_layer = torch.nn.ReLU()
            elif activation == 'leaky_relu':
                activation_layer = torch.nn.LeakyReLU()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
                activation_layer,
                torch.nn.Conv2d(n_output, n_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(n_output),
                activation_layer
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, layers=[32,64,128], n_input_channels=3, n_classes=6, activation='relu'):
        super().__init__()

        # Activation Function
        if activation == 'relu':
            activation_layer = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            activation_layer = torch.nn.LeakyReLU(negative_slope=.1)

        # First layers
        L = [torch.nn.Conv2d(n_input_channels, 32, kernel_size=7, padding=3, stride=2),
             torch.nn.BatchNorm2d(32),
            activation_layer,
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        # Add Layers
        c = 32
        for l in layers:
            L.append(self.Block(c, l, stride=2))
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
    def __init__(self, in_channels=3, n_classes=5):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        self.encoder_conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)

        # Decoder (upsampling path)
        self.decoder_upconv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.decoder_conv2 = torch.nn.Conv2d(64, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """

        # Normalize input
        normalize = Normalize(mean=[0.2784, 0.2653, 0.2624], std=[0.3466, 0.3290, 0.3459])
        x = normalize(x)

        # Encoder
        x1 = torch.relu(self.encoder_conv1(x))
        x2 = torch.relu(self.encoder_conv2(x1))

        # Decoder
        x3 = torch.relu(self.decoder_upconv1(x2))

        # Crop the output to match the input size
        target_size = x1.size()[2:]  # Size of the encoder output
        crop_h = (x3.size()[2] - target_size[0]) // 2
        crop_w = (x3.size()[3] - target_size[1]) // 2
        x3_cropped = x3[:, :, crop_h:crop_h + target_size[0], crop_w:crop_w + target_size[1]]

        x4 = self.decoder_conv2(x3_cropped)

        return x4


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
