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

        # U
        self.decoder_upconv1 = torch.nn.ConvTranspose2d(128, 256, kernel_size=3, padding=1, output_padding=1, stride=2)
        self.decoder_conv2 = torch.nn.Conv2d(256, n_classes, kernel_size=3, padding=1)

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

        # Down convolutions
        x1 = torch.relu(self.encoder_conv1(x))
        x2 = torch.relu(self.encoder_conv2(x1))

        # Up convolution
        x3 = torch.relu(self.decoder_upconv1(x2))

        # Crop the output to match the input size
        x3_cropped = x3[:, :, :x1.size()[2], :x1.size()[3]]

        x4 = self.decoder_conv2(x3_cropped)

        return x4