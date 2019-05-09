class Conv2D(nn.Module):

    def __init__(self, inchannel, outchannel, kernel_size, stride, padding, bias = True):
        super(Conv2D, self).__init__()

        self.inchannel = inchannel
        self.outchannel = outchannel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Weights and Bias initialisation.
        self.weights = nn.Parameter(torch.Tensor(outchannel, inchannel, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weights.data, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(outchannel, ))
            init.kaiming_uniform_(self.bias.data, a=math.sqrt(5))
        else:
            self.bias = None

    def forward(self, x):
        batch_size, channels, h_in, w_in = x.size()
        x_unfolded = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        out_unf = x_unfolded.transpose(1, 2).matmul(self.weights.view(self.weights.size(0), -1).t()).transpose(1, 2)

        h_out = int((h_in - self.kernel_size + 2 * self.padding) / self.stride + 1)
        w_out = int((w_in - self.kernel_size + 2 * self.padding) / self.stride + 1)

        if self.bias is not None:
            temp = torch.ones((batch_size, h_out * w_out, 1))
            b = self.bias.reshape(1, self.outchannel)
            out_unf += (temp @ b).transpose(1, 2)

        output = out_unf.view(batch_size, self.outchannel, h_out, w_out)

        return output


class MaxPool2D(nn.Module):

    def __init__(self, pooling_size):
        # assume pooling_size = kernel_size = stride
        super(MaxPool2D, self).__init__()
        self.pooling_size = pooling_size


    def forward(self, x):
        batch_size, c_x, h_x, w_x = x.size()
        h_out = int((h_x - self.pooling_size) / self.pooling_size + 1)
        w_out = int((w_x - self.pooling_size) / self.pooling_size + 1)

        x_unf = F.unfold(x, kernel_size=self.pooling_size, stride=self.pooling_size)
        x_unf = x_unf.view(batch_size, c_x, self.pooling_size**2, h_out*w_out)
        output, _ = torch.max(x_unf, 2)

        return output


# define resnet building blocks
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(Conv2D(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  nn.ReLU(inplace=True),
                                  Conv2D(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel))

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(Conv2D(inchannel, outchannel, kernel_size=1, stride=stride,
                                                 padding=0, bias=False),
                                          nn.BatchNorm2d(outchannel) )


    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# define resnet
class ResNet(nn.Module):

    def __init__(self, ResidualBlock, num_classes = 10):
        super(ResNet, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(Conv2D(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU())

        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride = 1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride = 2)
        self.maxpool = MaxPool2D(4)
        self.fc = nn.Linear(512, num_classes)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
