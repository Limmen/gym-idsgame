"""
A CNN model with Softmax output defined in PyTorch
"""
import torch
from typing import List
from gym_idsgame.agents.training_agents.models.idsgame_resnet import IdsGameResNet

class CNNwithSoftmax(torch.nn.Module):
    """
    Implements a CNN with parameterizable number of layers, dimensions, and hidden activations.
    The CNN alternates pooling and Conv layers

    Sub-classing the torch.nn.Module to be able to use high-level API for creating the custom network
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int = 2,
                 hidden_activation: str = "ReLU", conv_kernels : List[int] = None, conv_strides : List[int] = None,
                 conv_out_channels : List[int] = None,
                 pool_kernels : List[int] = None, pool_strides : List[int] = None, pool : List[bool] = None,
                 flat_dim : int = 256, conv_2d :bool = True, conv_1d : bool = False):
        """
        Builds the model

        :param input_dim: the input dimension
        :param output_dim: the output dimension
        :param hidden_dim: the hidden dimension
        :param num_hidden_layers: the number of hidden layers
        :param hidden_activation: hidden activation type
        :param conv_kernels: size of the conv kernels
        :param conv_strides: size of the conv strides
        :param conv_out_channels: size of the output channels for conv layers
        :param pool_kernels: size of the pool kernels
        :param pool_strides: size of the pool strides
        :param pool: boolean vector whether to add a pooling layer after each conv layer
        :param flat_dim: dimension of the flatten layer at the end
        :param conv_1d: boolean flag, whether to use 1D convs
        :param conv_2d: boolean flag, whether to use 2D convs
        """
        super(CNNwithSoftmax, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_layers = num_hidden_layers + 2
        self.hidden_activation = hidden_activation
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.pool_kernels = pool_kernels
        self.pool_strides = pool_strides
        self.conv_out_channels = conv_out_channels
        self.pool = pool
        self.flat_dim = flat_dim
        self.conv_1d = conv_1d
        self.conv_2d = conv_2d

        # Define layers of CNN
        self.layers = torch.nn.ModuleList()

        # Input layer
        if self.conv_2d:
            self.layers.append(torch.nn.Conv2d(in_channels=input_dim[0], out_channels=3,
                                               kernel_size=2, stride=1,
                                               padding=0))
            # self.layers.append(torch.nn.Conv2d(in_channels=input_dim[0], out_channels=self.conv_out_channels[0],
            #                                    kernel_size=self.conv_kernels[0], stride=self.conv_strides[0], padding=0))
        elif self.conv_1d:
            self.layers.append(torch.nn.Conv1d(in_channels=input_dim[0], out_channels=self.conv_out_channels[0],
                                               kernel_size=self.conv_kernels[0], stride=self.conv_strides[0],
                                               padding=0))
        if pool[0]:
            self.layers.append(torch.nn.MaxPool2d(kernel_size=self.pool_kernels[0], stride=self.pool_strides[0], padding=0))

        # Hidden Layers
        for i in range(self.num_hidden_layers):
            if self.conv_2d:
                self.layers.append(torch.nn.Conv2d(in_channels=self.conv_out_channels[i-1], out_channels=self.conv_out_channels[i+1],
                                                   kernel_size=self.conv_kernels[i+1], stride=self.conv_strides[i+1],
                                                   padding=0))
            elif self.conv_1d:
                self.layers.append(torch.nn.Conv1d(in_channels=self.conv_out_channels[i - 1],
                                                   out_channels=self.conv_out_channels[i + 1],
                                                   kernel_size=self.conv_kernels[i + 1],
                                                   stride=self.conv_strides[i + 1],
                                                   padding=0))
            if pool[i+1]:
                self.layers.append(
                    torch.nn.MaxPool2d(kernel_size=self.pool_kernels[i+1], stride=self.pool_strides[i+1], padding=0))
        if self.conv_2d:
            self.layers.append(
                torch.nn.Conv2d(in_channels=self.conv_out_channels[-2], out_channels=self.conv_out_channels[-1],
                                kernel_size=self.conv_kernels[-1], stride=self.conv_strides[-1], padding=0))
        elif self.conv_1d:
            self.layers.append(
                torch.nn.Conv1d(in_channels=self.conv_out_channels[-2], out_channels=self.conv_out_channels[-1],
                                kernel_size=self.conv_kernels[-1], stride=self.conv_strides[-1], padding=0))
        # Output layer
        self.layers.append(torch.nn.Linear(self.flat_dim, self.output_dim))
        self.layers.append(torch.nn.Softmax())

    def get_hidden_activation(self):
        """
        Interprets the hidden activation

        :return: the hidden activation function
        """
        if self.hidden_activation == "ReLU":
            return torch.nn.ReLU()
        elif self.hidden_activation == "LeakyReLU":
            return torch.nn.LeakyReLU()
        elif self.hidden_activation == "LogSigmoid":
            return torch.nn.LogSigmoid()
        elif self.hidden_activation == "PReLU":
            return torch.nn.PReLU()
        elif self.hidden_activation == "Sigmoid":
            return torch.nn.Sigmoid()
        elif self.hidden_activation == "Softplus":
            return torch.nn.Softplus()
        elif self.hidden_activation == "Tanh":
            return torch.nn.Tanh()
        else:
            raise ValueError("Activation type: {} not recognized".format(self.hidden_activation))

    def forward(self, x):
        """
        Forward propagation

        :param x: input tensor
        :return: Output prediction
        """
        y = x
        # print("y shape:{}".format(y.shape))
        # y = torch.nn.Conv2d(3, out_channels=12, kernel_size=2, stride=1, padding=0)(y)
        # print("y shape:{}".format(y.shape))
        # y = torch.nn.ReLU()(y)
        # y = torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=2, stride=1, padding=0)(y)
        # print("y shape:{}".format(y.shape))
        # y = torch.nn.ReLU()(y)
        # y = torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=1, stride=1, padding=0)(y)
        # y = torch.nn.ReLU()(y)
        # print("y shape:{}".format(y.shape))
        # y = torch.nn.Flatten()(y)
        # print("y shape:{}".format(y.shape))
        # y = torch.nn.Linear(36, 44)(y)
        # print("y shape:{}".format(y.shape))
        # y = torch.nn.Softmax()(y)

        # self.cnn = torch.nn.Sequential(torch.nn.Conv2d(6, out_channels=64, kernel_size=1, stride=1, padding=0),
        #                          torch.nn.ReLU(),
        #                          torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
        #                          torch.nn.ReLU(),
        #                          torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
        #                          torch.nn.ReLU(),
        #                          torch.nn.Flatten(),
        #                          torch.nn.Linear(768, 44),
        #                          torch.nn.Softmax())

        # self.cnn = torch.nn.Sequential(torch.nn.Conv2d(6, out_channels=64, kernel_size=1, stride=1, padding=0),
        #                                torch.nn.MaxPool2d(kernel_size=2, stride=1,padding=0),
        #                                torch.nn.ReLU(),
        #                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1,
        #                                                padding=0),
        #                                torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
        #                                torch.nn.ReLU(),
        #                                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1,
        #                                                padding=0),
        #                                torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
        #                                torch.nn.ReLU(),
        #                                torch.nn.Flatten(),
        #                                torch.nn.Linear(768, 44),
        #                                torch.nn.Softmax())
        #resnet18 = models.resnet18(pretrained=False, num_classes=44)
        # my_resnet = IdsGameResNet(in_channels=6)
        # self.cnn = my_resnet
        # self.cnn = torch.nn.Sequential(torch.nn.Conv2d(3, out_channels=2, kernel_size=1, stride=1, padding=0),
        #                          torch.nn.ReLU(),
        #                          torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0),
        #                          torch.nn.ReLU(),
        #                          torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0),
        #                          torch.nn.ReLU(),
        #                          torch.nn.Flatten(),
        #                          torch.nn.Linear(6, 44),
        #                          torch.nn.Softmax())
        self.cnn = torch.nn.Sequential(torch.nn.Conv2d(3, out_channels=2, kernel_size=3, stride=1, padding=0),
                                 torch.nn.ReLU(),
                                 torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0),
                                 torch.nn.ReLU(),
                                 torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=0),
                                 torch.nn.ReLU(),
                                 torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1,padding=0),
                                 torch.nn.ReLU(),
                                 torch.nn.Flatten(),
                                 torch.nn.Linear(18, 10),
                                 torch.nn.Softmax())
        y = self.cnn(y)
        #print("y shape:{}".format(y.shape))
        # for i in range(len(self.layers)):
        #     print("layer i:{}".format(i))
        #     print("input shape:{}".format(y.shape))
        #     # Flatten
        #     if i == len(self.layers)-1:
        #         y = self.layers[i](y.view(y.size(0), -1))
        #     else:
        #         y = self.layers[i](y)
        return y


def test() -> None:
    """
    A basic test-case to verify that the model can fit some randomly generated data

    :return: None
    """
    # Constants
    input_dim = (3, 8, 8)
    output_dim = 10
    hidden_dim = 64
    batch_size = 1

    # Create model
    model = CNNwithSoftmax(input_dim, output_dim, hidden_dim, num_hidden_layers=2, conv_kernels=[2,1,1,2,2,2],
                           conv_strides=[1,1,1,1,1,1], conv_out_channels=[2,2,2,2,2,2], pool_kernels=[2,2,2,2,2],
                           pool_strides=[None, None, None, None, None], pool=[False, False, False, False, False, False],
                           flat_dim=3, conv_1d=False, conv_2d=True)

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(batch_size, input_dim[0], input_dim[1], input_dim[2])
    #x = torch.randn(batch_size, input_dim[0], input_dim[1])
    y = torch.empty(batch_size, dtype=torch.long).random_(output_dim)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the layers in the model
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(20000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        #print("y shape:{}, y_pred shape:{}".format(y.shape, y_pred.shape))
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print("step: {}, loss:{}".format(t, loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    test()
