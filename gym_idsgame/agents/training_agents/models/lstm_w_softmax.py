"""
A LSTM model with Softmax output defined in PyTorch
"""
import torch


class LSTMwithSoftmax(torch.nn.Module):
    """
    Implements a LSTM with parameterizable number of layers, dimensions, and hidden activations.

    Sub-classing the torch.nn.Module to be able to use high-level API for creating the custom network
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_lstm_layers: int = 2,
                 num_hidden_linear_layers: int = 2,
                 hidden_activation: str = "ReLU", seq_length : int = 4):
        """
        Builds the model

        :param input_dim: the input dimension
        :param output_dim: the output dimension
        :param hidden_dim: the hidden dimension
        :param num_lstm_layers: the number of hidden layers
        :param num_hidden_linear_layers: the number of linear layers
        :param hidden_activation: hidden activation type
        :param seq_length: length of the sequence
        """
        super(LSTMwithSoftmax, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_lstm_layers
        self.num_lstm_layers = num_lstm_layers
        self.num_hidden_linear_layers = num_hidden_linear_layers
        self.num_layers = self.num_lstm_layers + self.num_hidden_linear_layers + 1
        self.hidden_activation = hidden_activation

        # Define layers of FNN
        self.layers = torch.nn.ModuleList()

        # Input layer
        self.layers.append(torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                                         num_layers=num_lstm_layers,
                                         batch_first=True))

        # self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        # self.layers.append(self.get_hidden_activation())

        # Hidden Layers
        for i in range(self.num_hidden_linear_layers):
            if i == 0:
                #self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(torch.nn.Linear(hidden_dim*seq_length, hidden_dim))
            else:
                self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(self.get_hidden_activation())

        # Output layer
        self.layers.append(torch.nn.Linear(hidden_dim, self.output_dim))
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
        hidden_vec, last_hidden = self.layers[0](y)
        y = hidden_vec
        # flatten
        y = torch.reshape(y, (y.shape[0], y.shape[1]*y.shape[2]))
        for i in range(1, len(self.layers)):
            y = self.layers[i](y)
        return y


def test() -> None:
    """
    A basic test-case to verify that the model can fit some randomly generated data

    :return: None
    """
    # Constants
    batch_size = 64
    input_dim = 44
    output_dim = 44
    hidden_dim = 64

    # Create model
    model = LSTMwithSoftmax(input_dim, output_dim, hidden_dim, num_lstm_layers=2)

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(batch_size, 4, input_dim)
    y = torch.empty(batch_size, dtype=torch.long).random_(output_dim)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the layers in the model
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(20000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        print("y_pred shape:{}".format(y_pred.shape))
        loss = criterion(y_pred, y.squeeze())
        if t % 100 == 99:
            print("step: {}, loss:{}".format(t, loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    test()
