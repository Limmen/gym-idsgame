import torch


class SixLayerFNN(torch.nn.Module):
    def __init__(self, input_dim : int, output_dim : int, hidden_dim : int):
        super(SixLayerFNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = 6

        # Define layers of FNN
        self.hidden_dim = hidden_dim
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.input_relu = torch.nn.ReLU()
        self.hidden_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_1_relu = torch.nn.ReLU()
        self.hidden_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_2_relu = torch.nn.ReLU()
        self.hidden_3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_3_relu = torch.nn.ReLU()
        self.hidden_4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.hidden_4_relu = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(hidden_dim, self.output_dim)


    def forward(self, x):
        input = self.input_relu(self.input_layer(x))
        hidden_1 = self.hidden_1_relu(self.hidden_1(input))
        hidden_2 = self.hidden_2_relu(self.hidden_2(hidden_1))
        hidden_3 = self.hidden_3_relu(self.hidden_3(hidden_2))
        hidden_4 = self.hidden_4_relu(self.hidden_4(hidden_3))
        y_hat = self.output_layer(hidden_4)
        return y_hat


def test():
    # Constants
    input_dim = 44
    output_dim = 44
    hidden_dim = 64
    batch_size = 64

    # Create model
    model = SixLayerFNN(input_dim, output_dim, hidden_dim)

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, output_dim)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the layers in the model
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(20000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if t % 100 == 99:
            print("step: {}, loss:{}".format(t, loss.item()))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



if __name__ == '__main__':
    test()

