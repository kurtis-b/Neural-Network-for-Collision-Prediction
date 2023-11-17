import torch
import torch.nn as nn
from Data_Loaders import Data_Loaders


class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, hidden_size=256, output_size=1):
        # STUDENTS: __init__() must initiatize nn.Module and define your network's
        # custom architecture
        super(Action_Conditioned_FF, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Non-linearity
        self.relu = nn.ReLU()
        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Sigmoid to keep between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        # Define the forward pass through the network
        # Linear function
        output = self.fc1(input)
        # Non-linearity
        output = self.relu(output)
        # Linear function
        output = self.fc2(output)
        # Sigmoid (readout)
        output = self.sigmoid(output)
        return output

    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.
        model.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for _, test_sample in enumerate(test_loader):
                # Forward pass to get outputs/logits
                outputs = model(test_sample['input'])
                loss = loss_function(outputs, test_sample['label'])
                total_loss += loss.item()
                total_samples += test_sample['label'].size(0)
        return total_loss / total_samples  # return average loss


def main():
    # Load the dataset
    batch_size = 50  # sample size / batch size --> number of iterations. with 20000 samples: 400 iterations
    data_loaders = Data_Loaders(batch_size)

    # Make the dataset iterable
    n_iters = 80000  # for 200 epochs
    num_epochs = n_iters / (len(data_loaders.data) / batch_size)
    num_epochs = int(num_epochs)

    # Instantiate the model class
    # inputs: 5 sensor readings + action
    # output: one class where 1 means collision and 0 means no collision
    model = Action_Conditioned_FF()

    # Instantiate the loss class
    # loss_fn = nn.MSELoss()
    loss_fn = nn.BCELoss()

    # Instantiate the optimizer class
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    prev_loss = 0
    for epoch in range(num_epochs):
        for _, train_sample in enumerate(data_loaders.train_loader):
            # print(train_sample['input'].shape, train_sample['label'].shape)
            # Clear gradients with respect to parameters
            optimizer.zero_grad()
            # Forward pass to get outputs/logits
            outputs = model(train_sample['input'])
            # Calculate Loss:
            # print(outputs.shape)
            # print(outputs)
            loss = loss_fn(outputs, train_sample['label'])
            # print(loss)
            # Getting gradients with respect to parameters
            loss.backward()
            # Updating parameters
            optimizer.step()
        average_loss = model.evaluate(model, data_loaders.test_loader, loss_fn)
        print("epoch=", epoch + 1, "average loss=", average_loss)
        # if prev_loss > average_loss:
        #     break  # Early stop when the loss starts to increase
        # else:
        #     prev_loss = average_loss
    torch.save(model.state_dict(), "saved/saved_model.pkl")  # TODO: Seems like torch.save() doesn't overwrite the file?


if __name__ == '__main__':
    main()
