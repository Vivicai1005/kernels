import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

# Define a simple neural network by subclassing nn.Module.
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Check if CUDA is available, and set the device accordingly.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create an instance of the model and move it to the device.
model = SimpleNet(input_size=10, hidden_size=5, output_size=2).to(device)
print("Model is running on device:", device)

# Create a dummy input tensor with a batch size of 2 and 10 features, then move it to the device.
dummy_input = torch.randn(2, 10).to(device)
print("Dummy Input:\n", dummy_input)

# Pass the dummy input through the network to get the output.
output = model(dummy_input)
print("Output:\n", output)
