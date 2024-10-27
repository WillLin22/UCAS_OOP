import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)
        nn.conv

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the neural network
model = SimpleNN()

# Define a loss function and an optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy input and target tensors
input_tensor = torch.randn(1, 10)
target_tensor = torch.randn(1, 1)

# Forward pass
output = model(input_tensor)
loss = criterion(output, target_tensor)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'Output: {output}')
print(f'Loss: {loss.item()}')