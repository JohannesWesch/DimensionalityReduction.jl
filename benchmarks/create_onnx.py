import torch
import torchvision.transforms as transforms
from sklearn import datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn import datasets
import onnx
from skimage.transform import downscale_local_mean
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
#input-dim 16
#input_size = 16
#hidden_size1 = 8
#hidden_size2 = 64

#input-dim 64
input_size = 64
hidden_size1 = 32
hidden_size2 = 128


num_classes = 10
num_epochs = 2
batch_size = 32
learning_rate = 0.001
onnx_file = 'benchmarks/digits/digit-net_64x6.onnx'

# Load the dataset
sklearn_data = datasets.load_digits()
data = sklearn_data.data

scaled = []
for img in data:
    scaled.append(downscale_local_mean(img.reshape(8, 8), (2,2)).flatten())
scaled = np.array(scaled)

target = sklearn_data.target

# Convert the NumPy arrays to PyTorch tensors
#input dim 16
#data_tensor = torch.tensor(scaled, dtype=torch.float32)

#input dim 64
data_tensor = torch.tensor(data, dtype=torch.float32)

target_tensor = torch.tensor(target, dtype=torch.long)


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y


# Create an instance of the custom dataset
custom_dataset = CustomDataset(data_tensor, target_tensor)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensors
])

# Create a data loader
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)

# Fully connected neural network with one hidden layer
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet2, self).__init__()
        self.input_size = input_size
        self.l0 = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.l0(x)
        out = self.l1(out)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    
class NeuralNet4(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet4, self).__init__()
        self.input_size = input_size
        self.l0 = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, hidden_size2)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_size2, hidden_size2)
        self.relu4 = nn.ReLU()
        self.l5 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.l0(x)
        out = self.l1(out)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        out = self.relu4(out)
        out = self.l5(out)
        # no activation and no softmax at the end
        return out
    
class NeuralNet6(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet6, self).__init__()
        self.input_size = input_size
        self.l0 = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, hidden_size2)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_size2, hidden_size2)
        self.relu4 = nn.ReLU()
        self.l5 = nn.Linear(hidden_size2, hidden_size2)
        self.relu5 = nn.ReLU()
        self.l6 = nn.Linear(hidden_size2, hidden_size2)
        self.relu6 = nn.ReLU()
        self.l7 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.l0(x)
        out = self.l1(out)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        out = self.relu3(out)
        out = self.l4(out)
        out = self.relu4(out)
        out = self.l5(out)
        out = self.relu5(out)
        out = self.l6(out)
        out = self.relu6(out)
        out = self.l7(out)
        # no activation and no softmax at the end
        return out

# neural net with 2 hidden layers
#model = NeuralNet2(input_size, hidden_size1,hidden_size2, num_classes).to(device)

# neural net with 4 hidden layers
#model = NeuralNet4(input_size, hidden_size1,hidden_size2, num_classes).to(device)

# neural net with 6 hidden layers
model = NeuralNet6(input_size, hidden_size1,hidden_size2, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

# torch.save(model.state_dict(), 'mnist.pt')

dummy_input = torch.randn(1, input_size, 1).to(device)
input_names = ["input_0"]
output_names = ["output_0"]
torch.onnx.export(model, dummy_input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

#model = onnx.load(onnx_file)

#model.graph.node[2].output[0] = "relu1"
#model.graph.node[3].input[0] = "relu1"
#model.graph.node[4].output[0] = "relu2"
#model.graph.node[5].input[0] = "relu2"

#onnx.save(model, onnx_file)