import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils.rnn import pad_sequence


import torchbnn as bnn
import torch
import torch.nn as nn
import torch.optim as optim
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, 128)  # Embedding layer for word embeddings
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        h0 = torch.zeros(self.num_layers * 2 if self.lstm.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2 if self.lstm.bidirectional else self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(embedded, (h0, c0))
        out = self.fc(out[:, -1, :])  # Get the last time step
        return out

# Model Parameters
input_size = len(vectorizer.vocabulary_)
hidden_size = 64
output_size = len(label_encoder.classes_)

# Initialize the model, loss function, and optimizer
model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Train the Model

def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Train the model
train_model(model, train_loader, criterion, optimizer)

# 4. Evaluate the Model

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Evaluate on the test dataset
evaluate_model(model, test_loader)