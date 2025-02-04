import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["interaction"] = pd.to_numeric(df["interaction"], errors='coerce')  # Ensure labels are numeric
    df.dropna(inplace=True)  # Remove rows with NaN values

    # Convert categorical columns to numerical using label encoding
    for col in ["drug_1", "drug_2"]:
        df[col] = df[col].astype("category").cat.codes

    X = df.drop(columns=["interaction"]).values.astype(np.float32)  # Ensure correct type
    y = df["interaction"].values.astype(np.float32)  # Target (1 = interaction, 0 = no interaction)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Define neural network for DDI prediction
class DrugInteractionPredictor(nn.Module):
    def __init__(self, input_size):
        super(DrugInteractionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


# Main function
def main():
    X_train, X_test, y_train, y_test = load_data("drug_interaction_data.csv")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)  # Ensure correct shape

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = DrugInteractionPredictor(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, epochs=10)
    print("Model training complete!")


if __name__ == "__main__":
    main()
