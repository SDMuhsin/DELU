import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram
from torchvision.models import resnet18, resnet34, resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class AudioDataset(SPEECHCOMMANDS):
    def __init__(self, subset):
        super().__init__("./data", download=True, subset=subset)
        self.transform = MelSpectrogram(n_mels=64, n_fft=1024, hop_length=512)
        self._create_label_mapping()

    def _create_label_mapping(self):
        all_labels = set()
        for item in self._walker:
            # The label is always the parent directory name
            label = os.path.basename(os.path.dirname(item))
            all_labels.add(label)
        self.labels = sorted(list(all_labels))
        self.label_to_index = {label: index for index, label in enumerate(self.labels)}

    def __getitem__(self, index):
        waveform, sample_rate, label, _, _ = super().__getitem__(index)
        mel_spectrogram = self.transform(waveform)
        label_index = self.label_to_index[label]
        return mel_spectrogram, label_index

    def __len__(self):
        return super().__len__()

def get_model(model_name, num_classes):
    if model_name == 'resnet18':
        model = resnet18(pretrained=False)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=False)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return running_loss / len(test_loader), accuracy, precision, recall, f1

def main():
    args = get_args()
    set_seed(args.seed)
    
    # Set the backend to wav explicitly
    torchaudio.set_audio_backend("sox_io")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure the data directory exists
    os.makedirs("./data", exist_ok=True)
    
    train_dataset = AudioDataset("training")
    test_dataset = AudioDataset("testing")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = get_model(args.model, num_classes=len(train_dataset.labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, accuracy, precision, recall, f1 = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print()

if __name__ == "__main__":
    main()
