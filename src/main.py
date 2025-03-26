import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import random
import os
from sklearn.metrics import classification_report

# Параметры спектрограммы
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNEL_CENTER_BIN = {1: 12, 2: 26, 3: 40, 4: 54}

def bin_to_freq(bin_index):
    return 2.4 + (bin_index / (IMG_HEIGHT - 1)) * 0.1

def generate_spectrum(with_wifi=False, wifi_channel=None):
    spectrum = np.random.normal(loc=0.2, scale=0.05, size=(IMG_HEIGHT, IMG_WIDTH))
    if with_wifi and wifi_channel in CHANNEL_CENTER_BIN:
        center = CHANNEL_CENTER_BIN[wifi_channel]
        sigma = 5.0
        freqs = np.arange(IMG_HEIGHT)
        gauss = np.exp(-((freqs - center) ** 2) / (2 * sigma ** 2))

        time_mask = np.zeros(IMG_WIDTH)
        num_bands = random.randint(1, 3)
        for _ in range(num_bands):
            start = random.randint(0, IMG_WIDTH - 10)
            width = random.randint(5, 20)
            time_mask[start:start + width] = 1.0

        signal = np.outer(gauss, time_mask) * 1.0
        spectrum += signal

        noise = np.random.normal(loc=0.0, scale=0.1, size=(IMG_HEIGHT, IMG_WIDTH))
        spectrum += signal * noise

    spectrum = np.clip(spectrum, 0, 1)
    return spectrum.astype(np.float32)


def generate_random_sample(prob_wifi=0.5):
    if random.random() < prob_wifi:
        channel = random.choice(list(CHANNEL_CENTER_BIN.keys()))
        spectrum = generate_spectrum(with_wifi=True, wifi_channel=channel)
        label = channel
    else:
        spectrum = generate_spectrum(with_wifi=False)
        label = 0
    return spectrum, label


def create_dataset(n_samples=1000):
    data = np.zeros((n_samples, 1, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        spectrum, label = generate_random_sample()
        data[i, 0] = spectrum
        labels[i] = label
    return data, labels

class WifiDetector(nn.Module):
    def __init__(self):
        super(WifiDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16*32*32, 128),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 16*32*32),
            nn.Unflatten(1, (16, 32, 32)),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        logits = self.classifier(features)
        return reconstruction, logits

def train_model(model, data, labels, epochs=50, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    X = torch.from_numpy(data)
    y = torch.from_numpy(labels)
    
    n_samples = X.shape[0]
    metrics = {
        'total_loss': [],
        'recon_loss': [],
        'class_loss': [],
        'accuracy': [],
        'all_preds': [],
        'all_labels': []
    }
    
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n_samples)
        epoch_loss = epoch_recon = epoch_class = total_correct = 0
        preds = []
        true_labels = []
        
        for i in range(0, n_samples, batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[indices].to(device), y[indices].to(device)
            
            optimizer.zero_grad()
            recon, logits = model(batch_x)
            
            loss_recon = mse_loss(recon, batch_x)
            loss_class = ce_loss(logits, batch_y)
            loss = loss_recon + 10 * loss_class
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            epoch_recon += loss_recon.item() * batch_x.size(0)
            epoch_class += loss_class.item() * batch_x.size(0)
            
            batch_preds = torch.argmax(logits, dim=1)
            total_correct += (batch_preds == batch_y).sum().item()
            preds.extend(batch_preds.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
        
        metrics['total_loss'].append(epoch_loss / n_samples)
        metrics['recon_loss'].append(epoch_recon / n_samples)
        metrics['class_loss'].append(epoch_class / n_samples)
        metrics['accuracy'].append(total_correct / n_samples)
        metrics['all_preds'].extend(preds)
        metrics['all_labels'].extend(true_labels)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Loss: {metrics['total_loss'][-1]:.4f} (Recon: {metrics['recon_loss'][-1]:.4f}, Class: {metrics['class_loss'][-1]:.4f})")
        print(f"  Accuracy: {metrics['accuracy'][-1]:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(metrics['all_labels'], metrics['all_preds'], target_names=['No WiFi']+[f'Ch {i}' for i in range(1,5)]))
    
    return model, metrics

