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


