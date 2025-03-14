import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import tensorflow as tf
from tensorflow.keras import layers, Model
import threading
import queue
import time

class SpectrumAnalyzer:
    def __init__(self):
        self.fs = 5e9
        self.duration = 1e-6
        self.n_samples = 4096
        self.freq_range = (2.4e9, 2.5e9)
        self.fixed_size = 500  # Фиксированный размер спектрограммы

    def generate_spectrum(self, add_wifi=False):
        t = np.linspace(0, self.duration, self.n_samples)
        signal = np.random.normal(0, 0.2, self.n_samples)
        
        if add_wifi:
            channel = np.random.choice([1, 6, 11])
            wifi_freq = 2.412e9 + (channel-1)*0.005e9
            signal += 0.8 * np.sin(2 * np.pi * wifi_freq * t)
        
        yf = fft(signal)
        xf = fftfreq(self.n_samples, 1/self.fs)
        
        mask = (xf >= self.freq_range[0]) & (xf <= self.freq_range[1])
        xf_filtered = xf[mask]
        yf_filtered = np.abs(yf[mask])**2
        
        
        
        # Защита от пустых данных
        if len(xf_filtered) < 2:
            xf_fixed = np.linspace(self.freq_range[0], self.freq_range[1], self.fixed_size)
            yf_fixed = np.full(self.fixed_size, 1e-12)  # Заполняем минимальным значением
        else:
            # Интерполяция и защита от отрицательных значений
            interp_func = interp1d(xf_filtered, yf_filtered, kind='linear', fill_value="extrapolate")
            xf_fixed = np.linspace(self.freq_range[0], self.freq_range[1], self.fixed_size)
            yf_fixed = interp_func(xf_fixed)
            yf_fixed = np.clip(yf_fixed, a_min=1e-12, a_max=None)  # Обрезаем отрицательные значения
        
        return xf_fixed, 10 * np.log10(yf_fixed + 1e-12)

class Autoencoder(Model):
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Reshape([500, 1]),
            layers.Conv1D(32, 5, activation='relu', padding='same'),
            layers.MaxPooling1D(2),
            layers.Conv1D(16, 5, activation='relu', padding='same')
        ])
        
        self.decoder = tf.keras.Sequential([
            layers.Conv1DTranspose(16, 5, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            layers.Conv1DTranspose(32, 5, activation='relu', padding='same'),
            layers.Conv1D(1, 3, activation='sigmoid', padding='same'),
            layers.Reshape([500])
        ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WiFi Spectrum Detector")
        self.analyzer = SpectrumAnalyzer()
        self.model = Autoencoder()
        self.data_queue = queue.Queue(maxsize=10)
        self.running = False
        self.training = False
        self.threshold = 0.15
        
        self.init_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.model.compile(optimizer='adam', loss='mse')
        
    def init_ui(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(pady=10)
        
        self.train_btn = ttk.Button(
            control_frame, 
            text="Обучить модель", 
            command=self.start_training
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(
            control_frame, 
            text="Пуск детектора", 
            command=self.toggle_scan
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack()
        
        self.ax.set_xlim(2.4e9, 2.5e9)
        self.ax.set_ylim(-100, 20)
        self.ax.set_xlabel('Frequency [GHz]')
        self.ax.set_ylabel('Power [dBm]')
        self.ax.grid(True)
        self.ax.xaxis.set_major_formatter(lambda x, _: f'{x/1e9:.3f}')
        self.line, = self.ax.plot([], [], lw=1, color='blue')
        self.anomaly_patch = self.ax.axvspan(0, 0, facecolor='red', alpha=0.3)
        
    def start_training(self):
        if not self.training:
            self.training = True
            self.train_btn.config(text="Обучение...")
            threading.Thread(target=self.train_model).start()
            
    def train_model(self):
        X_train = np.array([self.analyzer.generate_spectrum()[1] for _ in range(1000)])
        X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min() + 1e-12)
        self.model.fit(X_train, X_train, epochs=10, batch_size=32, verbose=1)
        self.training = False
        self.train_btn.config(text="Обучить модель")
        
    def toggle_scan(self):
        if not self.running:
            self.start_scan()
        else:
            self.stop_scan()
            
    def start_scan(self):
        self.running = True
        self.start_btn.config(text="Стоп детектор")
        threading.Thread(target=self.continuous_scan).start()
        self.after(100, self.update_plot)
            
    def stop_scan(self):
        self.running = False
        self.start_btn.config(text="Пуск детектора")
        
    def continuous_scan(self):
        while self.running:
            try:
                add_wifi = np.random.rand() > 0.5
                xf, yf = self.analyzer.generate_spectrum(add_wifi)
                self.data_queue.put((xf, yf, add_wifi))
                time.sleep(0.1)
            except:
                break
    
    def detect_anomaly(self, signal):
        signal_norm = (signal - signal.mean()) / signal.std()
        reconstructed = self.model.predict(signal_norm[np.newaxis, :], verbose=0)
        return np.mean(np.square(signal_norm - reconstructed)) > self.threshold
    
    def update_plot(self):
        if not self.running:
            return
            
        try:
            xf, yf, true_label = self.data_queue.get_nowait()
            
            # Обновление графика
            self.line.set_data(xf, yf)
            
            # Детекция аномалий
            anomaly = self.detect_anomaly(yf)
            self.anomaly_patch.set_visible(anomaly)
            title = f"WiFi Detected: {anomaly} (True: {true_label})"
            self.ax.set_title(title)
            
            # Автомасштабирование
            y_min = np.min(yf) - 5 if len(yf) > 0 else -100
            y_max = np.max(yf) + 5 if len(yf) > 0 else 20
            self.ax.set_ylim(y_min, y_max)
            
            self.canvas.draw_idle()
            
        except queue.Empty:
            pass
        
        self.after(100, self.update_plot)
    
    def on_close(self):
        self.stop_scan()
        self.destroy()

if __name__ == "__main__":
    app = Application()
    app.mainloop()