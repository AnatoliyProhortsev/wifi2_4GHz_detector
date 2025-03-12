# SignalGenerator.py
import numpy as np
import zlib
import threading
import queue
import time
import h5py

class SignalGenerator:
    def __init__(self, sample_rate=100e6, log_queue=None):
        self.running = False
        self.status = "Stopped"
        self.sample_rate = sample_rate
        self.log_queue = log_queue or queue.Queue()
        self.current_params = {
            'base_freq': 2.4e9,
            'channel_width': 20e6,
            'snr_db': 20,
            'freq_offset': 5e3,
            'distortion_type': 'none'
        }
        self.distortion_types = ['multipath', 'nonlinear', 'phase_noise']

    def log(self, message):
        self.log_queue.put(f"[Generator] {message}")

    def generate_ofdm_signal(self):
        """Генерация базового OFDM сигнала"""
        t = np.arange(int(self.sample_rate)) / self.sample_rate  # Создаем временную ось
        num_subcarriers = 64
        data = np.random.randn(num_subcarriers) + 1j*np.random.randn(num_subcarriers)
        return np.fft.ifft(data), t  # Возвращаем и сигнал, и временную ось

    def apply_distortions(self, signal):
        distortion = self.current_params['distortion_type']
        if distortion == 'multipath':
            delayed = np.roll(signal, 100) * 0.5
            return signal + delayed
        elif distortion == 'nonlinear':
            return np.tanh(1.5 * signal.real) + 1j * np.tanh(1.5 * signal.imag)
        elif distortion == 'phase_noise':
            phase_noise = 0.1 * np.random.randn(len(signal))
            return signal * np.exp(1j * phase_noise)
        return signal

    def add_noise(self, signal):
        snr_linear = 10**(self.current_params['snr_db'] / 10)
        power = np.mean(np.abs(signal)**2)
        noise_power = power / snr_linear
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
        return signal + noise

    def generate(self):
        try:
            ofdm_signal, t = self.generate_ofdm_signal()  # Получаем временную ось
            signal = self.apply_distortions(ofdm_signal)
            signal = self.add_noise(signal)
            signal *= np.exp(2j*np.pi*self.current_params['freq_offset']*t)
            return zlib.compress(signal.tobytes())
        except Exception as e:
            self.log(f"Error: {str(e)}")
            return None

    def run(self, output_queue, mode='realtime'):
        self.running = True
        self.status = "Running"
        
        if mode == 'dataset':
            with h5py.File("dataset.h5", "w") as f:
                for i in range(1000):
                    if not self.running: 
                        break
                    compressed = self.generate()
                    f.create_dataset(f"signal_{i}", data=compressed)
        else:
            while self.running:
                compressed = self.generate()
                if compressed:
                    try: 
                        output_queue.put(compressed, timeout=0.1)
                    except queue.Full: 
                        self.log("Queue full")
                time.sleep(0.1)
        
        self.status = "Stopped"

    def update_params(self, params):
        self.current_params.update(params)
        self.log(f"Params updated: {params}")

    def stop(self):
        self.running = False