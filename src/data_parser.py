# parser.py
import numpy as np
import zlib
import threading
import queue
import time
import h5py
import logging
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

class DataParser:
    def __init__(self, input_queue, output_queue, window_size=1024, 
                 log_queue=None, dataset_mode=False):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.window_size = window_size
        self.log_queue = log_queue or queue.Queue()
        self.dataset_mode = dataset_mode
        self.running = False
        self.status = "Idle"
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Инициализация HDF5 файла для режима датасета
        self.h5_file = None
        self.dataset_group = None
        self.sample_counter = 0

    def log(self, message, level=logging.INFO):
        """Логирование сообщений в систему"""
        log_entry = f"[Parser] {level}: {message}"
        self.log_queue.put(log_entry)

    def init_dataset_storage(self):
        """Инициализация хранилища для режима датасета"""
        if self.dataset_mode:
            try:
                self.h5_file = h5py.File("training_dataset.h5", "w")
                self.dataset_group = self.h5_file.create_group("processed")
                self.log("HDF5 storage initialized")
            except Exception as e:
                self.log(f"Failed to initialize HDF5: {str(e)}", logging.ERROR)

    def process_signal(self, raw_signal):
        """Основной конвейер обработки сигнала"""
        try:
            # 1. Декомпрессия данных
            signal = self.decompress_signal(raw_signal)
            
            # 2. Сегментация на окна
            segments = self.segment_signal(signal)
            
            # 3. Нормализация
            normalized = self.normalize(segments)
            
            # 4. Преобразование в спектрограмму
            spectrograms = self.create_spectrograms(normalized)
            
            return spectrograms
            
        except Exception as e:
            self.log(f"Processing failed: {str(e)}", logging.ERROR)
            return None

    def decompress_signal(self, raw_data):
        """Распаковка сжатых данных"""
        try:
            return np.frombuffer(zlib.decompress(raw_data), dtype=np.complex64)
        except zlib.error as e:
            self.log(f"Decompression error: {str(e)}", logging.ERROR)
            return None

    def segment_signal(self, signal):
        """Сегментация сигнала на перекрывающиеся окна"""
        segments = []
        for i in range(0, len(signal)-self.window_size, self.window_size//2):
            segment = signal[i:i+self.window_size]
            segments.append(segment)
        return np.array(segments)

    def normalize(self, segments):
        """Нормализация данных"""
        reshaped = segments.reshape(-1, 1)
        self.scaler.partial_fit(reshaped)
        return self.scaler.transform(reshaped).reshape(segments.shape)

    def create_spectrograms(self, segments):
        """Создание спектрограмм для каждого сегмента"""
        spectrograms = []
        for segment in segments:
            f, t, Sxx = signal.spectrogram(segment, fs=100e6)
            spectrograms.append(Sxx)
        return np.array(spectrograms)

    def save_to_dataset(self, data, labels):
        """Сохранение обработанных данных в HDF5"""
        if self.dataset_mode and self.h5_file:
            try:
                self.dataset_group.create_dataset(
                    f"sample_{self.sample_counter}",
                    data=data,
                    compression="gzip"
                )
                self.dataset_group.create_dataset(
                    f"label_{self.sample_counter}",
                    data=labels,
                    compression="gzip"
                )
                self.sample_counter += 1
            except Exception as e:
                self.log(f"Failed to save sample: {str(e)}", logging.ERROR)

    def run(self):
        """Основной цикл обработки"""
        self.running = True
        self.status = "Running"
        self.init_dataset_storage()
        self.log("Parser started")

        while self.running:
            try:
                # Получение сырых данных из генератора
                raw_data = self.input_queue.get(timeout=0.5)
                
                # Обработка сигнала
                processed_data = self.process_signal(raw_data)
                
                if processed_data is not None:
                    if self.dataset_mode:
                        # В режиме датасета: сохранение с метками
                        labels = self.extract_labels(raw_data)
                        self.save_to_dataset(processed_data, labels)
                    else:
                        # В реальном времени: отправка в нейросеть
                        self.output_queue.put(processed_data)
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.log(f"Unexpected error: {str(e)}", logging.CRITICAL)

        # Завершение работы
        if self.dataset_mode and self.h5_file:
            self.h5_file.close()
        self.status = "Idle"
        self.log("Parser stopped")

    def extract_labels(self, raw_data):
        """Извлечение меток из сырых данных (заглушка)"""
        # Реальная реализация будет зависеть от формата данных генератора
        return np.random.randint(0, 2, size=1)

    def stop(self):
        """Корректная остановка парсера"""
        self.running = False
        self.log("Stopping parser...")