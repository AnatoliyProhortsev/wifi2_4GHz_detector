import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

import threading
import queue

class WiFiDetectorModel(tf.keras.Model):
    def __init__(self, input_shape=(1024, 2)):
        super().__init__()
        self.model = self.build_model(input_shape)
        self.status = "Idle"
        self.stop_training = False
        self.history = None
        
    def build_model(self, input_shape):
        model = models.Sequential(name="WiFi_Detector_2.4GHz")

        # Входной слой (принимает I/Q-данные: 1024 отсчёта, 2 канала)
        model.add(layers.InputLayer(input_shape=(1024, 2)))

        # Блок 1: CNN для извлечения признаков
        model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(2))

        # Блок 2: Углубление признаков
        model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling1D(2))

        # Блок 3: Дополнительные слои
        model.add(layers.Conv1D(128, 3, activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.GlobalAveragePooling1D())

        # Полносвязные слои для регрессии
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation='relu'))

        # Выходной слой: 2 нейрона (центральная частота и ширина канала)
        model.add(layers.Dense(2, activation='linear'))

        # Компиляция модели
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='mse',  # Mean Squared Error для регрессии
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                tf.keras.metrics.RootMeanSquaredError(name='rmse')
            ]
        )
        return model
    
    def load_data(self, file_path):
        """Загрузка и подготовка данных из HDF5-файла"""
        with h5py.File(file_path, 'r') as f:
            signals = []
            labels = []
            
            # Обход всех групп в файле
            for group in ['train', 'val', 'test']:
                grp = f['processed'][group]
                num_datasets = len([k for k in grp.keys() if 'signals' in k])
                
                # Сбор всех сигналов и меток
                for i in range(num_datasets):
                    signals.append(grp[f'signals_{i}'][:])
                    labels.append(grp[f'labels_{i}'][:])
            
            # Объединение данных
            X = np.concatenate(signals)
            y = np.concatenate(labels)
            
            # Преобразование комплексных чисел в 2 канала
            X = np.stack([X.real, X.imag], axis=-1)
            
            # Нормализация
            self.x_mean, self.x_std = np.mean(X), np.std(X)
            X = (X - self.x_mean) / self.x_std
            
            # Разделение на train/val
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            return (X_train, y_train), (X_val, y_val)
        
    def train(self, dataset_path, epochs=100, batch_size=64):
        self.status = "Training"
        
        try:
            # 1. Загрузка данных
            (X_train, y_train), (X_val, y_val) = self.load_data(dataset_path)
            
            # 2. Создание колбэков
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ModelCheckpoint('best_model.h5', save_best_only=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
            ]
            
            # 3. Обучение модели
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # 4. Сохранение финальной модели
            self.model.save("wifi_model.h5")
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
        
        self.status = "Idle"
        
    def realtime_analysis(self, input_queue):
        self.status = "Analyzing"
        while True:
            try:
                signal = input_queue.get(timeout=0.1)
                pred = self.predict(signal)
                input_queue.task_done()
            except queue.Empty:
                if self.stop_training:
                    break
        self.status = "Idle"