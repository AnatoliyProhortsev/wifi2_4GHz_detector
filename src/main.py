import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Функции для генерации синтетических сигналов

def generate_wifi_signal(length):
    """
    Генерация синтетического Wi‑Fi сигнала.
    Здесь используется равномерное распределение RSSI от -100 до -30 dBm.
    """
    signal = np.random.uniform(-100, -30, size=length)
    return signal

def generate_noise_signal(length):
    """
    Генерация шумового (не Wi‑Fi) сигнала.
    Здесь выбирается диапазон значений, характерный для фонового шума.
    """
    signal = np.random.uniform(-100, -95, size=length)
    return signal

def compute_fixed_spectrogram(signal, fs=1, nperseg=32):
    """
    Вычисление спектрограммы с фиксированными параметрами.
    Добавляем небольшую константу, чтобы избежать log(0).
    """
    frequencies, times, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg)
    Sxx_dB = 10 * np.log10(Sxx + 1e-8)
    return Sxx_dB

# Параметры генерации
num_samples_per_class = 100  # число сэмплов для каждого класса
time_series_length = 128     # длина временного ряда (количество отсчетов)

wifi_signals = []
noise_signals = []

# Генерация сигналов для обоих классов
for _ in range(num_samples_per_class):
    wifi_signals.append(generate_wifi_signal(time_series_length))
    noise_signals.append(generate_noise_signal(time_series_length))

# Вычисление спектрограмм для каждого сэмпла
X = []
y = []

for signal in wifi_signals:
    spec = compute_fixed_spectrogram(signal, fs=1, nperseg=32)
    X.append(spec)
    y.append(1)  # метка 1 для Wi‑Fi сигнала

for signal in noise_signals:
    spec = compute_fixed_spectrogram(signal, fs=1, nperseg=32)
    X.append(spec)
    y.append(0)  # метка 0 для не Wi‑Fi сигнала

X = np.array(X)
y = np.array(y)

# Добавляем измерение канала для CNN (требуется форма: (samples, height, width, channels))
X = X[..., np.newaxis]

# Разбиваем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация изображений спектрограмм (масштабирование к диапазону [0, 1])
min_val = X_train.min()
max_val = X_train.max()
X_train_norm = (X_train - min_val) / (max_val - min_val)
X_test_norm = (X_test - min_val) / (max_val - min_val)

# Определяем модель с помощью функционального API
inputs = tf.keras.Input(shape=X_train_norm.shape[1:])
x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Обучение модели
history = model.fit(X_train_norm, y_train, epochs=10, batch_size=16, validation_data=(X_test_norm, y_test))

# Оценка модели на тестовой выборке
loss, accuracy = model.evaluate(X_test_norm, y_test)
print("Test accuracy:", accuracy)

# Пример предсказания для нескольких тестовых сэмплов
predictions = model.predict(X_test_norm)
print("Примеры предсказаний (вероятность того, что сигнал Wi‑Fi):", predictions[:5].flatten())

# Визуализация нескольких спектрограмм с предсказаниями
n_to_plot = 4
plt.figure(figsize=(12, 8))
for i in range(n_to_plot):
    plt.subplot(2, 2, i+1)
    plt.imshow(X_test_norm[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"Label: {y_test[i]}, Pred: {predictions[i][0]:.2f}")
    plt.colorbar(label='Нормированная мощность')
plt.tight_layout()
plt.show()
