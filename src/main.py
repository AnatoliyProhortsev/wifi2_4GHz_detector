# main.py
import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
from tkinter import messagebox
import zlib
import numpy as np
import h5py
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Импорт модулей из предыдущих скриптов
from signal_generator import SignalGenerator
from data_parser import DataParser
from wifi_detector_model import WiFiDetectorModel

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wi-Fi Signal Analyzer 2.4GHz")
        self.geometry("1400x900")
        
        # Инициализация компонентов
        self.init_queues()
        self.init_components()
        self.init_gui()
        self.bind_events()
        
    def init_queues(self):
        """Инициализация очередей для межпоточного взаимодействия"""
        self.generator_queue = queue.Queue(maxsize=100)
        self.detection_queue = queue.Queue(maxsize=50)
        self.log_queue = queue.Queue()
        
    def init_components(self):
        """Инициализация основных компонентов системы"""
        # Генератор сигналов
        self.generator = SignalGenerator(log_queue=self.log_queue)
        
        # Парсер данных
        self.parser = DataParser(
            input_queue=self.generator_queue,
            output_queue=self.detection_queue,
            window_size=1024,
            log_queue=self.log_queue,
            dataset_mode=(self.mode_var.get() == "dataset")
        )
        
        # Нейросетевая модель
        self.model = WiFiDetectorModel(log_queue=self.log_queue)
        self.training_params = {
            'epochs': 100,
            'batch_size': 64,
            'dataset_path': 'dataset.h5'
        }
        
    def update_params(self, param, value):
        self.generator.update_params({param: value})
        
    def init_gui(self):
        """Инициализация графического интерфейса"""
        self.init_control_panel()
        self.init_visualization()
        self.init_status_bar()
        self.init_log_panel()
        
    def init_control_panel(self):
        """Панель управления"""
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Кнопки управления
        self.start_btn = ttk.Button(control_frame, text="Start", command=self.start)
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        
        # Выбор режима
        self.mode_var = tk.StringVar(value="realtime")
        mode_frame = ttk.LabelFrame(control_frame, text="Operation Mode")
        ttk.Radiobutton(mode_frame, text="Real-Time", variable=self.mode_var, 
                       value="realtime").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Training", variable=self.mode_var,
                       value="training").pack(side=tk.LEFT)
        
        # Параметры обучения
        self.epochs_var = tk.IntVar(value=100)
        ttk.Label(mode_frame, text="Epochs:").pack(side=tk.LEFT)
        ttk.Entry(mode_frame, textvariable=self.epochs_var, width=5).pack(side=tk.LEFT)
        
        # Расположение элементов
        mode_frame.pack(side=tk.LEFT, padx=10)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
    def init_visualization(self):
        """Область визуализации"""
        vis_frame = ttk.Frame(self)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Основной график
        self.main_figure = Figure(figsize=(10, 4))
        self.main_ax = self.main_figure.add_subplot(111)
        self.main_canvas = FigureCanvasTkAgg(self.main_figure, master=vis_frame)
        self.main_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Дополнительный график для обучения
        self.train_figure = Figure(figsize=(10, 3))
        self.train_ax = self.train_figure.add_subplot(111)
        self.train_canvas = FigureCanvasTkAgg(self.train_figure, master=vis_frame)
        
    def init_status_bar(self):
        """Строка состояния"""
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self, textvariable=self.status_var)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def init_log_panel(self):
        """Панель логов"""
        log_frame = ttk.Frame(self)
        log_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, width=40, height=20)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def bind_events(self):
        """Привязка обработчиков событий"""
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def start(self):
        """Запуск системы"""
        mode = self.mode_var.get()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Запуск потоков
        self.generator_thread = threading.Thread(
            target=self.generator.run,
            kwargs={'output_queue': self.generator_queue},
            daemon=True
        )
        
        self.parser_thread = threading.Thread(
            target=self.parser.run,
            daemon=True
        )
        
        self.nn_thread = threading.Thread(
            target=self.run_neural_network,
            daemon=True
        )
        
        self.generator_thread.start()
        self.parser_thread.start()
        self.nn_thread.start()
        
        # Запуск обновления GUI
        self.update_gui()
        
    def stop(self):
        """Остановка системы"""
        self.generator.stop()
        self.parser.stop()
        self.model.stop_training = True
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
    def run_neural_network(self):
        """Запуск нейросетевого компонента"""
        if self.mode_var.get() == "training":
            self.training_params['epochs'] = self.epochs_var.get()
            self.model.train(**self.training_params)
            self.show_training_results()
        else:
            self.model.realtime_analysis(self.detection_queue)
            
    def update_gui(self):
        """Обновление графического интерфейса"""
        # Обновление статусов
        status_info = [
            f"Generator: {self.generator.status}",
            f"Parser: {self.parser.status}",
            f"Model: {self.model.status}"
        ]
        self.status_var.set(" | ".join(status_info))
        
        # Обновление графиков
        self.update_main_plot()
        self.update_logs()
        
        # Периодический вызов
        self.after(100, self.update_gui)
        
    def update_main_plot(self):
        """Обновление основного графика"""
        if self.mode_var.get() == "realtime":
            self.update_realtime_plot()
        else:
            self.update_training_plot()
            
    def plot_signal(self):
        compressed = self.generator_queue.get()
        signal = np.frombuffer(zlib.decompress(compressed), dtype=np.complex64)
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(np.real(signal[:500]), label='Real')
        ax.plot(np.imag(signal[:500]), label='Imag')
        ax.legend()
        self.canvas.draw()
            
    def update_realtime_plot(self):
        """Обновление графика в реальном времени"""
        if not self.detection_queue.empty():
            signal, pred = self.detection_queue.get()
            self.main_ax.clear()
            
            # Отображение сигнала
            self.main_ax.plot(signal.real, label='Real Part', alpha=0.7)
            self.main_ax.plot(signal.imag, label='Imag Part', alpha=0.7)
            
            # Отображение предсказания
            if pred[1] > 0.8:  # Порог обнаружения
                self.main_ax.axvspan(
                    np.argmax(signal.real)-50, 
                    np.argmax(signal.real)+50, 
                    color='red', alpha=0.3
                )
                
            self.main_ax.legend()
            self.main_canvas.draw()
            
    def update_training_plot(self):
        """Обновление графиков обучения"""
        if self.model.history:
            self.train_ax.clear()
            
            # График потерь
            self.train_ax.plot(self.model.history.history['loss'], label='Train Loss')
            self.train_ax.plot(self.model.history.history['val_loss'], label='Val Loss')
            self.train_ax.set_title('Training Progress')
            self.train_ax.legend()
            
            self.train_canvas.draw()
            
    def update_logs(self):
        """Обновление логов"""
        while not self.log_queue.empty():
            log_entry = self.log_queue.get()
            self.log_text.insert(tk.END, log_entry + "\n")
            self.log_text.see(tk.END)
            
    def show_training_results(self):
        """Показать результаты обучения"""
        best_loss = min(self.model.history.history['val_loss'])
        messagebox.showinfo(
            "Training Complete",
            f"Best validation loss: {best_loss:.4f}\n"
            f"Final model saved to: wifi_model.h5"
        )
        
    def on_close(self):
        """Обработчик закрытия окна"""
        self.stop()
        self.destroy()

if __name__ == "__main__":
    app = Application()
    app.mainloop()