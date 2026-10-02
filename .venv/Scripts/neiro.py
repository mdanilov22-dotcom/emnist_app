import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import os


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание EMNIST")
        self.root.geometry("800x600")

        # Загрузка модели
        self.load_model()

        if self.model is None:
            return

        # Создание mapping классов
        self.create_class_mapping()

        # Создание интерфейса
        self.create_widgets()

        # Переменные для рисования
        self.last_x = None
        self.last_y = None
        self.image_data = np.ones((280, 280), dtype=np.float32)

    def load_model(self):
        """Загрузка модели"""
        model_path = 'emnist_byclass_improved.h5'
        if not os.path.exists(model_path):
            print(f"Модель не найдена: {model_path}")
            print("Поместите файл модели в папку с программой")
            self.model = None
        else:
            try:
                self.model = tf.keras.models.load_model(model_path)
                print("Модель загружена успешно!")
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                self.model = None

    def create_class_mapping(self):
        """Создание маппинга классов"""
        self.class_mapping = {}

        # Цифры 0-9
        for i in range(10):
            self.class_mapping[i] = str(i)

        # Прописные буквы A-Z
        for i, letter in enumerate(range(ord('A'), ord('Z') + 1)):
            self.class_mapping[10 + i] = chr(letter)

        # Строчные буквы a-z
        for i, letter in enumerate(range(ord('a'), ord('z') + 1)):
            self.class_mapping[36 + i] = chr(letter)

    def create_widgets(self):
        """Создание виджетов интерфейса"""
        # Основной фрейм
        main_frame = tk.Frame(self.root, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Левая панель - рисование
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas для рисования
        self.canvas = tk.Canvas(left_frame, width=280, height=280, bg='white', cursor="cross")
        self.canvas.pack(pady=(0, 10))
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.start_pos)

        # Кнопки
        btn_frame = tk.Frame(left_frame)
        btn_frame.pack()

        tk.Button(btn_frame, text="Распознать", command=self.recognize,
                  width=15, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Очистить", command=self.clear,
                  width=15, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)

        # Правая панель - результаты
        right_frame = tk.Frame(main_frame, padx=20)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Результат
        tk.Label(right_frame, text="Результат:", font=("Arial", 16, "bold")).pack(anchor=tk.W)

        self.result_label = tk.Label(right_frame, text="Нарисуйте символ",
                                     font=("Arial", 32, "bold"), fg="#2196F3")
        self.result_label.pack(pady=(10, 5))

        self.confidence_label = tk.Label(right_frame, text="", font=("Arial", 12))
        self.confidence_label.pack()

        # Предпросмотр обработанного изображения
        tk.Label(right_frame, text="Предпросмотр:", font=("Arial", 12)).pack(anchor=tk.W, pady=(20, 5))
        self.preview_canvas = tk.Canvas(right_frame, width=56, height=56, bg='white', bd=1, relief=tk.SUNKEN)
        self.preview_canvas.pack()

        # Топ-3 предсказания
        self.top_frame = tk.Frame(right_frame)
        self.top_frame.pack(pady=(20, 0), fill=tk.X)

        tk.Label(self.top_frame, text="Топ-3 предсказания:", font=("Arial", 12, "bold")).pack(anchor=tk.W)

        # Инструкция
        tk.Label(right_frame, text="\nИнструкция:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 0))
        tk.Label(right_frame,
                 text="1. Нарисуйте символ в белом поле\n2. Нажмите 'Распознать'\n3. Для очистки нажмите 'Очистить'",
                 font=("Arial", 10), justify=tk.LEFT).pack(anchor=tk.W)

    def start_pos(self, event):
        """Начало рисования"""
        self.last_x, self.last_y = event.x, event.y

    def paint(self, event):
        """Рисование на canvas"""
        if self.last_x and self.last_y:
            # Рисование на canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=20, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)

            # Обновление image_data
            x1, y1 = int(self.last_x), int(self.last_y)
            x2, y2 = int(event.x), int(event.y)

            for x in range(max(0, min(x1, x2) - 5), min(280, max(x1, x2) + 5)):
                for y in range(max(0, min(y1, y2) - 5), min(280, max(y1, y2) + 5)):
                    if (x - x1) ** 2 + (y - y1) ** 2 < 100 or (x - x2) ** 2 + (y - y2) ** 2 < 100:
                        self.image_data[y, x] = 0.0

            self.last_x, self.last_y = event.x, event.y

    def clear(self):
        """Очистка canvas"""
        self.canvas.delete("all")
        self.result_label.config(text="Нарисуйте символ", fg="#2196F3")
        self.confidence_label.config(text="")
        self.image_data = np.ones((280, 280), dtype=np.float32)
        self.preview_canvas.delete("all")
        self.clear_top_predictions()

    def clear_top_predictions(self):
        """Очистка топ предсказаний"""
        for widget in self.top_frame.winfo_children():
            if widget.winfo_class() == 'Frame':
                widget.destroy()

    def recognize(self):
        """Распознавание символа"""
        try:
            # Преобразование 280x280 в 28x28
            img_small = np.zeros((28, 28))
            for i in range(28):
                for j in range(28):
                    patch = self.image_data[i * 10:(i + 1) * 10, j * 10:(j + 1) * 10]
                    img_small[i, j] = 1.0 - np.mean(patch)

            # Поворот и зеркаливание (как в EMNIST)
            img_small = np.rot90(img_small, k=-1)
            img_small = np.fliplr(img_small)

            # Показ обработанного изображения
            self.show_preview(img_small)

            # Подготовка для модели
            img_array = img_small.reshape(1, 28, 28, 1)

            # Предсказание
            prediction = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = np.max(prediction[0])

            # Получение топ-3 предсказаний
            top_n = 3
            top_indices = np.argsort(prediction[0])[-top_n:][::-1]
            top_confidences = prediction[0][top_indices]

            # Отображение результата
            symbol = self.class_mapping[predicted_class]
            self.result_label.config(text=f"'{symbol}'")
            self.confidence_label.config(text=f"Уверенность: {confidence:.2%}")

            # Цвет результата
            if confidence > 0.8:
                self.result_label.config(fg="#4CAF50")  # зеленый
            elif confidence > 0.5:
                self.result_label.config(fg="#FF9800")  # оранжевый
            else:
                self.result_label.config(fg="#f44336")  # красный

            # Показ топ-3 предсказаний
            self.show_top_predictions(top_indices, top_confidences)

        except Exception as e:
            print(f"Ошибка распознавания: {e}")
            self.result_label.config(text="Ошибка!", fg="#f44336")

    def show_preview(self, img_array):
        """Показ обработанного изображения"""
        self.preview_canvas.delete("all")

        # Нормализуем для отображения
        img_normalized = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-7)
        img_normalized = (img_normalized * 255).astype(np.uint8)

        # Создаем изображение
        img = Image.fromarray(img_normalized, mode='L')
        img = img.resize((56, 56), Image.Resampling.NEAREST)

        # Конвертируем для tkinter
        photo = ImageTk.PhotoImage(img)
        self.preview_canvas.create_image(28, 28, image=photo)
        self.preview_canvas.image = photo  # Сохраняем ссылку

    def show_top_predictions(self, indices, confidences):
        """Показ топ-3 предсказаний"""
        self.clear_top_predictions()

        for idx, conf in zip(indices, confidences):
            symbol = self.class_mapping[idx]

            frame = tk.Frame(self.top_frame)
            frame.pack(fill=tk.X, pady=2)

            # Символ
            tk.Label(frame, text=f"'{symbol}'", width=5, font=("Arial", 12)).pack(side=tk.LEFT)

            # Прогресс-бар (упрощенный)
            canvas = tk.Canvas(frame, width=100, height=20, bg='#e0e0e0', highlightthickness=0)
            canvas.pack(side=tk.LEFT, padx=5)

            # Рисуем прогресс
            width = int(conf * 100)
            canvas.create_rectangle(0, 0, width, 20, fill='#2196F3', outline='')
            canvas.create_text(50, 10, text=f"{conf:.1%}", font=("Arial", 9))

            # Процент
            tk.Label(frame, text=f"{conf:.2%}", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()