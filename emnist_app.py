"""Распознавание рукописных символов EMNIST (ByClass, 62 класса).

Требования: tensorflow, numpy, Pillow >= 9.1
"""
from __future__ import annotations

import os
import string
import tkinter as tk
from tkinter import messagebox

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageTk

# ----------------------------- Константы -----------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "emnist_byclass_improved.h5")

CANVAS_SIZE = 280
BRUSH = 20
PREVIEW_SIZE = 112
AUTO_RECOGNIZE_MS = 400          # пауза после отпускания кнопки мыши
TOP_N = 3

CONF_HIGH = 0.8
CONF_MID = 0.5
COLOR_HIGH = "#4CAF50"
COLOR_MID = "#FF9800"
COLOR_LOW = "#f44336"
COLOR_IDLE = "#2196F3"

# Порядок классов EMNIST ByClass: 0-9, A-Z, a-z
CLASSES = string.digits + string.ascii_uppercase + string.ascii_lowercase

# Буквы, у которых прописная и строчная формы почти неразличимы
CASE_AMBIGUOUS = "CIJKLMOPSUVWXYZ"


# ------------------------- Логика распознавания ----------------------------
def preprocess(pil_img: Image.Image) -> np.ndarray | None:
    """Приводит рисунок к виду EMNIST: обрезка, масштаб до 20x20 с сохранением
    пропорций, центрирование по центру масс в поле 28x28.

    Возвращает массив (28, 28) float32 в диапазоне [0, 1] или None, если холст пуст.
    """
    arr = 255 - np.array(pil_img, dtype=np.uint8)      # штрихи = яркие
    ys, xs = np.where(arr > 20)
    if xs.size == 0:
        return None

    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    scale = 20 / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    crop = np.array(Image.fromarray(crop).resize((nw, nh), Image.Resampling.LANCZOS),
                    dtype=np.float32) / 255.0

    out = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    out[top:top + nh, left:left + nw] = crop

    # сдвиг по центру масс
    total = out.sum()
    if total <= 0:
        return None
    yy, xx = np.indices(out.shape)
    cy, cx = (yy * out).sum() / total, (xx * out).sum() / total
    out = np.roll(out, (round(13.5 - cy), round(13.5 - cx)), axis=(0, 1))

    # EMNIST хранится повёрнутым/зеркальным: это эквивалентно транспонированию
    return out.T


class Recognizer:
    """Обёртка над моделью: загрузка, прогрев, предсказание."""

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена:\n{model_path}")
        self.model = tf.keras.models.load_model(model_path)
        # прогревочный вызов, чтобы первое распознавание не «зависало»
        self.model(np.zeros((1, 28, 28, 1), np.float32), training=False)

    def predict(self, img28: np.ndarray, ignore_case: bool = False) -> np.ndarray:
        x = img28.reshape(1, 28, 28, 1).astype(np.float32)
        probs = self.model(x, training=False).numpy()[0].copy()
        if ignore_case:
            probs = self._merge_case(probs)
        return probs

    @staticmethod
    def _merge_case(probs: np.ndarray) -> np.ndarray:
        """Суммирует вероятности пар вроде 'c' + 'C' и отдаёт сумму прописной букве."""
        for ch in CASE_AMBIGUOUS:
            up, low = CLASSES.index(ch), CLASSES.index(ch.lower())
            probs[up] += probs[low]
            probs[low] = 0.0
        return probs


# ------------------------------- GUI ---------------------------------------
class DrawingApp:
    def __init__(self, root: tk.Tk, recognizer: Recognizer):
        self.root = root
        self.recognizer = recognizer
        self.root.title("Распознавание EMNIST")
        self.root.resizable(False, False)

        self.last_x: int | None = None
        self.last_y: int | None = None
        self._after_id: str | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None

        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 255)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self.ignore_case = tk.BooleanVar(value=False)

        self.create_widgets()

        self.root.bind("<Return>", lambda e: self.recognize())
        self.root.bind("<Escape>", lambda e: self.clear())

    # -- интерфейс --
    def create_widgets(self) -> None:
        main = tk.Frame(self.root, padx=20, pady=20)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH)

        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="white", cursor="cross", highlightthickness=1,
                                highlightbackground="#999")
        self.canvas.pack(pady=(0, 10))
        self.canvas.bind("<ButtonPress-1>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        btns = tk.Frame(left)
        btns.pack()
        tk.Button(btns, text="Распознать (Enter)", command=self.recognize,
                  width=18, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(btns, text="Очистить (Esc)", command=self.clear,
                  width=18, bg="#f44336", fg="white").pack(side=tk.LEFT, padx=5)

        tk.Checkbutton(left, text="Не различать регистр похожих букв (c/C, o/O, ...)",
                       variable=self.ignore_case,
                       command=self.recognize).pack(pady=(10, 0))

        right = tk.Frame(main, padx=20)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="Результат:", font=("Arial", 16, "bold")).pack(anchor=tk.W)
        self.result_label = tk.Label(right, text="Нарисуйте символ",
                                     font=("Arial", 28, "bold"), fg=COLOR_IDLE)
        self.result_label.pack(pady=(10, 5))
        self.confidence_label = tk.Label(right, text="", font=("Arial", 12))
        self.confidence_label.pack()

        tk.Label(right, text="Предпросмотр:", font=("Arial", 12)).pack(anchor=tk.W, pady=(20, 5))
        self.preview_canvas = tk.Canvas(right, width=PREVIEW_SIZE, height=PREVIEW_SIZE,
                                        bg="white", bd=1, relief=tk.SUNKEN)
        self.preview_canvas.pack(anchor=tk.W)

        tk.Label(right, text=f"Топ-{TOP_N} предсказания:",
                 font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 0))
        self.top_frame = tk.Frame(right)
        self.top_frame.pack(fill=tk.X)

    # -- рисование --
    def start_pos(self, event: tk.Event) -> None:
        self._cancel_auto()
        self.last_x, self.last_y = event.x, event.y
        self._draw_dot(event.x, event.y)       # одиночный клик тоже рисует точку

    def paint(self, event: tk.Event) -> None:
        if self.last_x is None or self.last_y is None:
            self.last_x, self.last_y = event.x, event.y
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                width=BRUSH, fill="black", capstyle=tk.ROUND)
        self.pil_draw.line((self.last_x, self.last_y, event.x, event.y),
                           fill=0, width=BRUSH)
        self._draw_pil_dot(event.x, event.y)
        self.last_x, self.last_y = event.x, event.y

    def end_stroke(self, _event: tk.Event) -> None:
        self.last_x = self.last_y = None
        self._cancel_auto()
        self._after_id = self.root.after(AUTO_RECOGNIZE_MS, self.recognize)

    def _draw_dot(self, x: int, y: int) -> None:
        r = BRUSH // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self._draw_pil_dot(x, y)

    def _draw_pil_dot(self, x: int, y: int) -> None:
        r = BRUSH // 2
        self.pil_draw.ellipse((x - r, y - r, x + r, y + r), fill=0)

    def _cancel_auto(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def clear(self) -> None:
        self._cancel_auto()
        self.canvas.delete("all")
        self.pil_image.paste(255, (0, 0, CANVAS_SIZE, CANVAS_SIZE))
        self.last_x = self.last_y = None
        self._reset_results()

    def _reset_results(self) -> None:
        self.result_label.config(text="Нарисуйте символ", fg=COLOR_IDLE)
        self.confidence_label.config(text="")
        self.preview_canvas.delete("all")
        self._clear_top()

    # -- распознавание --
    def recognize(self) -> None:
        self._cancel_auto()
        try:
            img28 = preprocess(self.pil_image)
            if img28 is None:
                self._reset_results()
                return

            self.show_preview(img28)
            probs = self.recognizer.predict(img28, self.ignore_case.get())

            top_idx = np.argsort(probs)[-TOP_N:][::-1]
            best = int(top_idx[0])
            confidence = float(probs[best])

            self.result_label.config(text=f"'{CLASSES[best]}'")
            self.confidence_label.config(text=f"Уверенность: {confidence:.2%}")
            if confidence > CONF_HIGH:
                color = COLOR_HIGH
            elif confidence > CONF_MID:
                color = COLOR_MID
            else:
                color = COLOR_LOW
            self.result_label.config(fg=color)

            self.show_top(top_idx, probs[top_idx])
        except Exception as e:  # noqa: BLE001
            print(f"Ошибка распознавания: {e}")
            self.result_label.config(text="Ошибка!", fg=COLOR_LOW)

    def show_preview(self, img: np.ndarray) -> None:
        self.preview_canvas.delete("all")
        norm = (img - img.min()) / (img.max() - img.min() + 1e-7)
        pil = Image.fromarray((norm * 255).astype(np.uint8), mode="L")
        pil = pil.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.Resampling.NEAREST)
        self._preview_photo = ImageTk.PhotoImage(pil)     # держим ссылку
        self.preview_canvas.create_image(PREVIEW_SIZE // 2, PREVIEW_SIZE // 2,
                                         image=self._preview_photo)

    def _clear_top(self) -> None:
        for w in self.top_frame.winfo_children():
            w.destroy()

    def show_top(self, indices: np.ndarray, confidences: np.ndarray) -> None:
        self._clear_top()
        bar_w, bar_h = 140, 20
        for idx, conf in zip(indices, confidences):
            row = tk.Frame(self.top_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"'{CLASSES[int(idx)]}'", width=5,
                     font=("Arial", 12)).pack(side=tk.LEFT)
            bar = tk.Canvas(row, width=bar_w, height=bar_h, bg="#e0e0e0",
                            highlightthickness=0)
            bar.pack(side=tk.LEFT, padx=5)
            bar.create_rectangle(0, 0, int(conf * bar_w), bar_h, fill=COLOR_IDLE, outline="")
            bar.create_text(bar_w // 2, bar_h // 2, text=f"{conf:.1%}", font=("Arial", 9))


# ------------------------------- Запуск ------------------------------------
def main() -> None:
    root = tk.Tk()
    try:
        recognizer = Recognizer(MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        messagebox.showerror("Ошибка загрузки модели", str(e))
        root.destroy()
        return
    DrawingApp(root, recognizer)
    root.mainloop()


if __name__ == "__main__":
    main()
