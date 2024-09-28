import os
from pathlib import Path
import tkinter as tk
from tkinter import Label

import keras
import numpy as np
from PIL import Image, ImageGrab, ImageTk

# Параметры изображения
IMG_SIZE = (224, 224)

# Загружаем обученную модель
models_dir = Path("models")
models = os.listdir(models_dir)
model = keras.models.load_model(models_dir.joinpath(models[-1]))  # lates by default

# Словарь классов (нужно заменить на реальные классы, которые использовались при обучении)
class_names = ["Урбанистика", "Фантастика", "Пейзаж"]


# Функция для предсказания класса изображения
def classify_image(img):
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0  # Нормализуем изображение
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность для батча
    predictions = model.predict(img_array)[0]  # Предсказания модели
    result_text = "Результаты классификации:\n"
    for i, class_name in enumerate(class_names):
        confidence = predictions[i] * 100  # Уверенность по каждому классу
        result_text += f"{class_name}: {confidence:.2f}%\n"
    return result_text


# Функция для вставки изображения из буфера и вывода предсказания
def paste_image():
    try:
        # Получаем изображение из буфера обмена
        img = ImageGrab.grabclipboard()
        if isinstance(img, Image.Image):
            img.thumbnail((300, 300))  # Уменьшаем изображение для отображения
            img_tk = ImageTk.PhotoImage(img)
            label_image.config(image=img_tk)
            label_image.image = img_tk  # Сохраняем ссылку на изображение

            # Классифицируем изображение
            img = img.resize((224, 224))
            img = img.convert("RGB")
            result = classify_image(img)
            label_result.config(text=result)
        else:
            label_result.config(text="Буфер обмена не содержит изображения.")
    except Exception as e:
        label_result.config(text=f"Ошибка: {str(e)}")


# Создаем окно приложения
window = tk.Tk()
window.title("Классификация изображений")

# Кнопка вставки изображения из буфера обмена
btn_paste = tk.Button(window, text="Вставить изображение из буфера", command=paste_image)
btn_paste.pack()

# Метка для отображения изображения
label_image = Label(window)
label_image.pack()

# Метка для отображения результата классификации
label_result = Label(window, text="Класс: -\nУверенность: -", font=("Helvetica", 14))
label_result.pack()

# Запуск основного цикла приложения
window.mainloop()
