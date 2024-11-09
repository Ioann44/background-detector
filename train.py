from datetime import datetime
from pathlib import Path

from keras import layers, models
from keras.api.applications import InceptionV3
from keras.api.applications.mobilenet_v2 import preprocess_input
from keras.api.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Параметры
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3  # Пейзаж, Урбанистика, Фантастика
EPOCHS = 300

early_stopping = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

# Создаем генератор данных с аугментацией и делением на обучающую и валидационную выборки
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, validation_split=0.2  # 80% на обучение, 20% на валидацию
)

# Путь к директории с папками категорий
data_dir = "resized"

# Генератор тренировочной выборки
train_generator = datagen.flow_from_directory(
    data_dir,  # Путь к директории с изображениями
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",  # Использовать 80% данных для обучения
)

# Генератор валидационной выборки
validation_generator = datagen.flow_from_directory(
    data_dir,  # Путь к той же директории
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",  # Использовать 20% данных для валидации
)

# Вывод информации о классах
print("Классы:", train_generator.class_indices)

# Модель на базе MobileNetV2
base_model = InceptionV3(input_shape=IMG_SIZE + (3,), include_top=False)
base_model.trainable = False  # Заморозим веса предобученной части

# Добавляем полносвязную голову
model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(units=128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),  # 3 класса
    ]
)

# Компилируем модель
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Тренировка модели
history = model.fit(
    train_generator, epochs=EPOCHS, validation_data=validation_generator, callbacks=[early_stopping]
)

# Сохранение модели
date = datetime.now()
date_str = date.strftime("%Y-%m-%d_%H-%M")
model.save(Path("models").joinpath(date_str).with_suffix(".keras"))
