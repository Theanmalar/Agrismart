import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8  # smaller dataset -> smaller batch
EPOCHS = 15     # adjust for your dataset size

def train_model(train_path, test_path, num_classes, model_name):
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    test_gen = test_datagen.flow_from_directory(
        test_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    # Simple CNN model
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS)

    model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
    model.save(model_path)
    print(f"{model_name} saved at {model_path}")

# ------------------ Train Soil Model ------------------
soil_train = os.path.join(BASE_DIR, "datasets", "soil_images", "train")
soil_test = os.path.join(BASE_DIR, "datasets", "soil_images", "test")
train_model(soil_train, soil_test, num_classes=3, model_name="soil_images")

# ------------------ Train Disease Model ------------------
disease_train = os.path.join(BASE_DIR, "datasets", "disease_images", "train")
disease_test = os.path.join(BASE_DIR, "datasets", "disease_images", "test")
train_model(disease_train, disease_test, num_classes=4, model_name="disease_images")