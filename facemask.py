import numpy as np
import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
import cv2
import datetime
# Load the MobileNetV2 model
from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import matplotlib.pyplot as plt

# Build the CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dense(100, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

base_model = MobileNetV2(weights= 'imagenet', include_top=False, input_shape=(150, 150, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#  3. Adjust Learning Rate
from  keras.optimizers import Adam
model.compile(optimizer= Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# 4. Address Overfitting
# from keras.layers import Dropout
# model.add(Dropout(0.5))

# 5. Evaluate Performance


from keras.regularizers import l2
Dense(100, activation='relu', kernel_regularizer=l2(0.01))

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'test',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary'
)

# Train the model
model.fit(
    training_set,
    epochs=20,
    validation_data=test_set
)

history = model.fit(training_set, validation_data=test_set, epochs = 20)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Save the model
model.save('mymodel.h5')

# Load the model
mymodel = load_model('mymodel.h5')

# Live Detection
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    while cap.isOpened():
        _, img = cap.read()
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (150, 150))
            face_array = np.expand_dims(face_img, axis=0) / 255.0
            pred = mymodel.predict(face_array)[0][0]
            label = "NO MASK" if pred > 0.5 else "MASK"
            color = (0, 0, 255) if pred > 0.5 else (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow('Live Face Mask Detector', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
