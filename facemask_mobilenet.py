from keras.applications import MobileNetV2
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# Load MobileNetV2 as the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global pooling layer to reduce dimensions
x = Dense(128, activation='relu')(x)  # Fully connected layer
x = Dropout(0.5)(x)  # Dropout to reduce overfitting
predictions = Dense(1, activation='sigmoid')(x)  # Output layer with sigmoid activation

# Combine base model and custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    brightness_range=[0.8, 1.2],
    width_shift_range=0.2,
    height_shift_range=0.2,
    validation_split=0.2
)


test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    'FaceMask-Detector/train',  # Replace with your train dataset path
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

test_set = test_datagen.flow_from_directory(
    'FaceMask-Detector/test',  # Replace with your test dataset path
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
)

# Compile model with L2 regularization
# from tensorflow.

# x = Dense(128, activation='relu', kernel_regularizer=l2(0.01)(base_model.output))
# x = Dropout(0.5)(x)
# predictions = Dense(1, activation='sigmoid')(x)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    training_set,
    validation_data=test_set,
    epochs=10,
    callbacks=[early_stopping]
)

# Fine-tune the base model
for layer in base_model.layers[:100]:
    layer.trainable = True

# Re-compile the model for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model
history_fine_tune = model.fit(
    training_set,
    validation_data=test_set,
    epochs=10,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('new_face_mask_detector_2.h5')

# Evaluate the model
loss, accuracy = model.evaluate(test_set)
print(f"Test Accuracy: {accuracy:.2f}")

# Visualize training results
plt.plot(history.history['accuracy'], label='Train Accuracy (Initial)')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy (Initial)')
plt.plot(history_fine_tune.history['accuracy'], label='Train Accuracy (Fine-tune)')
plt.plot(history_fine_tune.history['val_accuracy'], label='Validation Accuracy (Fine-tune)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
