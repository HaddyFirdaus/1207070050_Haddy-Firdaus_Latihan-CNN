# Mendefinisikan direktori utama dataset
#dataset ini upload terlebih dulu ke drive agar bisa di load di google colab,
#atau kalian bisa langsung download lewat kaggle langsung di google colab

import os
import matplotlib.pyplot as  plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow import keras


base_dir = 'D:/LATIHAN CNN/flowers'
print(os.listdir(base_dir))

# Menghitung jumlah gambar pada dataset
number_label = {}
total_files = 0
for i in os.listdir(base_dir):
    counting = len(os.listdir(os.path.join(base_dir, i)))
    number_label[i] = counting
    total_files += counting

print("Total Files : " + str(total_files))

# Visualisasi jumlah gambar tiap kelas

plt.bar(number_label.keys(), number_label.values());
plt.title("Jumlah Gambar Tiap Label");
plt.xlabel('Label');
plt.ylabel('Jumlah Gambar');
plt.show()

img_each_class = 1
img_samples = {}
classes = list(number_label.keys())


for c in classes:
    temp = os.listdir(os.path.join(base_dir, c))[:img_each_class]
    for item in temp:
        img_path = os.path.join(base_dir, c, item)
        img_samples[c] = img_path

for i in img_samples:
    fig = plt.gcf()
    img = mpimg.imread(img_samples[i])
    plt.title(i)
    plt.imshow(img)
    plt.show()

IMAGE_SIZE = (100,100)
BATCH_SIZE = 64
SEED = 999
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2
)
# Menyiapkan data train dan data validation
train_data = datagen.flow_from_directory(
    base_dir,
    class_mode='categorical',
    subset='training',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

valid_data = datagen.flow_from_directory(
    base_dir,
    class_mode='categorical',
    subset='validation',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)
# Image Augmentation
data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal",
                      input_shape=(IMAGE_SIZE[0],
                                  IMAGE_SIZE[1],
                                  3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1./255)
  ]
)
# Loading DenseNet201 model
base_densenet_model = tf.keras.applications.DenseNet201(include_top=False,
                                                        weights='imagenet',
                                                        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                                        pooling='max')
base_densenet_model.trainable=False
train_data.preprocessing_function = tf.keras.applications.densenet.preprocess_input
# Transfer learning DenseNet201
densenet_model = tf.keras.models.Sequential([
  data_augmentation,
  base_densenet_model,
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(5, activation='softmax')
])

# Compiling model
densenet_model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
  )
#Melatih Model
# Melatih model DenseNet201
densenet_hist = densenet_model.fit(
    train_data,
    epochs=10,
    validation_data = valid_data
)
import numpy as np
from keras.preprocessing import image
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# Create a tkinter window
root = tk.Tk()

# Hide the main tkinter window
root.withdraw()

# Open a file dialog for file selection
uploaded = filedialog.askopenfilename()

for fn in uploaded:
    # prediksi gambar
    path = fn
    img = cv2.imread(uploaded)
    img = cv2.resize(img, (100, 100))  # Resize the image to the input size required by the model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image back to RGB color space
    imgplot = plt.imshow(img)
    x = img / 255.0  # Normalize the image
    x = np.expand_dims(x, axis=0)  # Add an extra dimension to match the model's input shape

    images = np.vstack([x])
    classes = densenet_model.predict(images, batch_size=BATCH_SIZE)
    classes = np.argmax(classes)

    print(fn)
    if classes == 0:
        print('daisy')
    elif classes == 1:
        print('dandelion')
    elif classes == 2:
        print('rose')
    elif classes == 3:
        print('sunflower')
    else:
        print('tulip')

plt.show()
densenet_model.save('model-flowers-recognition.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(densenet_model)
tflite_model = converter.convert()
with tf.io.gfile.GFile('model-flowers-recognition.tflite', 'wb') as f:
  f.write(tflite_model)