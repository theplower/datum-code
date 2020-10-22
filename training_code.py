#getting the training dataset from my google drive to the colab
#!cp -r drive/My\ Drive/ArSL localData/

#the dataset is named ArSL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

num_classes = 32
input_shape = (64, 64, 1)
image_size = (64, 64)

#making the model 
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.7),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

#getting the batches and doing some augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        )

train_generator = train_datagen.flow_from_directory(
        'localData/ArSL',
        target_size=image_size,
        batch_size=32,
        class_mode='categorical', color_mode='grayscale')


#train the model
callbacks = [
    keras.callbacks.ModelCheckpoint("saves3/save_at_{epoch}.h5"),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
        train_generator,
        callbacks=callbacks,
        epochs=20)

