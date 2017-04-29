import random
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.image as mpimg

# Let's load the downloaded training data
lines = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
lines = lines[1:]

# I'll use 20% of the data for validation, and the rest for training
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
print('Training samples = ', len(train_samples))
print('Validation samples = ', len(validation_samples))

def data_generator(samples, batch_size=32):
    """A generator that produces samples a batch at a time.

    Parameters
    ----------
    `samples` : <list>
        A collection of training or validation samples

    `batch_size` : <int>
        The subset-size of each yield of the generator

    Yields
    ------
    <list>
        A subset of the `samples` collection
    """

    while True:
        random.shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset : offset + batch_size]

            batch_images = []
            batch_angles = []

            for batch_sample in batch_samples:
                img_center_name = batch_sample[0]
                img_left_name = batch_sample[1]
                img_right_name = batch_sample[2]

                img_center = mpimg.imread(img_center_name.strip())
                img_left = mpimg.imread(img_left_name.strip())
                img_right = mpimg.imread(img_right_name.strip())

                angle_center = float(batch_sample[3])
                angle_left = float(batch_sample[3])+0.5
                angle_right = float(batch_sample[3])-0.5

                batch_images.append(img_center)
                batch_images.append(img_left)
                batch_images.append(img_right)

                batch_angles.append(angle_center)
                batch_angles.append(angle_left)
                batch_angles.append(angle_right)

            augmented_images, augmented_angles = [], []

            for image, angle in zip(batch_images, batch_angles):
                augmented_images.append(image)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle)
                augmented_angles.append(angle * -1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

training_generator = data_generator(train_samples)
validation_generator = data_generator(validation_samples)

# The model architecture
from keras.models import Sequential
from keras.layers import (
    Convolution2D,
    Cropping2D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
)

model = Sequential()
model.add(
    Lambda(
        lambda x: x/255.-1.,        # normalization
        input_shape=(160, 320, 3),
        output_shape=(160, 320, 3)
    )
)
model.add(Cropping2D(cropping=((70, 25),(0, 0))))
model.add(Convolution2D(12, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 3, 3, activation="relu"))
model.add(Convolution2D(60, 3, 3, activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

# model training
history_object = model.fit_generator(
    training_generator,
    samples_per_epoch=len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=10
)

#save the model
model.save('model.h5')

### plot the training and validation loss for each epoch
import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
