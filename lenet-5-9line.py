import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras import models, layers
import matplotlib.pyplot as plt
import os # For saving model
from keras.models import model_from_json

def show_train_history(train_history):
    plt.figure()
    ax = plt.subplot(1,2,1)
    ax.plot([None] + train_history.history['accuracy'], 'o-')
    ax.plot([None] + train_history.history['val_accuracy'], 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    ax.legend(['Train acc', 'Validation acc'], loc = 0)
    ax.set_title('Training/Validation acc per Epoch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('acc')

    fx = plt.subplot(1,2,2)
    fx.plot([None] + train_history.history["loss"], 'o-')
    fx.plot([None] + train_history.history["val_loss"], 'x-')
    # Plot legend and use the best location automatically: loc = 0.
    fx.legend(['Train Loss', 'Validation Loss'], loc = 0)
    fx.set_title('Training/Validation Loss per Epoch')
    fx.set_xlabel('Epoch')
    fx.set_ylabel('Loss')
    plt.show()

# Data Section
# Load dataset as train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Set numeric type to float32 from uint8
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform lables to one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28,28,1)


# Model Section
model = Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation = 'softmax'))

model.summary()
print("")

# Training Section
# MODEL name and weights
MODEL_NAME = 'mnist_model_lenet-5-9line.model'
MODEL_WEIG = 'mnist_model_lenet-5-9line.h5'

use_exist_model = False
if os.path.isfile(MODEL_NAME):
    hist = None
    with open(MODEL_NAME, 'r') as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json) 
    model.load_weights(MODEL_WEIG)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
    use_exist_model = True
else:
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='SGD', metrics=['accuracy'])
    hist = model.fit(x=x_train,
                     y=y_train, validation_data=(x_test, y_test),
                     epochs=10, batch_size=128, verbose=1)

# hist = model.fit(x=x_train,y=y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), verbose=1)


# Test Section and saving model
test_score = model.evaluate(x_test, y_test)
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))

# show history, if there is
if hist:
    show_train_history(hist)

# save MODEL
if not use_exist_model:
    print("\t[Info] Serialized Keras model to %s..." % (MODEL_NAME))
    with open(MODEL_NAME, 'w') as f:
        f.write(model.to_json()) 
    model.save_weights(MODEL_WEIG)
    print("\t[Info] Done!")

print("Program finished.")
