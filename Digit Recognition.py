import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Plot first 10 images
def plot_input_img(i):
    plt.imshow(x_train[i], cmap='binary')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
    plt.show()

for i in range(10):
    plot_input_img(i)

# Preprocess data
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Build model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))  # 10 output classes for digits 0-9

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
cb = [early_stopping, mc]

# Train model
history = model.fit(x_train, y_train, epochs=50, validation_split=0.3, callbacks=cb)

# Load best saved model
model_S = keras.models.load_model('best_model.h5')

# Evaluate on test data
score = model_S.evaluate(x_test, y_test)
print(f'Test accuracy: {score[1]:.4f}')
