import cv2
import tensorflow as tf
import numpy as np 
from mnist import MNIST
import time

labels_map = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',  
              'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

# ============================ Data loading and preprocessing ===================================

dataset_split = 'balanced'
classes = 47

emnist_data = MNIST(path='emnist\\' + dataset_split, return_type='numpy')
emnist_data.select_emnist(dataset_split)
x_train, y_train = emnist_data.load_training()
x_test, y_test = emnist_data.load_testing()

# Reshape the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
y_train = y_train.reshape(x_train.shape[0], 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_test = y_test.reshape(x_test.shape[0], 1)

x_train = x_train/255.
x_test = x_test/255.

# Encode Categorical Integer Labels Using a One-Hot Scheme
y_train = tf.keras.utils.to_categorical(y_train, num_classes = classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes = classes)

print('X train shape', x_train.shape)
print('y train shape', y_train.shape)
print('X test shape', x_test.shape)
print('y test shape', y_test.shape)

# ============================ Model definition ===================================
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1), activation = 'relu', padding = 'same'),
  tf.keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64, 3, input_shape=(28, 28, 1), activation = 'relu', padding = 'same'),
  tf.keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(classes, activation=tf.nn.softmax)
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# ============================ Training ===================================
model.fit(x_train, y_train, batch_size=128, epochs=40, validation_split=0.1)

score = model.evaluate(x_test, y_test)
model.save('balanced_model.h5')
print('Test error', score[0])
print('Test accuracy', score[1])

# ============================ Inferencing ===================================

# model = tf.keras.models.load_model('balanced_model.h5')
image = cv2.imread('') #input image here
w, h = image.shape
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28,28))
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
x = np.reshape(thresh1,[-1, 28, 28, 1])
start_time = time.time()
yhat = model.predict(x)
print('Inferencing time = ' + str(time.time()-start_time) + ' sec')
idx = np.argmax(yhat)
pred = labels_map[idx]
cv2.rectangle(image, (w-20,h-30), (w, h), (0, 255, 0), cv2.FILLED)
cv2.putText(image, pred, (w-15,h-25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
cv2.imwrite('output.jpg', image)
