import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt 


# load dataset 

data = keras.datasets.fashion_mnist 

 # Pass 80% of data for training and 20% for testing

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#shrink pixel values from 0-255 to 0-1
train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=3)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy:", test_acc )

prediction = model.predict(test_images)


# To verify

for i in range(5):
	plt.grid(False)
	plt.imshow(test_images[i], cmap = plt.cm.binary) #plt.cm.binary shows us the grey-scale images
	plt.xlabel("Actual:" + class_names[test_labels[i]])
	plt.title("Prediction" + class_names[np.argmax(prediction[i])])
	plt.show()
