import JSML
import numpy as np
import os
import cv2

def Load_MNIST_Dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                os.path.join(path, dataset, label, file),
                cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)


def Create_Data_MNIST(path):

    # Load both sets separately
    X, y = Load_MNIST_Dataset('train', path)
    X_test, y_test = Load_MNIST_Dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test

# Create dataset
X, y, X_test, y_test = Create_Data_MNIST('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
          127.5) / 127.5

# Instantiate the model
model = JSML.Model()

# Add layers
model.add(JSML.Layer_Dense(X.shape[1], 128))
model.add(JSML.Activation_ReLU())
model.add(JSML.Layer_Dense(128, 128))
model.add(JSML.Activation_ReLU())
model.add(JSML.Layer_Dense(128, 10))
model.add(JSML.Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=JSML.Loss_CategoricalCrossentropy(),
    optimizer=JSML.Optimizer_Adam(decay=1e-3),
    accuracy=JSML.Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=10, batch_size=128, print_every=100)

model.evaluate(X_test, y_test)