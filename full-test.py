from scipy.signal import convolve2d
import numpy as np
from JSML.basic import *
from JSML.utils import *
import skimage.measure
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# NOTE: for tommrow, continnue debugging the conv layer. The shape is not aaccrate to what it should be and the way I'm coimbinnging finnal ouuutpuuts is wrong. Fix it and conntine debugging
# ? why is valid producing inhomogenous output??
# Convolutional Layer

class Layer_Convolutional:
    def __init__(self, filters, padding_type='valid', custom_padding=0, biases=None, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.filters = filters
        self.num_filters = len(filters)
        self.padding_type = padding_type
        self.custom_padding = custom_padding
        self.biases = np.array(biases, dtype=float) if biases is not None else np.zeros(self.num_filters)
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def apply_padding(self, image, filter_shape):
        if self.padding_type == 'valid':
            return image
        elif self.padding_type == 'same':
            pad_h = (filter_shape[0] - 1) // 2
            pad_w = (filter_shape[1] - 1) // 2
            # Padding applied to height and width only, not to batch dimension
            padding = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))
            return np.pad(image, padding, mode='constant', constant_values=(0, 0))
        else:
            raise ValueError(f"Unsupported padding type: {self.padding_type}")

    def apply_filter(self, image, filter, mode='forward'):
            # Debugging print: Check the mode and dimensions of input and filter
            print(f"Mode: {mode}, Image Shape: {image.shape}, Filter Shape: {filter.shape}")

            if mode == 'forward':
                if image.ndim == 3:  # 2D image with batch size
                    # Debugging print: Entering forward mode with 3D image
                    print("Entering forward mode with 3D image")
                    result = np.array([convolve2d(img, filter, mode='valid') for img in image])

                    # Debugging print: Check the shape of the result after convolution
                    print(f"Resulting shape after convolution (forward): {result.shape}")
                    return result
                else:
                    raise ValueError("Unsupported input dimensionality for forward mode.")
            elif mode == 'backward':
                if image.ndim == 3:  # 2D image with batch size
                    # Debugging print: Entering backward mode with 3D image
                    print("Entering backward mode with 3D image")
                    result = np.array([convolve2d(img, np.rot90(filter, 2), mode='full') for img in image])

                    # Debugging print: Check the shape of the result after convolution
                    print(f"Resulting shape after convolution (backward): {result.shape}")
                    return result
                else:
                    raise ValueError("Unsupported input dimensionality for backward mode.")
            else:
                raise ValueError("Unsupported mode. Choose 'forward' or 'backward'.")

    def forward(self, inputs, training):

        '''print("Pre Convolution - Output Shape:", np.array(inputs).shape)
        plt.imshow(inputs[0])
        plt.title("Output Image - Pre Convolution")
        plt.show()'''

        #print("Input Shape:", inputs.shape)
        self.inputs = inputs
        self.outputs = []

        for i, filter in enumerate(self.filters):
            padded_inputs = self.apply_padding(inputs, filter.shape)
            filter_output = self.apply_filter(padded_inputs, filter)
            self.outputs.append(filter_output)

        if self.padding_type == 'valid':
            # If padding type is 'valid', add biases to each element of the output
            for i in range(self.num_filters):
                self.outputs[i] += self.biases[i]
        elif self.padding_type == 'same':
            # For other padding types, reshape biases and add
            self.biases = np.reshape(self.biases, (self.num_filters, 1, 1, 1))
            self.outputs += self.biases
        else:
            raise ValueError(f"Unsupported padding type: {self.padding_type}")
        self.output = self.outputs
        return self.output

    def backward(self, dvalues):
        self.dweights = []
        self.dbiases = np.zeros_like(self.biases)
        self.dinputs = np.zeros_like(self.inputs)

        for i, filter in enumerate(self.filters):
            filter_dvalues = dvalues if i == 0 else self.outputs[i-1]
            for j, img in enumerate(self.inputs):
                self.dinputs[j] += self.apply_filter(filter_dvalues[j], filter, mode='backward')
            self.dweights.append(self.apply_filter(np.sum(filter_dvalues, axis=0), img, mode='backward'))

        self.dbiases += np.sum(dvalues, axis=(0, 1, 2))

        if self.weight_regularizer_l1 > 0:
            for i in range(len(self.filters)):
                self.dweights[i] += self.weight_regularizer_l1 * np.sign(self.filters[i])

        if self.weight_regularizer_l2 > 0:
            for i in range(len(self.filters)):
                self.dweights[i] += 2 * self.weight_regularizer_l2 * self.filters[i]

        return self.dinputs


# Convolutional Layer Output Normalization
    
class Layer_ConvOutNormalization:
    def pad_to_max_dimensions(self, input_array, max_height, max_width):
        current_height, current_width = input_array.shape[1], input_array.shape[2]
        padding_height = max_height - current_height
        padding_width = max_width - current_width

        top_padding = padding_height // 2
        bottom_padding = padding_height - top_padding
        left_padding = padding_width // 2
        right_padding = padding_width - left_padding

        if padding_height % 2 != 0:
            bottom_padding += 1
        if padding_width % 2 != 0:
            right_padding += 1

        padded_array = np.pad(input_array, ((0, 0), (top_padding, bottom_padding), (left_padding, right_padding)), mode='constant')
        return padded_array

    def forward(self, inputs, training):
        max_height = max(output.shape[1] for output in inputs)
        max_width = max(output.shape[2] for output in inputs)

        padded_outputs = [self.pad_to_max_dimensions(filter_output, max_height, max_width) for filter_output in inputs]
        self.output = np.sum(padded_outputs, axis=0)
        return self.output

# Pooling Parent layer

class Pooling:
    def __init__(self, filterSize):
        self.filterSize = (filterSize, filterSize)

    def forward(self, input, training):
        pass

# Maxpooling Layer

class Layer_MaxPooling(Pooling):
    def forward(self, inputs, training):
        self.inputs = np.array(inputs)
        # Apply max pooling
        self.output = [skimage.measure.block_reduce(image, self.filterSize, np.max) for image in self.inputs]
        print("Max Pooling - Output Shape:", np.array(self.output).shape)
        return self.output

    def backward(self, dvalues):
        # Initialize gradient array with zeros
        self.dinput = np.zeros(self.inputs.shape)
        print("Max Pooling - dvalues Shape:", dvalues.shape)

        for i in range(self.inputs.shape[0]):
            for j in range(dvalues.shape[1]):
                for k in range(dvalues.shape[2]):
                    # Determine the region of the input that contributed to the max pooling
                    region = self.inputs[i, j*self.filterSize[0]:(j+1)*self.filterSize[0],
                                         k*self.filterSize[1]:(k+1)*self.filterSize[1]]
                    maxVal = np.max(region)
                    # Create a mask where the maximum value occurred
                    mask = (region == maxVal)
                    # Propagate the gradient to the maximum value
                    self.dinput[i, j*self.filterSize[0]:(j+1)*self.filterSize[0],
                                k*self.filterSize[1]:(k+1)*self.filterSize[1]] += mask * dvalues[j, k]

        return self.dinput

# Averagepooling layer

class Layer_AveragePooling(Pooling):
    def forward(self, inputs, training):
        inputs = np.array(inputs)
        self.input = inputs
        self.input_shape = inputs.shape
        self.output = [skimage.measure.block_reduce(image, self.filterSize, np.mean) for image in inputs]
        
        '''print("Average Pooling - Output Shape:", np.array(self.output).shape)
        plt.imshow(self.output[0])
        plt.title("Output Image - Average Pooling")
        plt.show()'''
        
        return np.array(self.output)

    def backward(self, dvalues):
        self.dinput = np.zeros(self.inputs.shape)
        print("Max Pooling - dvalues Shape:", dvalues.shape)

        for j in range(dvalues.shape[0]):
            for k in range(dvalues.shape[1]):
                # Calculate the start and end indices for the pooling region
                start_j = j * self.filterSize[0]
                end_j = start_j + self.filterSize[0]
                start_k = k * self.filterSize[0]
                end_k = start_k + self.filterSize[0]
                
                # Extract the region
                region = self.inputs[start_j:end_j, start_k:end_k]
                maxVal = np.max(region)
                mask = (region == maxVal)

                # Distribute the gradient
                self.dinput[start_j:end_j, start_k:end_k] += mask * dvalues[j, k]

        return self.dinput

# Flatten layer

class Layer_Flatten:

    # forward
    def forward(self, inputs, training):
        # ? Why did I take the batch size from the model? Possible change necesary.
        self.batch_size = np.array(inputs).shape[0]

        # Define output list
        self.output = []

        # Define the shapes of the
        # inputs for backward pass
        self.InputShape = []

        # For every input, apend the
        # flattened version of it
        for i, matrix in enumerate(inputs):

            # Append to output
            self.output.append(matrix.ravel())

            # Get the shape of
            # the current input
            self.InputShape.append(matrix.shape)

        self.output = np.concatenate(self.output)
        self.output = np.reshape(self.output, [self.batch_size, -1])
        return self.output

    # Backward
    def backward(self, dvalues):

        self.dvalues = np.ravel(dvalues)

        # Set dinputs as a
        # blank array to be
        # appended to
        self.dinputs = []

        # Set the starting index
        self.start = 0
        self.end = 0

        # For every input in
        # the forward pass
        for i, shape in enumerate(self.InputShape):

            # Multiply the length by
            # hight to find the amount
            # of numbers in the input shape
            self.size = np.prod(shape)

            self.end += self.size
            self.end = int(self.end)

            # For the amount of numbers in
            # the input shape, starting at
            # the end of all the previous
            # amounts of numbers in all of
            # the shapes combined, append
            # those number reshaped to be
            # the size of the inputs into the output
            self.dinputsPreReshape = self.dvalues[self.start:self.end]

            self.dinputs.append(
                self.dinputsPreReshape.reshape(shape[0], shape[1]))

            # Add the amount of numbers
            # used to self.start to find
            # the next starting point
            self.start = self.end
            self.start = int(self.start)

        # initialize a dictionary to store the sums
        sums = {}

        # iterate over the inputs in self.dinputs
        for input_matrix in self.dinputs:
            shape = input_matrix.shape

            # sum the input matrix with the same shape using advanced indexing and broadcasting
            if shape not in sums:
                sums[shape] = input_matrix
            else:
                sums[shape] += input_matrix

        # create a new array to store the sums
        self.summed_inputs = []

        # iterate over the keys of the sums dictionary to add the sums to the new array
        for shape, sum_input in sums.items():
            self.summed_inputs.append(sum_input)

        # convert the summed_inputs array to a NumPy array
        self.dinputs = np.array(self.summed_inputs, dtype=object)


# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

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
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test

# Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Reshape and scale the data
X = (X.reshape(X.shape[0], 28, 28).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], 28, 28).astype(np.float32) - 127.5) / 127.5


# Create filters and biases using the Create_Filters function
filter_shapes = [(3, 3), (5, 5)]  # Example filter shapes
filters, biases = Create_Filters(filter_shapes, Biases=True)
# Define the metrics you want to track
track_settings = {
    "loss": True,
    "accuracy": True,
    "learning_rate": True,
    "gradient_norms": True,
    "param_mean": True,
    # Add other metrics as needed
}

# Define graph assignments for each metric (optional)
graph_assignments = {
    "loss": [1],
    "accuracy": [1, 3],
    "learning_rate": [2],
    "gradient_norms": [3],
    "param_mean": [2],
    # Assign other metrics to graphs as needed
}

# Create a data_tracking_config object with show_data=True and save_data=False
data_tracking_config = DataTracker(track_settings, graph_assignments, show_data=True, save_data=False)

model = Model()
model.add(Layer_Convolutional(filters=filters, biases=biases[0], padding_type='valid'))
model.add(Activation_ReLU())
model.add(Layer_ConvOutNormalization())
model.add(Layer_MaxPooling(2))
model.add(Layer_AveragePooling(2))
model.add(Activation_ReLU())
model.add(Layer_Flatten())
model.add(Layer_Dense(49, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test),
            epochs=20, batch_size=128, print_every=100, data_tracking_config=data_tracking_config)

model.evaluate(X_test, y_test)