import numpy as np
import skimage.measure

class Pooling:
    def __init__(self, filterSize):
        self.filterSize = (filterSize, filterSize)
        self.input = np.ones((128, 7, 7))


class Layer_MaxPooling(Pooling):

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
    
# Test input
input = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

filter_size = 2  # Scalar value
max_pooling = Layer_MaxPooling(filter_size)

# dvalues - based on a 2x2 output from max pooling
dvalues = np.array([
    [1, 2],
    [3, 4]
])

# Expected output of the backward pass
# Non-zero gradients are placed at the positions of the max values in each 2x2 block
expected_output = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 2],
    [0, 0, 0, 0],
    [0, 3, 0, 4]
])

# Running the backward method
max_pooling.inputs = input  # Mocking the forward pass inputs
backward_output = max_pooling.backward(dvalues)

# Verification
if np.array_equal(backward_output, expected_output):
    print("Backward method test - PASS")
else:
    print("Backward method test - FAIL")
    print("Expected output:")
    print(expected_output)
    print("Actual output:")
    print(backward_output)