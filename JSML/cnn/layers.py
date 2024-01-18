from scipy.signal import convolve2d
import numpy as np


# Convolutional Layer


class Layer_Convolutional:
    def __init__(self, filters, padding_type='valid', biases=None, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0, input_formatting_hierarchy='default'):
        self.filters = [self.validate_filter(f) for f in self.ensure_list(filters)]
        self.padding_type = self.validate_padding_type(padding_type)
        self.biases = np.array(biases, dtype=float) if biases is not None else np.zeros(len(self.filters))
        self.weight_regularizer_l1 = max(weight_regularizer_l1, 0)
        self.weight_regularizer_l2 = max(weight_regularizer_l2, 0)
        self.bias_regularizer_l1 = max(bias_regularizer_l1, 0)
        self.bias_regularizer_l2 = max(bias_regularizer_l2, 0)
        self.input_formatting_hierarchy = input_formatting_hierarchy

    def ensure_list(self, item):
        return item if isinstance(item, list) else [item]

    def validate_filter(self, filter):
        if not isinstance(filter, np.ndarray) or filter.ndim not in [1, 2]:
            raise ValueError("Each filter must be a 1D or 2D numpy array.")
        return np.array(filter, dtype=float)

    def validate_padding_type(self, padding_type):
        if padding_type not in ['valid', 'same']:
            raise ValueError("Padding type must be either 'valid' or 'same'.")
        return padding_type

    def calculate_padding(self, image, filter):
        if self.padding_type == 'same':
            # Calculate padding for height and width
            pad_h = ((image.shape[1] - 1) * self.stride + filter.shape[0] - image.shape[1]) // 2
            pad_w = ((image.shape[2] - 1) * self.stride + filter.shape[1] - image.shape[2]) // 2
            return np.pad(image, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        else:
            return image

    def apply_filter(self, image, filter, mode='forward'):
        # Forward pass
        if mode == 'forward':
            if image.ndim == 1:
                # 1D convolution
                return scipy_convolve(image, filter, mode='valid')
            elif image.ndim == 2:
                # 2D convolution
                return scipy_convolve2d(image, filter, mode='valid')
            elif image.ndim == 3:
                # Multi-channel 2D convolution
                return np.sum([scipy_convolve2d(image[:, :, c], filter, mode='valid') for c in range(image.shape[2])], axis=0)
            else:
                raise ValueError("Unsupported input dimensionality for forward mode.")

        # Backward pass
        elif mode == 'backward':
            if image.ndim == 1:
                # 1D convolution derivative
                return scipy_convolve(image, np.flip(filter), mode='full')[1:-1]
            elif image.ndim == 2:
                # 2D convolution derivative
                return scipy_convolve2d(image, np.rot90(filter, 2), mode='full')
            elif image.ndim == 3:
                # Multi-channel 2D convolution derivative
                rotated_filter = np.rot90(filter, 2)
                channel_gradients = [scipy_convolve2d(image[:, :, c], rotated_filter, mode='full') for c in range(image.shape[2])]
                return np.sum(channel_gradients, axis=0)
            else:
                raise ValueError("Unsupported input dimensionality for backward mode.")
        else:
            raise ValueError("Unsupported mode. Choose 'forward' or 'backward'.")

    def interpret_input_format(self, image):
        if self.input_formatting_hierarchy == 'default':
            if image.ndim == 1:
                return image.reshape(1, -1, 1)
            elif image.ndim == 2:
                return image.reshape(image.shape[0], image.shape[1], 1)
            elif image.ndim == 3:
                return image
            else:
                raise ValueError("Invalid image dimensionality.")
        else:
            # Handle alternative setups based on 'input_formatting_hierarchy'
            pass  # Implement as needed

    def validate_input_filter_compatibility(self, image, filter):
        if (image.ndim - 1) != filter.ndim:
            raise ValueError("Filter dimension must match image dimension.")
        if any(f > d for f, d in zip(filter.shape, image.shape[:2])):
            raise ValueError("Filter dimensions must not exceed image dimensions.")

    def forward(self, inputs, training):
        self.inputs = self.ensure_list(inputs)
        self.outputs = []

        for input_image in self.inputs:
            filter_outputs = []
            for filter in self.filters:
                formatted_image = self.interpret_input_format(input_image)
                self.validate_input_filter_compatibility(formatted_image, filter)
                padded_image = self.apply_padding(formatted_image, filter)
                conv_result = self.apply_filter(padded_image, filter)
                filter_outputs.append(conv_result + self.biases)
            self.outputs.append(np.array(filter_outputs))

    def backward(self, dvalues):
        self.dweights = np.zeros_like(self.filters)
        self.dbiases = np.zeros_like(self.biases)
        self.dinputs = []

        if self.weight_regularizer_l1 > 0:
            self.dweights += self.weight_regularizer_l1 * np.sign(self.filters)
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.filters

        if self.bias_regularizer_l1 > 0:
            self.dbiases += self.bias_regularizer_l1 * np.sign(self.biases)
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        for i, input_image in enumerate(self.inputs):
            for j, filter in enumerate(self.filters):
                self.dinputs.append(self.apply_filter(np.pad(dvalues[j], 1), filter, mode='backward'))
                self.dweights[j] += self.apply_filter(dvalues[j], input_image, mode='backward')
            self.dbiases[j] += np.sum(dvalues[j])

        self.dinputs = np.array(self.dinputs)


# Flatten layer


class Layer_Flatten:

    # forward
    def forward(self, inputs, training, model=None):

        self.batch_size = len(model.batch_X)

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


# Maxpooling layer


class Layer_MaxPooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        batch_size, height, width, channels = self.input_shape
        
        # Calculate output dimensions
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output
        self.output = np.zeros((batch_size, output_height, output_width, channels))
        
        # Perform max pooling
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                self.output[:, i, j, :] = np.max(inputs[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
        return self.output

    def backward(self, dvalues):
        # Initialize gradient array
        self.dinputs = np.zeros(self.input_shape)
        
        # Iterate over each region and propagate the gradient
        for i in range(self.output.shape[1]):
            for j in range(self.output.shape[2]):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # Reshape for broadcasting
                mask = (self.output[:, i, j, :] == self.inputs[:, h_start:h_end, w_start:w_end, :])
                self.dinputs[:, h_start:h_end, w_start:w_end, :] += mask * dvalues[:, i:i+1, j:j+1, :]
        return self.dinputs


# Averagepooling layer


class Layer_AveragePooling:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        batch_size, height, width, channels = self.input_shape
        
        # Calculate output dimensions
        output_height = (height - self.pool_size) // self.stride + 1
        output_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output
        self.output = np.zeros((batch_size, output_height, output_width, channels))
        
        # Perform average pooling
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                self.output[:, i, j, :] = np.mean(inputs[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
        return self.output

    def backward(self, dvalues):
        # Initialize gradient array
        self.dinputs = np.zeros(self.input_shape)
        
        # Iterate over each region and propagate the gradient
        for i in range(self.output.shape[1]):
            for j in range(self.output.shape[2]):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # Calculate and distribute gradient
                gradient = dvalues[:, i:i+1, j:j+1, :] / (self.pool_size * self.pool_size)
                self.dinputs[:, h_start:h_end, w_start:w_end, :] += np.ones((self.pool_size, self.pool_size)) * gradient
        return self.dinputs