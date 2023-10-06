# Convolutional layer class


class Layer_Convolutional:

    # Initialization
    def __init__(self, Filters, Padding=0, Biases=0, IsMultipleFilters=True,
                 IsMultipleInputs=True, weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):

        # Set layer variables
        self.Padding = Padding
        self.Biases = Biases
        self.Filters = np.array(Filters, dtype=object)
        self.IsMultipleFilters = IsMultipleFilters
        self.IsMultipleInputs = IsMultipleInputs
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        # Define blank list
        # to append to
        self.FilterSizes = []

        # Itterate through every filter and append
        # it's size to the FilterSizes list
        for i, kernel in enumerate(self.Filters):

            # Append the size
            self.FilterSizes.append(np.array(self.Filters[i]).shape)

    # ConvolutionalSlicer method
    def ConvolutionalSlicer(self, kernel, SlicerInput, ConvolutionType='Basic_Convolution', Pass='forward', index=0):

        # Set the current sizes
        # (length x hight)
        self.KernelSize = [len(kernel[0]), len(kernel)]
        self.InputSize = [len(SlicerInput[0]), len(SlicerInput)]

        # Calculate output size
        # length x hight
        self.OutputSize = [self.InputSize[0] - self.KernelSize[0] +
                           1, self.InputSize[1] - self.KernelSize[1] + 1]

        # Define blank list
        self.ConvolutionalSlice = []

        # Add value to self object
        self.SlicerInput = SlicerInput

        self.ConvolutionResult = convolve2d(SlicerInput, kernel, mode='valid')

        # If its the foward pass
        # than add a bias
        if Pass == 'forward':

            if np.ndim(self.Biases) > 0:

                # Add the bias
                self.ConvolutionResult += self.Biases[0][index]

        # Return the reshaped output of
        # the convolution slice after
        # undergoing it's given equation
        return np.reshape(self.ConvolutionResult, self.OutputSize)

# Additive convolution


class Basic_Convolution(Layer_Convolutional):

    # Forward method
    def forward(self, inputs, training):

        self.batch_size = len(model.batch_X)

        self.XPadded = inputs

        # Padding check
        if self.Padding > 0:

            # Multiple inputs check
            if self.IsMultipleInputs == True:

                self.XPadded = []

                # If true, iterate through
                # inputs and pad
                for matrix in inputs:

                    # Add padding
                    self.XPadded.append(np.pad(matrix, self.Padding))

        # For singular input
        else:

            # Add padding
            self.XPadded = np.pad(self.XPadded, self.Padding)

        # Define blank array
        # for input sizes
        self.InputSize = []

        # If there are multiple inputs
        if self.IsMultipleInputs == True:

            # Itterate through each input
            for matrix in self.XPadded:

                # Append the hight x length
                # of each input to the variable
                self.InputSize.append([len(matrix[0]), len(matrix)])

        # If there is one input
        else:

            # Get hight x length
            # of the singular input
            # and append it to the variable
            self.InputSize = [len(self.XPadded[0]), len(self.XPadded)]

        self.output = []

        self.outputPreBatch = []

        # Itterate through each input
        for i, matrix in enumerate(self.XPadded):

            # And for every kernel
            for index, kernel in enumerate(self.Filters):

                # Append the output of the convolution
                self.outputPreBatch.append((self.ConvolutionalSlicer(
                    kernel, matrix, 'Basic_Convolution', 'forward', index)))

            self.output.append(self.outputPreBatch)

        self.output = np.array(self.outputPreBatch, dtype=object)

    def backward(self, dvalues):

        # Define blank lists to append to
        self.dweights = []
        self.dbiases = []
        self.dinputs = []

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.Filters)
            dL1[self.Filters < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                self.Filters
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.Biases)
            dL1[self.Biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                self.Biases

        # Iterate through every output (input on
        # the forward pass, since self.output's
        # first dimention is the inputs)
        for i in range(0, self.batch_size):

            # Iterate through every filter index
            for j in range(0, len(self.Filters)):

                # Get the rotated filter (180 degrees)
                self.rotated_filter = np.rot90(self.Filters[j], 2)

                # Convolve the gradient with the rotated filter
                self.dinputs.append(self.ConvolutionalSlicer(
                    self.rotated_filter, np.pad(dvalues[j], 1), 'Basic_Convolution', 'backward'))

                # Append the derivative of the weights at index j
                self.dweights.append(self.ConvolutionalSlicer(
                    dvalues[j], self.XPadded[i], 'Basic_Convolution', 'backward'))

                self.dbiases.append(sum(dvalues[j]))

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
