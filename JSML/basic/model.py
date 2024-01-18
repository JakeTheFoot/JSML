import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
import json
import time
import os

# Model class


class Model:

    def __init__(self):
        # Create a list of network objects
        self.layers = []
        # Softmax classifier's output object
        self.softmax_classifier_output = None

    # Add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    # Set loss, optimizer and accuracy

    def set(self, *, loss=None, optimizer=None, accuracy=None):

        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer

        if accuracy is not None:
            self.accuracy = accuracy

    # Finalize the model
    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Add all layer types used to a list
        # This is used later for training protocol establishment
        self.layerTypes = [object.__class__.__name__ for object in self.trainable_layers]
        self.layerTypes = list(dict.fromkeys(self.layerTypes))
        print(self.layerTypes)

        # Update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(
                self.trainable_layers
            )

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    # Train the model
    def train(self, X, y, *, epochs=1, batch_size=None,
              print_every=1, validation_data=None, data_tracking_config=None):

        # Initialize the DataTracker with the configuration
        if data_tracking_config:
            self.tracker = DataTracker(data_tracking_config.track_settings, 
                                           data_tracking_config.graph_assignments, 
                                           data_tracking_config.show_data, 
                                           data_tracking_config.save_data)
        else:
            # Default empty tracker if no configuration provided
            self.tracker = DataTracker({}, {}, False, False)

        # Initialize accuracy object
        self.accuracy.init(y)

        # Default value if batch size is not being set
        train_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if train_steps * batch_size < len(X):
                train_steps += 1

        # Main training loop
        for epoch in range(1, epochs+1):
            # Start tracking the epoch time if required
            if data_tracking_config and (data_tracking_config.show_data or data_tracking_config.save_data):
                self.tracker.start_epoch()

            # Print each epoch number
            print('\n\n\n================================\n'
                f'Epoch: {epoch}\n'
                '================================')

            # Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):

                # If batch size is not set -
                # train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y

                # Otherwise slice a batch
                else:
                    batch_X = X[step*batch_size:(step+1)*batch_size]
                    batch_y = y[step*batch_size:(step+1)*batch_size]

                # Perform the forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(
                    output)
                accuracy = self.accuracy.calculate(predictions,
                                                   batch_y)

                # Perform backward pass
                self.backward(output, batch_y)

                # Optimize (update parameters)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                # Print a summary
                # Inside your step loop
                if not step % print_every or step == train_steps - 1:
                    print(f'Step: {step}\n'
                        f'--------------------------------\n'    
                        f'Accuracy: {accuracy:.3f}\n'
                        f'Loss: {loss:.3f}\n'
                        f'   - Data Loss: {data_loss:.3f}\n'
                        f'   - Regularization Loss: {regularization_loss:.3f}\n'
                        f'Learning Rate: {self.optimizer.current_learning_rate}\n'
                        '--------------------------------\n')

            # Track data if enabled
            if 'loss' in data_tracking_config.track_settings and data_tracking_config.track_settings['loss']:
                self.tracker.track('loss', loss)
            if 'accuracy' in data_tracking_config.track_settings and data_tracking_config.track_settings['accuracy']:
                self.tracker.track('accuracy', accuracy)
            if 'learning_rate' in data_tracking_config.track_settings and data_tracking_config.track_settings['learning_rate']:
                self.tracker.track('learning_rate', self.optimizer.current_learning_rate)

            # Check and track gradient norms if enabled
            if 'gradient_norms' in data_tracking_config.track_settings and data_tracking_config.track_settings['gradient_norms']:
                gradient_norms = [np.linalg.norm(layer.dweights) for layer in self.trainable_layers if hasattr(layer, 'dweights')]
                self.tracker.track('gradient_norms', np.mean(gradient_norms))

            # Check and track parameter statistics if enabled
            if 'param_mean' in data_tracking_config.track_settings and data_tracking_config.track_settings['param_mean']:
                param_stats = [np.mean(layer.weights) for layer in self.trainable_layers if hasattr(layer, 'weights')]
                self.tracker.track('param_mean', np.mean(param_stats))

            # Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = \
                self.loss.calculate_accumulated(
                    include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            # Print a summary
            # After completing an epoch
            print(f'========================================\n'
                f'Training Summary for Epoch {epoch}\n'
                f'Accuracy: {epoch_accuracy:.3f}\n'
                f'Loss: {epoch_loss:.3f}\n'
                f'   - Data Loss: {epoch_data_loss:.3f}\n'
                f'   - Regularization Loss: {epoch_regularization_loss:.3f}\n'
                f'Learning Rate: {self.optimizer.current_learning_rate}\n'
                '========================================')

            # If there is the validation data
            if validation_data is not None:
                # Evaluate the model:
                val_loss, val_accuracy = self.evaluate(*validation_data,
                              batch_size=batch_size)
            self.tracker.track('val_loss', val_loss)
            self.tracker.track('val_accuracy', val_accuracy)

            if data_tracking_config and (data_tracking_config.show_data or data_tracking_config.save_data):
                self.tracker.end_epoch()

        # Display and save data if flags are true
        if data_tracking_config:
            if data_tracking_config.show_data:
                self.tracker.display_graphs()
            if data_tracking_config.save_data:
                self.tracker.save_data_to_json()

    # Evaluates the model using passed-in dataset
    def evaluate(self, X_val, y_val, *, batch_size=None):

        # Default value if batch size is not being set
        validation_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated values in loss
        # and accuracy objects
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            # Otherwise slice a batch
            else:
                batch_X = X_val[
                    step*batch_size:(step+1)*batch_size
                ]
                batch_y = y_val[
                    step*batch_size:(step+1)*batch_size
                ]

            # Perform the forward pass
            output = self.forward(batch_X, training=False)

            # Calculate the loss
            self.loss.calculate(output, batch_y)

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'========================================\n'
            f'Validation Summary\n'
            f'--------------------------------\n'
            f'Accuracy: {validation_accuracy:.3f}\n'
            f'Loss: {validation_loss:.3f}\n'
            f'========================================')

        return validation_loss, validation_accuracy

    # Predicts on the samples
    def predict(self, X, *, batch_size=None):

        # Default value if batch size is not being set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            # Dividing rounds down. If there are some remaining
            # data but not a full batch, this won't include it
            # Add `1` to include this not full batch
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        # Model outputs
        output = []

        # Iterate over steps
        for step in range(prediction_steps):

            # If batch size is not set -
            # train using one step and full dataset
            if batch_size is None:
                batch_X = X

            # Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            # Perform the forward pass
            batch_output = self.forward(batch_X, training=False)

            # Append batch prediction to the list of predictions
            output.append(batch_output)

        # Stack and return results
        return np.vstack(output)

    # Performs forward pass
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        return layer.output

    # Performs backward pass

    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):

        # Create a list for parameters
        parameters = []

        # Iterable trainable layers and get their parameters
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())

        # Return a list
        return parameters

    # Updates the model with new parameters

    def set_parameters(self, parameters):

        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters,
                                        self.trainable_layers):
            layer.set_parameters(*parameter_set)

    # Saves the parameters to a file
    def save_parameters(self, path):

        # Open a file in the binary-write mode
        # and save parameters into it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Loads the weights and updates a model instance with them
    def load_parameters(self, path):

        # Open file in the binary-read mode,
        # load weights and update trainable layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    # Saves the model
    def save(self, path):

        # Make a deep copy of current model instance
        model = copy.deepcopy(self)

        # Reset accumulated values in loss and accuracy objects
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Open a file in the binary-write mode and save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    # Loads and returns a model

    @staticmethod
    def load(path):

        # Open file in the binary-read mode, load a model
        with open(path, 'rb') as f:
            model = pickle.load(f)

        # Return a model
        return model


# Data tracking utils


class DataTracker:
    def __init__(self, track_settings, graph_assignments, show_data=False, save_data=False):
        #track_settings: Dictionary specifying which metrics to track.
        #graph_assignments: Dictionary specifying graph number assignments for each metric.
        #show_data: Flag to display graphs at the end of training.
        #save_data: Flag to save data to a JSON file.
        self.track_settings = track_settings
        self.graph_assignments = graph_assignments
        self.show_data = show_data
        self.save_data = save_data
        self.data = {metric: [] for metric in track_settings if track_settings[metric]}
        self.start_time = None

    def track(self, metric, value):
        #Track a metric if it's enabled
        if self.track_settings.get(metric, False):
            self.data[metric].append(value)

    def start_epoch(self):
        #Record the start time of an epoch.
        if self.track_settings.get("training_time", False):
            self.start_time = time.time()

    def end_epoch(self):
        #Record the training time for an epoch.
        if self.track_settings.get("training_time", False) and self.start_time is not None:
            epoch_time = time.time() - self.start_time
            self.data["training_time"].append(epoch_time)
            self.start_time = None

    def display_graphs(self):
            # Display graphs for tracked metrics
            if self.show_data:
                graph_data = {}
                for metric, assignments in self.graph_assignments.items():
                    for assignment in assignments:
                        # Check if the assignment is a tuple (graph_num, scale_factor)
                        if isinstance(assignment, tuple):
                            graph_num, scale_factor = assignment
                            scaled_data = [d * scale_factor for d in self.data[metric]]
                            graph_data.setdefault(graph_num, []).append((metric, scaled_data))
                        else:
                            graph_num = assignment
                            graph_data.setdefault(graph_num, []).append((metric, self.data[metric]))
                
                for graph_num, metrics in graph_data.items():
                    plt.figure(graph_num)
                    for metric, data in metrics:
                        plt.plot(data, label=metric)
                    plt.title(f"Graph {graph_num}")
                    plt.legend()
                    plt.xlabel("Epoch")
                    plt.ylabel("Value")
                plt.show()

    def save_data_to_json(self, file_name="training_data.json"):
        #Save tracked data to a JSON file.
        if self.save_data:
            with open(file_name, "w") as file:
                json.dump(self.data, file)


# Input "layer"


class Layer_Input:

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs


# Common loss class


class Loss:

    # Regularization loss calculation
    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            # calculate only when factor greater than 0
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights *
                           layer.weights)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.biases *
                           layer.biases)

        return regularization_loss

    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values

    def calculate(self, output, y, *, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Cross-entropy loss


class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(y_pred[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / y_pred
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step


class Activation_Softmax_Loss_CategoricalCrossentropy:

    # Backward pass
    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = y_pred.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Binary cross-entropy loss


class Loss_BinaryCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate sample-wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(y_pred[0])

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        clipped_y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -(y_true / clipped_y_pred -
                         (1 - y_true) / (1 - clipped_y_pred)) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Mean Squared Error loss


class Loss_MeanSquaredError(Loss):  # L2 loss

    # Forward pass
    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        # Return losses
        return sample_losses

    # Backward pass
    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(y_pred[0])

        # Gradient on values
        self.dinputs = -2 * (y_true - y_pred) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Mean Absolute Error loss


class Loss_MeanAbsoluteError(Loss):  # L1 loss

    def forward(self, y_pred, y_true):

        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        # Return losses
        return sample_losses

    # Backward pass

    def backward(self, y_pred, y_true):

        # Number of samples
        samples = len(y_pred)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(y_pred[0])

        # Calculate gradient
        self.dinputs = -np.sign(y_true - y_pred) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


# Common accuracy class


class Accuracy:

    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):

        # Get comparison results
        comparisons = self.compare(predictions, y)

        # Calculate an accuracy
        accuracy = np.mean(comparisons)

        # Add accumulated sum of matching values and sample count
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        # Return accuracy
        return accuracy

    # Calculates accumulated accuracy
    def calculate_accumulated(self):

        # Calculate an accuracy
        accuracy = self.accumulated_sum / self.accumulated_count

        # Return accuracy
        return accuracy

    # Reset variables for accumulated accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Accuracy calculation for classification model


class Accuracy_Categorical(Accuracy):

    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary

    # No initialization is needed
    def init(self, y):
        pass

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y


# Accuracy calculation for regression model


class Accuracy_Regression(Accuracy):

    def __init__(self):
        # Create precision property
        self.precision = None

    # Calculates precision value
    # based on passed-in ground truth values
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # Compares predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision


# ReLU activation


class Activation_ReLU:
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        # Initialize an empty list for the outputs
        self.output = []

        # Process each input element
        for img in inputs:
            # Apply ReLU (max with 0) element-wise
            self.output.append(np.maximum(0, img))

    # Backward pass
    def backward(self, dvalues):
        # Initialize an empty list for dinputs
        self.dinputs = []

        # Process each element in dvalues
        for i in range(len(dvalues)):
            # Make a copy of the current set of values
            current = dvalues[i].copy()

            # Zero gradient where input values were negative
            current[self.inputs[i] <= 0] = 0

            # Append to dinputs
            self.dinputs.append(current)

# Softmax 


class Activation_Softmax:
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        try:
            if np.array(inputs).ndim <= 2:
                exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
                self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
            else:
                raise ValueError("Input has more than two dimensions")
        except:
            # Building the output array from scratch
            self.output = np.array([exp_vals / np.sum(exp_vals) for exp_vals in np.exp(inputs - np.max(inputs, axis=1, keepdims=True))])

    # Backward pass
    def backward(self, dvalues):
        # Initialize an empty list for dinputs
        self.dinputs = []

        # Process each element in dvalues
        for i in range(len(dvalues)):
            # Flatten output array
            single_output = self.output[i].reshape(-1, 1)
            
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs.append(np.dot(jacobian_matrix, dvalues[i]))

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Sigmoid activation


class Activation_Sigmoid:
    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        try:
            if np.array(inputs).ndim <= 2:
                self.output = 1 / (1 + np.exp(-inputs))
            else:
                raise ValueError("Input has more than two dimensions")
        except:
            # Building the output array from scratch
            self.output = np.array([1 / (1 + np.exp(-img)) for img in inputs])

    # Backward pass
    def backward(self, dvalues):
        # Initialize an empty list for dinputs
        self.dinputs = []

        # Process each element in dvalues
        for i in range(len(dvalues)):
            # Derivative - calculates from output of the sigmoid function
            derivative = (1 - self.output[i]) * self.output[i]
            self.dinputs.append(dvalues[i] * derivative)

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

# Linear activation


class Activation_Linear:

    # Forward pass
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # Backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
