import numpy as np
import json

# Create filters


def Create_Filters(Shapes, Low=0, High=1, Biases=False, BiasLow=-1, BiasHigh=1):

    RandomFilters = []

    if Biases == True:

        RandomBiases = np.random.uniform(BiasLow, BiasHigh, [1, len(Shapes)])

    for shape in Shapes:

        RandomFilters.append(np.random.uniform(Low, High, shape))

    if Biases == False:

        return RandomFilters

    else:

        return RandomFilters, RandomBiases

# define public hyperparamater ranges


def All_Hyperparameter_Ranges_Create():
    return {
        'Agent_Learning_Rate': (0.1, 0.0001),
        'Agent_Learning_Rate_Min': (0.001, 0.00001),
        'Agent_Learning_Rate_Decay': (0.01, 0.00001),
        'gamma': (0.99, 0.5),
        'epsilon': (1.0, 0.1),
        'epsilon_min': (0.1, 0.01),
        'epsilon_decay': (0.01, 0.0001),
        'learning_rate': (0.1, 0.0001),
        'learning_rate_decay': (0.01, 0.0001),
        'batch_size': (2048, 32),
        'MaxMemoryLength': (100000, 1000),
        'Episodes': (10000, 10),
        'Epochs': (100, 1),
    }

# Create, save, and load hyperparameter ranges


def Handle_Hyperparameter_Ranges(path='filtered_hyperparameter_ranges.json', action=None, dictionary=False, **kwargs):
    all_hyperparameter_ranges = {
        'Agent_Learning_Rate': (0.1, 0.0001),
        'Agent_Learning_Rate_Min': (0.001, 0.00001),
        'Agent_Learning_Rate_Decay': (0.01, 0.00001),
        'gamma': (0.99, 0.5),
        'epsilon': (1.0, 0.1),
        'epsilon_min': (0.1, 0.01),
        'epsilon_decay': (0.01, 0.0001),
        'learning_rate': (0.1, 0.0001),
        'learning_rate_decay': (0.01, 0.0001),
        'batch_size': (2048, 32),
        'MaxMemoryLength': (100000, 1000),
        'Episodes': (10000, 10),
        'Epochs': (100, 1),
    }

    selected_hyperparameters = kwargs.get('selected_hyperparameters', [])

    # Filter the dictionary based on the provided list of strings in kwargs
    filtered_hyperparameter_ranges = {
        key: value for key, value in all_hyperparameter_ranges.items()
        if key in selected_hyperparameters
    }

    if action == 'load':
        # Load and return the saved hyperparameters
        with open(path, "r") as infile:
            loaded_data = json.load(infile)
        loaded_hyperparameters_tuples = [
            tuple(t) for t in loaded_data['tuples']]
        loaded_hyperparameters_names = loaded_data['names']
        if dictionary:
            loaded_hyperparameters_dict = {name: tuple(t) for name, t in zip(
                loaded_hyperparameters_names, loaded_hyperparameters_tuples)}
            return loaded_hyperparameters_tuples, loaded_hyperparameters_names, loaded_hyperparameters_dict
        return loaded_hyperparameters_tuples, loaded_hyperparameters_names

    # If action is None or 'save', create the lists of filtered hyperparameter tuples and names
    filtered_hyperparameters_tuples = [
        tuple(t) for t in filtered_hyperparameter_ranges.values()]
    filtered_hyperparameters_names = list(
        filtered_hyperparameter_ranges.keys())

    if action == 'save':
        # Save the filtered hyperparameter tuples and names as a JSON file at the given 'path'
        data_to_save = {
            'tuples': [list(t) for t in filtered_hyperparameters_tuples],
            'names': filtered_hyperparameters_names
        }
        if dictionary:
            data_to_save['dictionary'] = {name: list(t) for name, t in zip(
                filtered_hyperparameters_names, filtered_hyperparameters_tuples)}
        with open(path, "w") as outfile:
            json.dump(data_to_save, outfile)

    # If action is None, return the filtered hyperparameter tuples, names, and optionally, the dictionary
    if dictionary:
        filtered_hyperparameters_dict = {name: tuple(t) for name, t in zip(
            filtered_hyperparameters_names, filtered_hyperparameters_tuples)}
        return filtered_hyperparameters_tuples, filtered_hyperparameters_names, filtered_hyperparameters_dict
    return filtered_hyperparameters_tuples, filtered_hyperparameters_names

# Order hyperparameters


def Order_Hyperparameters(path=None, **kwargs):
    all_hyperparameter_ranges = {
        'Agent_Learning_Rate': (0.1, 0.0001),
        'Agent_Learning_Rate_Min': (0.001, 0.00001),
        'Agent_Learning_Rate_Decay': (0.01, 0.00001),
        'gamma': (0.99, 0.5),
        'epsilon': (1.0, 0.1),
        'epsilon_min': (0.1, 0.01),
        'epsilon_decay': (0.01, 0.0001),
        'learning_rate': (0.1, 0.0001),
        'learning_rate_decay': (0.01, 0.0001),
        'batch_size': (2048, 32),
        'MaxMemoryLength': (100000, 1000),
        'Episodes': (10000, 10),
        'Epochs': (100, 1),
    }

    if path:
        with open(path, "r") as infile:
            data = json.load(infile)
    else:
        data = kwargs

    ordered_tuples = []
    ordered_names = []
    ordered_dicts = {}

    for key in all_hyperparameter_ranges:
        if key in data.get('names', []):
            index = data['names'].index(key)
            ordered_tuples.append(tuple(data['tuples'][index]))
            ordered_names.append(key)
        if key in data.get('dictionary', {}):
            ordered_dicts[key] = data['dictionary'][key]

    output = [ordered_tuples, ordered_names, ordered_dicts]
    return tuple(output)

# Create Kwargs


def Create_Kwargs(variable_list):
    kwargs = {}
    key_names = ['tuples', 'names', 'dictionary']
    for i, var in enumerate(variable_list):
        key = key_names[i] if i < len(key_names) else f'arg{i}'
        kwargs[key] = var
    return kwargs

# Flip bounds


def flip_bounds(HP_Range):
    flipped_HP_Range = [(upper, lower) for lower, upper in HP_Range]
    return flipped_HP_Range
