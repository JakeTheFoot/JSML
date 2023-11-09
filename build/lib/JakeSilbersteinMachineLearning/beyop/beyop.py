from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from scipy.stats import norm
import numpy as np
import joblib
import time
import json
import time

# Bayesian Optimization


class BayesianOptimizer(ABC):
    def __init__(self, model=False):
        kernel = Matern(length_scale_bounds=(1e-06, 1.0))
        self.model = GaussianProcessRegressor(kernel=kernel)
        if model:
            joblib.dump(self.model, 'my_model.joblib')

    def set(self, model, names=None, xi=0.01, kappa=2.576):
        self.evaluation_function = model.train
        self.Model = model
        self.xi = xi
        self.kappa = kappa
        self.names = names

    def predict(self, x):
        return self.model.predict(x)

    def check_bounds(self, bounds):
        for i, (lower, upper) in enumerate(bounds):
            if lower > upper:
                print(
                    f"Bounds error: Lower bound ({lower}) is greater than upper bound ({upper}) at index {i}")
                return False
        return True

    def set_deep_q_network_params(self, param_names, param_values):
        if len(self.names) != len(param_values):
            raise ValueError(
                "Length of param_names and param_values must be equal.")

        param_dict = dict(zip(param_names, param_values))

        if 'MaxMemoryLength' in param_dict:
            param_dict['MaxMemoryLength'] = int(
                round(param_dict['MaxMemoryLength']))

        self.Model.set(**param_dict)

    def generate_initial_data(self, n_samples, bounds, names, seed=None, save_every=None, X=True, y=True, TotalSamples=0, training_iterations=0):
        if TotalSamples == 0:
            TotalSamples == n_samples
        self.names = names
        if seed is not None:
            np.random.seed(seed)

        self.n_hyperparameters = len(bounds)
        if X:
            X_original = np.random.rand(n_samples, self.n_hyperparameters)
            X_modified = X_original.copy()
            self.X = X_modified
        if y:
            self.y = []
        if X:
            for i, (lower, upper) in enumerate(bounds):
                self.X[:, i] = self.X[:, i] * (upper - lower) + lower

        for i in range(n_samples):
            print(f"Sample: {i + 1}")
            self.set_deep_q_network_params(
                self.names, self.X[i + (TotalSamples - n_samples)])
            self.y.append(self.evaluation_function(
                Plot=False, Start=True, OF=True))

            if save_every is not None and (i+1) % save_every == 0:
                self.save_progress_to_file(
                    self.y, n_samples - (i+1), training_iterations, False)

        return self.X, self.y

    def maximize(self, n_iter, X, y, bounds, save_every=None, model=False):
        if not model:
            self.model.fit(X, y)
        for i in range(n_iter):
            print(f"Point: {i + 1}")
            self.next_point = self._acquisition_function(X, y, bounds)
            self.set_deep_q_network_params(self.names, self.next_point)
            self.y_new = self.evaluation_function(
                Plot=False, Start=True, OF=True)
            self.X = np.vstack([self.X, self.next_point])
            self.y = np.append(self.y, self.y_new)
            self.model.fit(self.X, self.y)

            if save_every is not None and (i+1) % save_every == 0:
                self.save_progress_to_file(self.y, 0, n_iter - (i+1), True)

        max_index = np.argmax(y)
        return X[max_index], y[max_index]

    @abstractmethod
    def _acquisition_function(self, X, y, bounds):
        pass

    def save_progress_to_file(self, y, remaining_init_data_iters, remaining_max_iters, model=False):
        np.savetxt('hyperparameters.txt', self.X)
        np.savetxt('scores.txt', y)
        if model:
            joblib.dump(self.model, 'my_model.joblib')

        remaining_iters = {
            'remaining_init_data_iters': remaining_init_data_iters,
            'remaining_max_iters': remaining_max_iters
        }
        with open('remaining_iters.json', 'w') as f:
            json.dump(remaining_iters, f)

    def save_parameters_to_file(self, params):
        with open('train_parameters.json', 'w') as f:
            json.dump(params, f)

    def load_parameters_from_file(self):
        with open('train_parameters.json', 'r') as f:
            return json.load(f)

    def pickup_training(self, model=True):
        train_params = self.load_parameters_from_file()
        self.X = np.loadtxt('hyperparameters.txt')
        self.y = np.loadtxt('scores.txt')
        if model:
            self.model = GaussianProcessRegressor(kernel=Matern())
            self.model.fit(self.X, self.y)

        with open('remaining_iters.json', 'r') as f:
            remaining_iters = json.load(f)

        remaining_init_data_iters = remaining_iters['remaining_init_data_iters']
        remaining_max_iters = remaining_iters['remaining_max_iters']

        initial_samples = train_params['initial_samples']
        training_iterations = train_params['training_iterations']
        selected_hyperparameters = train_params['selected_hyperparameters']
        RegisterTime = train_params['RegisterTime']
        ReturnTime = train_params['ReturnTime']
        ReturnHP = train_params['ReturnHP']
        save_every = train_params['save_every']

        all_hyperparameter_ranges = All_Hyperparameter_Ranges_Create()
        filtered_hyperparameter_ranges = {
            key: value for key, value in all_hyperparameter_ranges.items()
            if key in selected_hyperparameters
        }

        HP_Ranges = [value for key,
                     value in filtered_hyperparameter_ranges.items()]
        HP_Names = [key for key in filtered_hyperparameter_ranges.keys()]
        HP_Dict = filtered_hyperparameter_ranges

        HP_Range_Flipped = flip_bounds(HP_Ranges)

        self.names = HP_Names

        if remaining_init_data_iters is not None and remaining_init_data_iters > 0:
            self.X, self.y = self.generate_initial_data(remaining_init_data_iters, HP_Ranges, self.names, save_every=save_every, X=False,
                                                        y=False, TotalSamples=initial_samples, training_iterations=training_iterations)

        if remaining_max_iters is not None and remaining_max_iters > 0:
            best_hyperparameters, best_objective = self.maximize(
                remaining_max_iters, self.X, self.y, HP_Range_Flipped, save_every=save_every, model=True)
        else:
            max_index = np.argmax(self.y)
            best_hyperparameters, best_objective = self.X[max_index], self.y[max_index]

        if ReturnHP == 3:
            if ReturnTime:
                return start, mid, end, HP_Names, HP_Ranges, HP_Dict, best_hyperparameters, best_objective
            return HP_Names, HP_Ranges, HP_Dict, best_hyperparameters, best_objective
        elif ReturnHP == 2:
            if ReturnTime:
                return start, mid, end, HP_Names, HP_Ranges, best_hyperparameters, best_objective
            return HP_Names, HP_Ranges, best_hyperparameters, best_objective
        elif ReturnHP == 1:
            if ReturnTime:
                return start, mid, end, HP_Names, best_hyperparameters, best_objective
            return HP_Names, best_hyperparameters, best_objective
        elif ReturnHP == 0:
            if ReturnTime:
                return start, mid, end, best_hyperparameters, best_objective
            return best_hyperparameters, best_objective

    def train(self, initial_samples, training_iterations, selected_hyperparameters, RegisterTime=False, ReturnTime=False, ReturnHP=1, save_every=None, TotalSamples=0):

        if not RegisterTime:
            ReturnTime = False

        all_hyperparameter_ranges = All_Hyperparameter_Ranges_Create()

        filtered_hyperparameter_ranges = {
            key: value for key, value in all_hyperparameter_ranges.items()
            if key in selected_hyperparameters
        }

        HP_Ranges = [value for key,
                     value in filtered_hyperparameter_ranges.items()]
        HP_Names = [key for key in filtered_hyperparameter_ranges.keys()]
        HP_Dict = filtered_hyperparameter_ranges

        self.names = HP_Names

        HP_Range_Flipped = flip_bounds(HP_Ranges)

        # Save the train method parameters
        train_params = {
            'initial_samples': initial_samples,
            'training_iterations': training_iterations,
            'selected_hyperparameters': selected_hyperparameters,
            'RegisterTime': RegisterTime,
            'ReturnTime': ReturnTime,
            'ReturnHP': ReturnHP,
            'save_every': save_every
        }
        self.save_parameters_to_file(train_params)

        # Generate initial data and maximize
        if RegisterTime:
            start = time.time()
        self.X, self.y = self.generate_initial_data(initial_samples, HP_Ranges, self.names, save_every=save_every, X=True,
                                                    y=True, TotalSamples=TotalSamples, training_iterations=training_iterations)
        if RegisterTime:
            mid = time.time()
        best_hyperparameters, best_objective = self.maximize(
            training_iterations, self.X, self.y, HP_Range_Flipped, save_every=save_every, model=False)
        if RegisterTime:
            end = time.time()

        if RegisterTime:
            print(
                f"Start to Mid: {mid - start}, Mid to End: {end - mid}, Full Program: {end - start}")

        if ReturnHP == 3:
            if ReturnTime:
                return start, mid, end, HP_Names, HP_Ranges, HP_Dict, best_hyperparameters, best_objective
            return HP_Names, HP_Ranges, HP_Dict, best_hyperparameters, best_objective
        elif ReturnHP == 2:
            if ReturnTime:
                return start, mid, end, HP_Names, HP_Ranges, best_hyperparameters, best_objective
            return HP_Names, HP_Ranges, best_hyperparameters, best_objective
        elif ReturnHP == 1:
            if ReturnTime:
                return start, mid, end, HP_Names, best_hyperparameters, best_objective
            return HP_Names, best_hyperparameters, best_objective
        elif ReturnHP == 0:
            if ReturnTime:
                return start, mid, end, best_hyperparameters, best_objective
            return best_hyperparameters, best_objective

# Probability improvement (PI) aquisition function


class ProbabilityImprovement(BayesianOptimizer):
    # Child class for Probability of Improvement (PI) acquisition function.
    def _acquisition_function(self, X, y, bounds):
        def neg_pi(x):
            mu, sigma = self.model.predict(x.reshape(1, -1))
            best_y = np.max(y)
            Z = (mu - best_y - self.xi) / sigma
            PI = norm.cdf(Z)
            return -PI

        res = minimize(neg_pi, X[np.argmax(y)], bounds=bounds)
        return res.x

# Expected improvement (PI) aquisition function


class ExpectedImprovement(BayesianOptimizer):
    # Child class for Expected Improvement (EI) acquisition function.
    def _acquisition_function(self, X, y, bounds):
        if not self.check_bounds(bounds):
            raise ValueError("Invalid bounds provided.")

        def neg_ei(x):
            mu, sigma = self.model.predict(x.reshape(1, -1), return_std=True)
            best_y = np.max(y)
            Z = (mu - best_y - self.xi) / sigma
            EI = (mu - best_y - self.xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -EI

        res = minimize(neg_ei, X[np.argmax(y)], bounds=bounds)
        return res.x

# Upper confidence bound (UCB) aquisition function


class UpperConfidenceBound(BayesianOptimizer):
    # Child class for Upper Confidence Bound (UCB) acquisition function.
    def _acquisition_function(self, X, y, bounds):
        def neg_ucb(x):
            mu, sigma = self.model.predict(x.reshape(1, -1))
            UCB = mu + self.kappa * sigma
            return -UCB

        res = minimize(neg_ucb, X[np.argmax(y)], bounds=bounds)
        return res.x
