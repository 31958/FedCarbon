import os
import flwr.dataset.utils.common
import time
from multiprocessing import Process
from typing import Tuple
import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.common import Weights, parameter
import flwr.server.strategy.fedadam

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def start_server(num_rounds: int, fit_clients: int, test_clients: int, fraction_fit: float):
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None, include_top=True)

    weights: Weights = model.weights

    strategy = fl.server.strategy.FedAdam(
        #server learning rate
        eta=0.01,
        #client learning rate
        eta_l=0.316,
        #degree of adaptability
        tau=0.001,
        #moment
        beta_1=0.9,
        #second moment
        beta_2=0.99,
        min_fit_clients = fit_clients,
        min_eval_clients = test_clients,
        min_available_clients = fit_clients + test_clients,
        fraction_fit = fraction_fit,
        initial_parameters=parameter.weights_to_parameters(weights)
    )
    fl.server.start_server(strategy=strategy, config={"num_rounds": num_rounds},  server_address="localhost:8080")

def start_client(dataset: Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> None:
    """Start a single client with the provided dataset."""

    # Load and compile a Keras model for CIFAR-10
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Unpack the CIFAR-10 dataset partition
    (x_train, y_train), (x_test, y_test) = dataset

    # Define a Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            """Return current weights."""
            return model.get_weights()

        def fit(self, parameters, config):
            """Fit model and return new weights as well as number of training
            examples."""
            model.set_weights(parameters)
            # Remove steps_per_epoch if you want to train over the full dataset
            # https://keras.io/api/models/model_training_apis/#fit-method
            model.fit(x_train, y_train, epochs=1, batch_size=20)
            return model.get_weights(), len(x_train), {}

        def evaluate(self, parameters, config):
            """Evaluate using provided parameters."""
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test, batch_size=20)
            return loss, len(x_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client("localhost:8080", client=CifarClient())

def run_simulation(num_rounds: int, fit_clients: int, test_clients, fraction_fit: float):
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        target=start_server, args=(num_rounds, fit_clients, test_clients, fraction_fit)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    train, test = tf.keras.datasets.cifar10.load_data()

    n = fit_clients + test_clients
    samples = 100 * n

    train = (train[0][:n * 100],train[1][:samples])
    test = (test[0][:samples],test[1][:samples])

    # Load the dataset partitions
    train_partitions, pdf = flwr.dataset.utils.common.create_lda_partitions(dataset = train, num_partitions=n, concentration=0.1)
    test_partitions, pdf = flwr.dataset.utils.common.create_lda_partitions(dataset = test, num_partitions=n,concentration=0.1)

    # Start all the clients
    for i in range (0, n):
        client_process = Process(target=start_client, args=((train_partitions[i],test_partitions[i]),))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    run_simulation(num_rounds=100, fit_clients=20, test_clients=4, fraction_fit=0.5)