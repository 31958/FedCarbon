import os

import flwr as fl
import tensorflow as tf
from flwr.server.strategy import FedAvg
import flwr.dataset.utils.common
from flwr.common.typing import Scalar
from typing import Dict, Callable, Optional, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, dataset):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = dataset

        self.model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        self.model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.device = tf.device("cpu")

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def get_properties(self, ins):
        return self.properties

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.to(self.device)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=20)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.to(self.device)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, batch_size=20)
        return loss, len(self.x_test), {"accuracy": accuracy}

def lda_partition(clients, samples_per_client, dataset):
    train, test = dataset
    train = (train[0][:clients * samples_per_client], train[1][:clients * samples_per_client])
    test = (test[0][:clients * samples_per_client], test[1][:clients * samples_per_client])
    train_partitions, pdf = fl.dataset.utils.common.create_lda_partitions(dataset=train, num_partitions=clients,concentration=0.1)
    test_partitions, pdf = fl.dataset.utils.common.create_lda_partitions(dataset=test, num_partitions=clients,concentration=0.1)
    partitions = []
    for i in range(0,clients):
        partitions.append((train_partitions[i],test_partitions[i]))
    return partitions

if __name__ == "__main__":
    num_clients = 50
    dataset =  tf.keras.datasets.cifar10.load_data()
    partitions = lda_partition(num_clients, 100, dataset)

    def client_fn(cid: str):
        id = int(cid)
        return CifarRayClient(cid, partitions[id])

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources={"num_cpus": 1},
        num_rounds=5,
        strategy= FedAvg(min_available_clients = 10),
        ray_init_args={"include_dashboard": False},
    )