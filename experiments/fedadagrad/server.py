import flwr as fl
import flwr.common.parameter as p
from flwr.common import Weights
import tensorflow as tf

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None, include_top = True)

weights:Weights = model.weights

strategy=fl.server.strategy.FedAdagrad(
    eta=0.1,
    eta_l=0.316,
    tau=0.5,
    initial_parameters=p.weights_to_parameters(weights)
)
fl.server.start_server(config={"num_rounds": 3}, server_address="localhost:8080", strategy=strategy)