import flwr as fl

strategy=fl.server.strategy.FedAvg()
fl.server.start_server(strategy=strategy,config={"num_rounds": 10}, server_address="localhost:8080")