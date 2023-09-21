import flwr as fl


# Start server
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,  # Sample 100% of available clients for the next round
    min_fit_clients=21,  # Minimum number of clients to be sampled for the next round
    min_available_clients=21,  # Minimum number of clients that need to be connected to the server before a training round can start
)
fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)