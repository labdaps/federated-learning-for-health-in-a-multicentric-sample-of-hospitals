import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    X_test, y_test = np.array(pd.read_csv('teste.csv').iloc[: , 0 : -1]) , np.array(pd.read_csv('teste.csv').iloc[: , -1] )

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        
        loss = log_loss(y_test, model.predict_proba(X_test))
        
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        #accuracy = model.score(X_test, y_test)
        
        
        return loss, {"auc": auc}

    return evaluate


# Start server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=21,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=5),
    )
