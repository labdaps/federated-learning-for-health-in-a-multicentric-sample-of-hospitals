import warnings
import flwr as fl
import numpy as np
import prep_iacov
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import sys

import utils

lista_auc = []
lista_modelo = []



if __name__ == "__main__":

# Load model and data 

    h = int(sys.argv[1])

    hospital = pd.read_csv('lista_hospitais.csv').iloc[h][0]

    dados = pd.read_csv('F_Tabela_Geral_Final.csv').query("X == @hospital")

    X_train, X_test, y_train, y_test  = prep_iacov.Prep(dados).executar_prep()

    # Local model

    model_local = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

    model_local.fit(X_train, y_train)


    auc_local = roc_auc_score(y_test, model_local.predict_proba(X_test)[:, 1])

    pd.DataFrame({'auc_local' : [auc_local], 'hospital' : [hospital]}).to_csv(hospital + '_local.csv', index=False)

    # End Local Model

    lista_auc = []

    # Split train set into 5 partitions and randomly use one for training.
    partition_id = np.random.choice(5)
    (X_train, y_train) = utils.partition(X_train, y_train, 5)[partition_id]

    # Create LogisticRegression Model
    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting
    )

   
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class IacovClient(fl.client.NumPyClient):
        def get_parameters(self, config):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['server_round']}")
            
            lista_modelo.append(model.coef_)

            print(lista_modelo)

            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            #accuracy = model.score(X_test, y_test)
            
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            
            lista_auc.append(auc)
            data_auc = pd.DataFrame({'auc' : lista_auc})
            data_auc['hospital'] = hospital

            data_auc.to_csv(hospital + '.csv', index=False)


            return loss, len(X_test), {"auc": auc}

    # Start Flower client
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=IacovClient())
