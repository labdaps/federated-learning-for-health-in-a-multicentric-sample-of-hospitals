import os
import pandas as pd
import sklearn
import sys
from sklearn.ensemble import RandomForestClassifier
import pickle
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.metrics import roc_auc_score


class RandomForestFlServer:

    def __init__(self, n_clients, n_estimators, session_id = None,
    receiving_path = '/home/murilo/mestrado/iacov_br_fl/modelagem/rf_fl_simulator/envio_server/',
    shipping_path = '/home/murilo/mestrado/iacov_br_fl/modelagem/rf_fl_simulator/envio_clients/',   
    max_depth = None, bootstrap =  True,
    ccp_alpha = 0.0, class_weight =  None, criterion = 'gini', max_features = 'auto',
    max_leaf_nodes = None, max_samples = None, min_impurity_decrease = 0.0, min_impurity_split = None,
    min_samples_leaf = 1, min_samples_split = 2, min_weight_fraction_leaf = 0.0, n_jobs =  None,
    oob_score = False, random_state = 0, verbose = 0, warm_start = False):
        
        self.session_id = session_id

        self.n_clients = n_clients
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap =  bootstrap
        self.ccp_alpha = ccp_alpha
        self.class_weight =  class_weight
        self.criterion = criterion
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.max_samples = max_samples
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.n_jobs =  n_jobs
        self.oob_score = oob_score
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

        self.receiving_path = receiving_path 
        self.shipping_path = shipping_path

        self.received_data = False
        self.received_models = False
        self.organization_start = False

    def quantity_receiver(self):        

        self.__clients_qtd = [f for f in listdir(self.receiving_path) if isfile(join(self.receiving_path, f)) if f[:3] == 'qtd']

        self.__clients_id = [i.split('.')[0].split('_')[1] for i in self.__clients_qtd]

        self.__qtd = np.array([pd.read_csv(f'{self.receiving_path}{i}').iloc[0] for i in self.__clients_qtd])

        self.qtd_p = self.__qtd / self.__qtd.sum()

        self.qtd_arvores =  np.round(self.n_estimators * self.qtd_p, decimals = 0)

        if len(self.__clients_id) == self.n_clients:

            print('All data has been received')

            self.received_data = True

        return self.received_data


    def parameter_sender(self):

        if self.received_data == True:

            # enviar hiperparametros aos clientes

            for i in range(len(self.__clients_id)):
                
                parametros_temp = {'n_estimators' :  int(self.qtd_arvores[i]),
                        'bootstrap': self.bootstrap,
            'ccp_alpha': self.ccp_alpha,
            'class_weight': self.class_weight,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'max_samples': self.max_samples,
            'min_impurity_decrease': self.min_impurity_decrease,
            'min_impurity_split': self.min_impurity_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_samples_split': self.min_samples_split,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'n_jobs': self.n_jobs,
            'oob_score': self.oob_score,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start}

                try:
                
                    with open(f"{self.shipping_path}parametros_client_{self.__clients_id[i]}.pkl", "wb") as tf:
                        pickle.dump(parametros_temp,tf)
                
                except:

                    print('Parameters not sending by error')

        else: 

            print('Data not received')

    def models_receiver(self):

        clients_model = [f for f in listdir(self.receiving_path) if isfile(join(self.receiving_path, f)) if f[:5] == 'model']

        self.clients_id_model = [i.split('.')[0].split('_')[-1] for i in clients_model]

        if len(self.clients_id_model) == self.n_clients:

            try:
                self.models = [pickle.load(open(f'{self.receiving_path}model_client_{i}.pkl', 'rb')) for i in self.__clients_id]
                
                print('All models has been received')

                self.received_models = True
            
            except:

                pass


    def global_model_sender(self, global_model):

        with open(f"{self.shipping_path}global_model.pkl", "wb") as tf:
            pickle.dump(global_model,tf)

        print('Global model has been sent')

    class GlobalModel:
    
        def __init__(self, model_list, p, threshold = 0.5):
            
            self.__model_list = model_list
            
            self.__p = p
            
            self.threshold = threshold
            
        def predict_proba(self, X, **kwargs):
            
            return np.sum([self.__p[i] * self.__model_list[i].predict_proba(X)[: , list(self.__model_list[i].classes_).index(1)] for i in range(len(self.__model_list))], axis = 0)
        
        def predict(self, X, **kwargs):
            
            return np.where(self.predict_proba(X) > self.threshold, 1, 0)

    def organization(self):

        auc_global_list = [i for i in os.listdir(self.receiving_path) if i[:10] == 'auc_global']

        if len(auc_global_list) == len(self.clients_id_model):

            self.organization_start = True
            
            new_dir = f'{self.receiving_path}session_id_{self.session_id}'
            
            try:
                os.system(f"rm -rf {new_dir}")
            except:
                pass

            os.mkdir(new_dir)

            print(f'A new directory has been created (session_id {self.session_id})')

            pd.concat(pd.read_csv(f'{self.receiving_path}{i}').T for i in auc_global_list).reset_index().rename(columns = {0 : 'auc', 'index' : 'client'}).to_csv(f'{new_dir}/auc_global.csv', index = False)
            pd.concat(pd.read_csv(f'{self.receiving_path}{i}').T for i in [j for j in os.listdir(self.receiving_path) if j[:10] == 'auc_modelo']).reset_index().rename(columns = {0 : 'auc', 'index' : 'client'}).to_csv(f'{new_dir}/auc_modelo.csv', index = False)
            pd.concat(pd.read_csv(f'{self.receiving_path}{i}').T for i in [j for j in os.listdir(self.receiving_path) if j[:3] == 'qtd']).reset_index().rename(columns = {0 : 'qtd', 'index' : 'client'}).to_csv(f'{new_dir}/qtd.csv', index = False)

            with open(f"{new_dir}/global_model.pkl", "wb") as tf:
                pickle.dump(self.modelo_global,tf)
            
            for f in [i for i in os.listdir(self.receiving_path) if i[:7] != 'session']:

                os.remove(os.path.join(self.receiving_path, f))

            for f in os.listdir(self.shipping_path):

                os.remove(os.path.join(self.shipping_path, f))

    
    def run(self):

        while self.received_data == False:
            self.quantity_receiver()

        self.parameter_sender()

        while self.received_models == False:
            self.models_receiver()

        self.modelo_global = self.GlobalModel(self.models, self.qtd_p)

        self.global_model_sender(self.modelo_global)

        while self.organization_start == False:
            self.organization()


        return "Successfully executed"



class GlobalModel:
    
    def __init__(self, model_list, p, threshold = 0.5):
        
        self.__model_list = model_list
        
        self.__p = p
        
        self.threshold = threshold
        
    def predict_proba(self, X, **kwargs):
        
        return np.sum([self.__p[i] * self.__model_list[i].predict_proba(X)[: , list(self.__model_list[i].classes_).index(1)] for i in range(len(self.__model_list))], axis = 0)
    
    def predict(self, X, **kwargs):
        
        return np.where(self.predict_proba(X) > self.threshold, 1, 0)


class RandomForestFlClient:

    def __init__(self, h, model = RandomForestClassifier(), 
    shipping_path = '/home/murilo/mestrado/iacov_br_fl/modelagem/rf_fl_simulator/envio_server/',
    receiving_path = '/home/murilo/mestrado/iacov_br_fl/modelagem/rf_fl_simulator/envio_clients/',   
   ):
        
        self.h = h

        self.model = model

        self.receiving_path = receiving_path 
        self.shipping_path = shipping_path

        self.sent_data = False
        self.parameter_received = False
        self.sent_model = False
        self.global_model_received = False

    def data_sender(self, X):

        pd.DataFrame({f'qtd_{self.h}' : [X.shape[0]]}).to_csv(f'{self.shipping_path}qtd_{self.h}.csv', index = False)

        self.sent_data = True

    def parameter_receiver(self):

        try:

            with open(f"{self.receiving_path}parametros_client_{self.h}.pkl", "rb") as tf:
                self.parametros = pickle.load(tf)

            self.parameter_received = True

            return self.parametros

        except:

            pass
    
    def model_sender(self, x_train, y_train):

        self.model.set_params(**self.parameter_receiver())

        self.model.fit(x_train, y_train)

        with open(f'{self.shipping_path}model_client_{self.h}.pkl', 'wb') as fp:
            pickle.dump(self.model, fp)

        self.sent_model = True

    def auc_sent(self, x_test, y_test):

        if self.sent_model == True:

            pd.DataFrame({f'auc_modelo_local_{self.h}' : [roc_auc_score(y_test, self.model.predict_proba(x_test)[:, 1])]}).to_csv(f'{self.shipping_path}auc_modelo_local_{self.h}.csv', index = False)
            
            print('AUC in the submitted model test set:', roc_auc_score(y_test, self.model.predict_proba(x_test)[:, 1]))

        else:

            print('Model not sent')
    
    def global_model_receiver(self):

        try:

            with open(f"{self.receiving_path}global_model.pkl", "rb") as tf:
                self.global_model = pickle.load(tf)

            self.global_model_received = True

            return self.global_model

        except:

            pass
    
    class GlobalModel:
    
        def __init__(self, model_list, p, threshold = 0.5):
            
            self.__model_list = model_list
            
            self.__p = p
            
            self.threshold = threshold
            
        def predict_proba(self, X, **kwargs):
            
            return np.sum([self.__p[i] * self.__model_list[i].predict_proba(X)[: , list(self.__model_list[i].classes_).index(1)] for i in range(len(self.__model_list))], axis = 0)
        
        def predict(self, X, **kwargs):
            
            return np.where(self.predict_proba(X) > self.threshold, 1, 0)

    def auc_received(self, x_test, y_test, modelo_global):

        pd.DataFrame({f'auc_modelo_global_{self.h}' : [roc_auc_score(y_test, modelo_global.predict_proba(x_test))]}).to_csv(f'{self.shipping_path}auc_global_local_{self.h}.csv', index=False)

        print('AUC in the global model test set:', roc_auc_score(y_test, modelo_global.predict_proba(x_test)))  


    def run(self, dados, x_train, x_test, y_train, y_test):
    
        self.data_sender(dados)

        while self.parameter_received == False:
            self.parameter_receiver()

        self.model_sender(x_train, y_train)

        self.auc_sent(x_test, y_test)

        while self.global_model_received == False:
            modelo_global = self.global_model_receiver()

        self.auc_received(x_test, y_test, modelo_global)




    

    




