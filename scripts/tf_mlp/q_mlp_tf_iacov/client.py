import os
import prep_iacov
import flwr as fl
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn import preprocessing
import keras
import sklearn
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data 

h = int(sys.argv[1])

hospital = pd.read_csv('lista_hospitais.csv').iloc[h][0]

dados = pd.read_csv('F_Tabela_Geral_Final.csv').query("X == @hospital")

x_train, x_test, y_train, y_test  = prep_iacov.Prep(dados).executar_prep()

lista_auc = []


# Transformacoes

#treino

le = preprocessing.LabelEncoder()

y_train = le.fit_transform(y_train)

y_train = keras.utils.to_categorical(y_train, num_classes = 2)

y_test = keras.utils.to_categorical(le.transform(y_test), num_classes = 2)

# modelo local

model_local = tf.keras.models.Sequential()
model_local.add(tf.keras.layers.InputLayer(input_shape=(24))) # input layer
model_local.add(tf.keras.layers.Dense(20, activation='relu')) # hidden layer 1
model_local.add(tf.keras.layers.Dense(10, activation='relu')) # hidden layer 2
model_local.add(tf.keras.layers.Dense(2, activation='softmax')) # output layer

model_local.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=[keras.metrics.AUC()])

model_local.fit(x_train, y_train, epochs=5, batch_size=50)

loss_local, auc_local = model_local.evaluate(x_test, y_test)

pd.DataFrame({'AUC_Local' : [auc_local], 'hospital' : [hospital]}).to_csv(hospital + '_local.csv', index=False)

# modelo

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(24))) # input layer
model.add(tf.keras.layers.Dense(20, activation='relu')) # hidden layer 1
model.add(tf.keras.layers.Dense(10, activation='relu')) # hidden layer 2
model.add(tf.keras.layers.Dense(2, activation='softmax')) # output layer

model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=[keras.metrics.AUC()])

# Define Flower client
class IacovClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)

        model.fit(x_train, y_train, epochs=1, batch_size=50)
        
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, auc = model.evaluate(x_test, y_test)

        lista_auc.append(auc)
        data_auc = pd.DataFrame({'auc' : lista_auc})
        data_auc['hospital'] = hospital

        data_auc.to_csv(hospital + '.csv', index=False)


        return loss, len(x_test), {"AUC": auc}


# Start Flower client
fl.client.start_numpy_client(server_address="[::]:8080", client=IacovClient())
