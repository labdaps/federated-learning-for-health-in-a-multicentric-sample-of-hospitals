import rf_fl_labdaps_simulator
import sys
import pandas as pd
import prep_iacov

h = int(sys.argv[1])

hospital = pd.read_csv('lista_hospitais.csv').iloc[h][0]
dados = pd.read_csv('F_Tabela_Geral_Final.csv').query("X == @hospital")

x_train, x_test, y_train, y_test  = prep_iacov.Prep(dados).executar_prep()

print('Qtde.: ', dados.shape[0])

executor = rf_fl_labdaps_simulator.RandomForestFlClient(h)

executor.run(dados, x_train, x_test, y_train, y_test)




