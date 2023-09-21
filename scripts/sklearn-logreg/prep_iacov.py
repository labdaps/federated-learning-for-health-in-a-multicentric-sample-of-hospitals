#!/usr/bin/env python
# coding: utf-8

import pandas as pd
pd.set_option('display.max_columns', 500)

import numpy as np

from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler



class Prep:
    
    def __init__(self, df, 
                 numericas = ['age', 'platelets', 'hcm','mcv','leukocytes' ,'rdw','crp','basophils', 'lymphocytes','eosinophils','red_cells_count','monocytes','hemoglobin','resp_rate','neutrophil','hematocrit','heart_rate','sys_press','dias_press','mean_press','temp'],
                 categoricas = ['genero'],
                 complementares = ['obito'],
                 variaveis_serem_criadas_encoder = ['genero_F', 'genero_M', 'genero_sem_preench'],
                 ):
        
        self.df = df.rename(columns = {'origem_cidade_hosp':'city_hospital', #maintain to identify internal/external records
                        'cd_paciente': 'cd_patient',                #maintain for further duplicity check
                        'tempo_permanencia':'hospital_time',
                        'idade':'age',
                        'raca':'race',

                        'fc':'heart_rate',
                        'fr':'resp_rate',
                        'pa_sist':'sys_press',
                        'pa_diast':'dias_press',
                        'pam':'mean_press',

                        'saturacao':'saturation',
                        'peso':'weight',
                        'altura':'height',
                        'hb':'hemoglobin',
                        'plaquetas':'platelets',
                        'ht':'hematocrit',
                        'hemacias':'red_cells_count',
                        'chcm':'mchc', #mean corpuscular hemoglobin concentration
#                         'hcm':'mch', #mean corpuscular hemoglobin
                        'vcm':'mcv', #mean corpuscular volume
#                         'imc':'bmi',
                        'leucocitos':'leukocytes',
                        'neutrofilos':'neutrophil',
                        'linfocitos':'lymphocytes',
                        'razao_neut_linf':'neutr_lymph_ratio',
                        'razao_linf_pcr':'lymph_crp_ratio',
                        'basofilos':'basophils',
                        'eosinofilos':'eosinophils',
                        'monocitos':'monocytes',
                        'proteina_c_reativa':'crp',
                        'albumina':'albumin',  
                        'dhl':'ldh', #lactate dehydrogenase level
                        'tgp':'alt', #alanine aminotransferase
                        'tgo':'ast', #aspartate aminotransferase
                        'bili_direta':'direct_bilirubin',
                        'bili_total':'total_bilirubin',
                        'bili_indireta':'indirect_bilirubin',
                        'ureia':'urea',
                        'sodio':'sodium',
                        'potassio':'potassium',
                        'creatinina':'creatinine',
                        'troponina':'troponin',
                        'd_dimeros':'d_dimer',
                        'lactato_venoso':'venous_lactate',
                        'fibrinogenio':'fibrinogen',

                        'inr':'inr', #internacional normalized ratio
                        'ttpa':'aptt', #Activated Partial Thromboplastin Time
                        'lactatoarterial':'arterial_lactate',   
                        'gaso_ph':'gas_ph', #Arterial Blood Gas - ph
                        'gao_po2':'gas_pao2',
                        'gaso_pco2':'gas_paco2',
                        'gaso_hco3':'gas_hco3', #Bicarbonate - HCO3
                        'gaso_eb':'gas_be', #Base Excess
                        'gaso_so2':'gas_so2', #Base Excess
                        'magnesio':'magnesium',   
#                         'gaso_sat':'gas_so2_saturation', #Oxygen Saturation sO2
                        'calcio_ionico':'calcium_ionised',
                        'calcio_total':'total_calcium',
                        'glicose':'glucose'
                       }).reset_index(drop=True)
        
        self.numericas = numericas
        
        self.categoricas = categoricas
        
        self.complementares = complementares
        
        self.criacao_encoder = variaveis_serem_criadas_encoder
        
        self.variaveis = self.numericas + self.categoricas + self.complementares + self.criacao_encoder 
        

    def unificador_campos(self):
        
        self.df.columns = [i.lower().split('.')[0] for i in self.df.columns]

        for k in [j for j in self.variaveis if j not in self.df.columns]:

            self.df[k] = np.nan
            
        self.df = self.df[self.variaveis]

    def campo_nulo(self):

        for i in self.numericas + self.complementares:
                        
            str_temp = i + '_sem_preench'

            self.df[str_temp] = np.where(self.df[i].isnull(), 1, 0)

    def unificar_genero(self):

        self.df['genero'] = self.df['genero'].map({'Fem' : 'F', 'Masc' : 'M'})


    def criar_target(self):

        self.df['target'] = np.where(self.df['obito'].isnull(), 0, 1)


    def apagar_colunas_antes_slit(self):

        apagar_lista = self.complementares

        self.df = self.df.drop(columns = apagar_lista)

        self.df = self.df.loc[:, ~self.df.columns.str.contains('obito')]        
        
        
    def criar_treino_teste(self):
        
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.drop(columns=['target']), 
                                                                                self.df[['target']], 
                                                                                test_size=0.3, random_state=42, stratify=self.df[['target']])
    def tratar_categoticos(self):
        
        
        encoder=OneHotEncoder(sparse=False)


        df1 = self.X_train[self.categoricas].fillna('_sem_preench')

        df1 = pd.DataFrame(encoder.fit_transform(df1), columns = encoder.get_feature_names(df1.columns))

        for i in [j for j in df1.columns if j in self.variaveis]:

            self.X_train[i] = self.X_train[i]
            
        df2 = self.X_test[self.categoricas].fillna('_sem_preench')
        
        df2 = pd.DataFrame(encoder.fit_transform(df2), columns = encoder.get_feature_names(df2.columns))

        for i in [j for j in df2.columns if j in self.variaveis]:

            self.X_test[i] = self.X_test[i]


    def preencher_mediana(self):
        
        for i in self.numericas:
                    
            self.X_train[i].fillna(self.X_train[i].mean(),inplace=True)

            self.X_test[i].fillna(self.X_train[i].mean(),inplace=True)
                
    def padronizacao(self):
                
        
        scaler = StandardScaler()
                
        col_num = self.numericas

        self.X_train[col_num] = pd.DataFrame(scaler.fit_transform(self.X_train[col_num].values), 
                                    index = self.X_train.index, columns = self.X_train[col_num].columns)

        self.X_test[col_num] = pd.DataFrame(scaler.transform(self.X_test[col_num]), 
                                   index = self.X_test.index, 
                                   columns = self.X_test[col_num].columns)
        
        
        self.X_train = self.X_train.drop(columns = self.categoricas)
        self.X_test = self.X_test.drop(columns = self.categoricas)
    
                    
    def executar_prep(self):
        
        self.unificador_campos()
        #self.campo_nulo()
        self.unificar_genero()
        self.criar_target()
        self.apagar_colunas_antes_slit()
        self.criar_treino_teste()
        self.tratar_categoticos()
        self.preencher_mediana()
        self.padronizacao()
        
        return self.X_train.fillna(0), self.X_test.fillna(0), self.y_train, self.y_test