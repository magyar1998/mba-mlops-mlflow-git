#!/usr/bin/env python
# coding: utf-8

## Desafio:
### Atuando como consultor para uma empresa, a mesma lhe forneceu um código legado de um projeto que não foi para frente com o time de analytics deles.
### A empresa é da área de óleo e gás e trabalha mapeando áreas com potencial para explorar.
### O projeto deles trata de tentar aumentar a granularidade(resolução) de um conjunto de dados inicial para um conjunto de dados final com "melhor resolução" que permita um mapeamento melhor.
### A empresa trabalha com um software comercial que produz resultados razoáveis, mas que é uma caixa preta e o time de negócios da empresa agora resolveu criar suas próprias soluções para ter mais controle e não precisar pagar mais a licença desse software e automatizar os processos.
### A empresa lhe forneceu os dados de treino, e os dados de inferência. Ambos em estrutura numpy array com coordenadas X,Y,Z,Propriedade(target).
### A empresa também lhe forneceu os dados do resultado gerado por eles com o software comercial, com o mesmo tipo de estrutura dos dados de treino e de inferencia, para que você compare com a solução criada por você.
### Cabe a você realizar experimentos novos que melhorem (em relação à solução do software comercial).
### Repare que a solução atual que já consta no código legado claramente apresenta artefatos estranhos, explore isso.

### Importando Bicliotecas

# In[1]:


import numpy as np
import simtoseis_library as sts
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from matplotlib.colors import TwoSlopeNorm
import mlflow


# ### Carrega os dados

# In[2]:

#Dados Treino
dados_treino = np.load("sim_slice.npy")
dados_treino


# In[3]:


dados_treino.shape


# In[4]:


#Dados para Inferência
dados_inferencia = np.load("seismic_slice.npy")
dados_inferencia


# In[5]:


dados_inferencia.shape


# In[6]:


#Dados de Referência para a modelagem(Software Comnercial)
dados_referencia_comercial = np.load("seismic_slice_GT.npy")
dados_referencia_comercial


# In[7]:


dados_referencia_comercial.shape


# ### Tratamento dos dados

# In[8]:


#Checando a quantidade original de dados
original_slice_shape = dados_treino.shape[0]
print(f"Original number of samples in simulation model: {original_slice_shape}")


# In[9]:


# Filtrando os dados
filtered_slice = dados_treino[ dados_treino[:, -1] != -99.0 ]
print(f"Final number of samples after cleaning: {filtered_slice.shape[0]}")


# In[10]:


# Calculate and report the percentage of data removed
percentage_loss = ((original_slice_shape - filtered_slice.shape[0]) / original_slice_shape) * 100
print(f"Percentage loss: {round(percentage_loss, 2)}%")


# In[11]:


dados_treino = sts.simulation_data_cleaning(simulation_data = dados_treino, value_to_clean = -99.0)


# In[12]:


dados_treino = sts.simulation_nan_treatment(simulation = dados_treino, value = 0, method = 'replace')


# ### Conversão de sinais

# In[13]:


dados_treino, dados_inferencia = sts.depth_signal_checking(simulation_data=dados_treino, seismic_data=dados_inferencia)


# ### Plotar os dados de treino

# In[14]:


sts.plot_simulation_distribution(dados_treino, bins=35, title="Distribuição da Propriedade da Simulação para os dados de Treino")


# ### Treinamento/Validação do Modelo de ML



# In[16]:

# Simulações

prop_treino = 0.75
n_estimators = 200
max_depth = 60
n_jobs = -1

dict_params = {"n_estimators":n_estimators,"max_depth":max_depth, "n_jobs":n_jobs,"proporcao_treino": prop_treino}

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id=423146738934403465)


with mlflow.start_run():

    mlflow.sklearn.autolog()


    dados_validacao, y, nrms_teste, r2_teste, mape_teste, modelo, modelo_ML, X = sts.ML_model_evaluation(dados_simulacao=dados_treino, modelo="extratrees", dict_params=dict_params)
    

    dict_metrics = {"parametros": dict_params, "nrms_teste":nrms_teste, "r2_teste":r2_teste}


    mlflow.log_metrics({"nrms_teste": nrms_teste})  


# In[18]:


dados_estimados_prop_vector, dados_estimados = sts.transfer_to_seismic_scale(dados_sismicos=dados_inferencia, nome_arquivo_segy= None, modelo=modelo_ML, X= X, y=y)

# ### Histograma dos dados de inferência

# In[19]:


sts.plot_simulation_distribution(dados_estimados, bins=35, title="Distribuição da Propriedade da Simulação para os dados Estimados")


# ### Calculo dos Residuos: Dados de Referencia(software comercial) - Dados da Inferência ML

# In[20]:


dados_estimados_residual_final = sts.residuos_calculation(dados_referencia_comercial=dados_referencia_comercial, dados_estimados = dados_estimados)

# ### Plotando resultados dos Resíduos

# In[21]:


sts.plot_simulation_distribution(dados_estimados_residual_final, bins=35, title="Distribuição da Propriedade da Simulação para os Resíduos")



# In[23]:

print(dados_treino.shape)
sts.plot_seismic_slice(dados_treino, title="Slice a ~5000m dos dados de treino")


# In[24]:

print(dados_referencia_comercial.shape)
sts.plot_seismic_slice(dados_referencia_comercial, title="Slice a ~5000m do Resultado-Referência(software comercial)")


# In[25]:

print(dados_estimados.shape)
sts.plot_seismic_slice(dados_estimados, title="Slice a ~5000m da Inferência ML")


# In[26]:


sts.plot_seismic_slice(dados_estimados_residual_final, title = "Slice a ~5000m - Residuo da Inferência")


# In[27]:


tuple_tags = ["parametros", "nrms_teste", "r2_teste", "mape_teste"]


# In[28]:


metrics_tuple = [dict_params, nrms_teste, r2_teste, mape_teste]


# In[29]:


dict_metrics = dict(zip(tuple_tags, metrics_tuple))
dict_metrics


# In[30]:

# MLFLOW Tracking









# %%
