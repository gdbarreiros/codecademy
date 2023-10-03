# US Medical Insurance

#%%
# Vamos começar importando a biblioteca pandas e abrindo o arquivo usando o métod read_csv()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df_insurance = pd.read_csv('insurance.csv')

#%%
# Verificando as 10 primeiras linhas do conjunto de dados

df_insurance.head(10)

#%%
# Verificando as 10 últimas linhas do conjunto de dados

df_insurance.tail(10)

# As informações contidas no conjunto de dados são: 
    # - age	
    # - sex	
    # - bmi	
    # - children	
    # - smoker	
    # - region	
    # - charges

#%%
# Vamos calcular algumas estatísticas descritivas (média, mediana, máximo, mínimo, assimetria), 
# ao menos para as colunas quantitativas

df_insurance.agg({'age': ['mean', 'median', 'min', 'max', 'skew']
                  ,'bmi': ['mean', 'median', 'min', 'max', 'skew']
                  ,'charges': ['mean', 'median', 'min', 'max', 'skew']
                  ,'children': ['mean', 'median', 'min', 'max', 'skew']})

#%%
# Vamos plotar a distribuição de algumas variáveis, para ver o seu comportamento
# Usamos o método axvline() para incluir uma linha horizontal no gráfico, cuja posição é a mediana 
# - que foi calculada usando o método quantile() da biblioteca numpy 

plt.hist(df_insurance['age'], bins = 15, color = 'blue')
plt.title('Distribuição das idades')
plt.axvline(x = np.quantile(df_insurance['age'], 0.5), color = 'black', label = '2º Quartil (Mediana)')
plt.legend()
plt.show()

plt.hist(df_insurance['bmi'], bins = 10, color = 'gray')
plt.title('Distribuição dos BMIs (IMCs)')
plt.axvline(x = np.quantile(df_insurance['bmi'], 0.5), color = 'black', label = '2º Quartil (Mediana)')
plt.legend()
plt.show()

plt.hist(df_insurance['charges'], bins = 10, color = 'green')
plt.title('Distribuição do valor das cobranças')
plt.axvline(x = np.quantile(df_insurance['charges'], 0.5), color = 'black', label = '2º Quartil (Mediana)')
plt.legend()
plt.show()

plt.hist(df_insurance['children'], bins = 5, color = 'red')
plt.title('Distribuição da quantidade de filhos')
plt.axvline(x = np.quantile(df_insurance['children'], 0.5), color = 'black', label = '2º Quartil (Mediana)')
plt.legend()
plt.show()

# Com base nas distribuições, é possível que os BMIs são os dados que mais se aproximam de uma 
# distribuição normal (gaussiana). As idades praticamente não possuem variação, com exceção da faixa
# entre 20 a 23 anos que tem um número elevado de observações. Por fim, as cobranças e a 
# quantidade de filhos possuem uma distribuição assimétrica à direita, o que significa que a moda < mediana < média
# %%
