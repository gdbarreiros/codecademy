# US Medical Insurance

#%%
# Vamos começar importando a biblioteca pandas e abrindo o arquivo usando o métod read_csv()

import pandas as pd
import matplotlib.pyplot as plt
df_insurance = pd.read_csv('insurance.csv')

#%%
# Verificando as 10 primeiras e as 10 últimas linhas do conjunto de dados

df_insurance.head(10)
df_insurance.tail(10)

# As informações contidas no conjunto de dados são: 
    # - age	
    # - sex	
    # - bmi	
    # - children	
    # - smoker	
    # - region	
    # - charges

# Vamos calcular algumas estatísticas descritivas (média, mediana, máximo, mínimo, assimetria), 
# ao menos para as colunas quantitativas

df_insurance.agg({'age': ['mean', 'median', 'min', 'max', 'skew']
                  ,'bmi': ['mean', 'median', 'min', 'max', 'skew']
                  ,'charges': ['mean', 'median', 'min', 'max', 'skew']})

#%%
# Vamos plotar a distribuição de algumas variáveis, para ver o seu comportamento

plt.hist(df_insurance['age'], bins = 15, color = 'blue')
plt.title('Distribuição das idades')
plt.show()
plt.hist(df_insurance['bmi'], bins = 10, color = 'gray')
plt.title('Distribuição dos BMIs (IMCs)')
plt.show()
plt.hist(df_insurance['charges'], bins = 10, color = 'green')
plt.title('Distribuição do valor das cobranças')
plt.show()
# %%
