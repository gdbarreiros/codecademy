# US Medical Insurance

#%%
# Vamos começar importando a biblioteca pandas e abrindo o arquivo usando o métod read_csv()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

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

# Com base nas distribuições, é possível perceber que os BMIs são os dados que mais se aproximam de uma 
# distribuição normal (gaussiana). As idades praticamente não possuem variação, com exceção da faixa
# entre 20 a 23 anos que tem um número elevado de observações. Por fim, as cobranças e a 
# quantidade de filhos possuem uma distribuição assimétrica à direita, o que significa que a moda < mediana < média

# %%
# Seguindo algumas sugestões propostas no site do Codecademy, vamos realizar algumas análises um pouco 
# mais específicas

# Em primeiro lugar, vamos verificar como está a distribuição especial do nosso dataset
    # - Vamos ver quais são as regiões distintas

print(df_insurance['region'].unique())

# ['southwest' 'southeast' 'northwest' 'northeast']

# Vamos ver qual dessas regiões possui mais pessoas. Podemos fazer isso agrupando o conjunto de dados 
# pela coluna 'region' e depois usando a função de agregação count. Usamos um subconjunto do dataframe, 
# já que se usássemos ele inteiro as informações se repetiriam ao longo das colunas. Depois disso, renomeamos
# a coluna para se chamar 'count' e ordenamos os valores do menos para o maior

df_insurance_grouped = df_insurance[['region', 'age']].groupby(['region']).count()
df_insurance_grouped = df_insurance_grouped.rename(columns = {'age': 'count'})
df_insurance_grouped = df_insurance_grouped.sort_values(by = ['count'], ascending = False)
df_insurance_grouped

# Com isso, podemos ver que a região que possui mais observações nesse conjunto de dados é southeast, 
# com 364 observações

# %%
# Em segundo lugar, vamos avaliar qual a impacto do hábito do cigarro nos custos do seguro médico
# Vamos dividir o dataframe em fumantes x não-fumantes

df_non_smokers = df_insurance[df_insurance['smoker'] != 'yes']
df_smokers = df_insurance[df_insurance['smoker'] == 'yes']

# A quantidade de pessoas não-fumantes nesse conjunto de dados é muito maior (por volta de 5x) que a 
# quantidade de pessoas fumantes, o que pode invalidar nossas conclusões. O ideal seria termos uma 
# distribuição equivalente entre os dois conjuntos. De toda forma, vamos seguir com as análises

df_joined = pd.DataFrame()
df_joined['non_smoker_charges'] = df_non_smokers.agg({'charges': ['mean', 'median', 'min', 'max', 'skew']})
df_joined['smoker_charges'] = df_smokers.agg({'charges': ['mean', 'median', 'min', 'max', 'skew']})
df_joined['diff'] = df_joined['smoker_charges'] - df_joined['non_smoker_charges']
df_joined

# Com base nessa análise, lembrando dos apontamentos anteriores, é notável a diferença no custo do seguro 
# médico de pessoas fumantes x pessoas não-fumantes. Em média, pessoas que não fumam pagam ~U$27k a menos 
# que pessoas fumantes, um número bastante impressionante

#%%
# Além dessas análises, eu também gostaria de calcular qual o coeficiente de correlação de Pearson das variáveis
# qualitativas em relação ao custo ('charges') e também plotar um gráfico de dispersão, para visualiza-la graficamente

correlation = df_insurance.corr()
plot = sn.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
plot

# Se observarmos a última coluna do heatmap, teremos os coeficientes de correlação das variáveis quantitativas com
# a variável 'charges'. Claramente não há uma correlação forte (nem fraca) entre as variáveis, talvez uma correlação
# positiva fraca entre 'charges' e 'age' 

#%%
#Vamos plotar os gráficos de dispersão, apenas para visualizar as relações

plt.scatter(df_insurance['age'], df_insurance['charges'])
plt.title('Relação entre cobrança x idade')
plt.show()

plt.scatter(df_insurance['bmi'], df_insurance['charges'])
plt.title('Relação entre cobrança x bmi')
plt.show()

plt.scatter(df_insurance['children'], df_insurance['charges'])
plt.title('Relação entre cobrança x filhos')
plt.show()

#%%
#Vamos transformar a coluna 'smoker' em dummies (binária) e ver se conseguimos estabelecer alguma correlação

df_insurance_dummy = pd.get_dummies(df_insurance, columns = ['smoker'])
df_insurance_dummy = df_insurance_dummy.drop(columns = ['smoker_no'])
df_insurance_dummy = df_insurance_dummy.rename(columns = {'smoker_yes': 'smoker'})
df_insurance_dummy = df_insurance_dummy[['age',	'sex', 'bmi', 'children', 'region', 'smoker', 'charges']]

# Vamos calcular a correlação novamente

correlation = df_insurance_dummy.corr()
plot = sn.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
plot

# Nesse caso, conseguimos ver que a correlação entre a cobrança do seguro médico e o hábito do cigarro é forte 
# positiva, o que significa que fumar ou não possui uma forte influência no total pago pelo usuário. 

#%% 
# Vamos analisar o gráfico de dispersão

plt.scatter(df_insurance_dummy['smoker'], df_insurance['charges'])
plt.title('Relação entre cobrança x hábito do cigarro')
plt.show()

#%%
# E se combinarmos os gráficos de dispersão da idade e do cigarro?

plt.scatter(df_insurance_dummy['smoker'], df_insurance['charges'])
plt.scatter(df_insurance['age'], df_insurance['charges'], c = 'red')
plt.title('Relação entre cobrança x hábito do cigarro x idade')
plt.show()

# Esse plot não fica bom, porque a escala das idades é muito maior que a escala do cigarro. Vamos normaliza-la 
# usando a fórmula 
    # idade_normalizada = (idade - min_idade) / (max_idade - min_idade)

min_age = df_insurance_dummy['age'].min() 
max_age = df_insurance_dummy['age'].max()
df_insurance_dummy['normalized_age'] = (df_insurance_dummy['age'] - min_age) / (max_age - min_age)

#%%
# Agora podemos plotar novamente o nosso gráfico combinado

plt.scatter(df_insurance_dummy['normalized_age'], df_insurance['charges'], c = 'red')
plt.scatter(df_insurance_dummy['smoker'], df_insurance['charges'])
plt.title('Relação entre cobrança x hábito do cigarro x idade')
plt.legend(labels =['Cobrança x idade', 'Cobrança x hábito de fumar'], loc = 'upper left')
plt.ylim(0, df_insurance['charges'].max() * 1.2)
plt.show()

# Os dois combinados já apresentam um resultado bem mais interessante, onde é possível perceber 
# uma tendência no aumento do preço conforme a idade aumenta e também nos casos em que a pessoa fuma

#%%
# Vamos criar uma função que normaliza uma determinada coluna categórica de um dataframe

def normalizacao(df, coluna):
    minimum = df[coluna].min()
    maximum = df[coluna].max()
    df['normalized'] = (df[coluna] - minimum) / (maximum - minimum)
    return df

df_teste = normalizacao(df_insurance, 'age')
df_teste

# A função realmente funciona

#%%