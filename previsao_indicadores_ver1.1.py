import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.preprocessing  import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

def configuraConjuntoInicial(caminhoArquivo=None,colunas=None):
    if caminhoArquivo is None:
        caminhoArquivo = 'DECFEC_light_2000_2017.csv'
    dataSet = pd.read_csv(caminhoArquivo)
    # mes_normalizado = np.array([float(i.split("-")[1])/12 for i in dataSet.index])
    # dataSet['Data'] = pd.to_datetime(dataSet["Data"])
    dataSet = dataSet.set_index(["Data"])
    if colunas is not None:
        dataSet = dataSet[colunas]
    return dataSet

def separaConjuntosDeTreinoETeste(conjunto, dataReferencia=None):
    
    if dataReferencia is None:
        dataReferencia = '2016-01-01'
    conjuntoTreino = conjunto[:dataReferencia]
    conjuntoTreino = conjuntoTreino[:-1]
    conjuntoTeste = conjunto[dataReferencia:]
    return conjuntoTreino,conjuntoTeste

def normalizaConjunto(conjunto):
    
    scaler = MinMaxScaler()
    colunasConjunto = [i for i in conjunto]
    conjuntoNormalizado = scaler.fit_transform(pd.DataFrame(conjunto))
    conjuntoNormalizado  = pd.DataFrame(conjuntoNormalizado, index=conjunto.index,columns=colunasConjunto)
    return conjuntoNormalizado,scaler

def codificaMesesDoConjunto(conjunto):
    mesCodificado = np.array([float(i.split("-")[1])/12 for i in conjunto.index])
    return mesCodificado

def configuraConjuntosDeEntradaEDeReferencia(conjuntoDeDados, janelasDeObservacao=None,atraso=1, colunaAdicional=None,labelColunaAdicional=None):
    
    if janelasDeObservacao is None:
        janelasDeObservacao = 12
    dataSetDeEntrada = []
    dataSetDeReferencia = []
    
    for ii in range(len(conjuntoDeDados) - janelasDeObservacao):
        janelaDeDados = conjuntoDeDados[ii:ii+ janelasDeObservacao]
        valorDeReferencia = conjuntoDeDados[ii+janelasDeObservacao]
        dataSetDeEntrada.append(janelaDeDados)
        dataSetDeReferencia.append(valorDeReferencia)
    
    dataSetDeEntrada = np.array(dataSetDeEntrada)
    dataSetDeReferencia = np.array(dataSetDeReferencia)
    conjuntoDeEntrada = pd.DataFrame(dataSetDeEntrada)
    conjuntoDeReferencia = pd.DataFrame(dataSetDeReferencia)
    
    if colunaAdicional is not None and labelColunaAdicional is not None:
        conjuntoDeEntrada[labelColunaAdicional] = colunaAdicional[janelasDeObservacao:]
    return conjuntoDeEntrada, conjuntoDeReferencia

def configuraRedeMLP(conjuntoEntrada,conjuntoSaida,neuronios_1=None,neuronios_2=None,showPlot=False):
    
    dimEntrada = len([i for i in conjuntoEntrada])
    conjuntoEntrada = conjuntoEntrada.values
    conjuntoSaida = conjuntoSaida.values
    
    earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 100, verbose = 1)
    K.clear_session()
    model = Sequential()
    if neuronios_1 is None:
        neuronios_1 = 12
    model.add(Dense(neuronios_1,  input_shape=(dimEntrada,),activation='relu',kernel_initializer ='random_normal'))
    
    if neuronios_2 is not None:
        model.add(Dense(neuronios_2, input_shape=(neuronios_1,), activation='relu'))
    
    # model.add(Dense(3, input_shape=(6,), activation='tanh'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=0.0001), loss='mean_squared_error',metrics=['mae'])
    historico = model.fit(conjuntoEntrada , conjuntoSaida,
                validation_split=0.20, epochs=1000,
                callbacks=[earlyStopping], verbose=0)
    if(showPlot):
        model.summary()
        plt.plot(historico.history['mean_absolute_error'])
        plt.plot(historico.history['val_loss'])
        plt.legend(['mean_absolute_error','val_loss'])
        plt.grid(True)
        plt.show()
    return model,historico

def executaPrevisaoDoModelo(modelo,conjuntoEntrada,conjuntoSaida=None,scalerSaida=None,showPlot=False):
    
    previsaoDeDados = modelo.predict(conjuntoEntrada)
    if(showPlot and scalerSaida is not None and conjuntoSaida is not None):
        dadosReais = scalerSaida.inverse_transform(conjuntoSaida)
        previsaoDeDados = scalerSaida.inverse_transform(previsaoDeDados)
        plt.plot(dadosReais)
        plt.plot(previsaoDeDados)
        plt.legend(['Real','Previsão'])
        plt.grid(True)
        plt.show()
        return previsaoDeDados,dadosReais
    return previsaoDeDados


def calculaErros(previsao,verdadeiro):
    rmse = (mean_squared_error(verdadeiro, previsao))**0.5
    mape = np.mean(np.abs((verdadeiro - previsao) / verdadeiro)) * 100
    print('Root Mean Squared Error: %f'%rmse)
    print('Mean Absolute Percentage Error: %f'%mape)
    return rmse,mape


def executaPrevisaoEmMultiEstagios(modelo:Sequential,conjuntoEntradaInicial,numeroEstagios,conjuntoSaida=None,scalerSaida=None,showPlot=False):
    
    dadosDeEntrada = conjuntoEntradaInicial.iloc[0:1]
    labels = [i for i in conjuntoEntradaInicial]
    previsaoFinal = []
    for i in range(numeroEstagios):
        previsao = executaPrevisaoDoModelo(modelo=modelo,conjuntoEntrada=dadosDeEntrada)
        previsao = previsao.flatten().tolist()[0]
        previsaoFinal.append(previsao)
        if i == (numeroEstagios-1):
            continue
        proximaEntrada = dadosDeEntrada.drop(columns='mes_codificado').values.flatten().tolist()
        proximaEntrada.append(previsao)
        proximaEntrada.append(conjuntoEntradaInicial['mes_codificado'].iloc[i+1])
        proximaEntrada.pop(0)
        dataFrameDeEntrada = []
        dataFrameDeEntrada.append(proximaEntrada)
        dadosDeEntrada = pd.DataFrame(dataFrameDeEntrada,columns=labels)
    
    previsaoFinal = pd.DataFrame(np.array(previsaoFinal))
    
    if(showPlot and scalerSaida is not None and conjuntoSaida is not None):
        dadosReais = scalerSaida.inverse_transform(conjuntoSaida)
        previsaoDeDados = scalerSaida.inverse_transform(previsaoFinal)
        plt.plot(dadosReais)
        plt.plot(previsaoDeDados)
        plt.legend(['Real','Previsão'])
        plt.grid(True)
        plt.show()
        return previsaoDeDados,dadosReais
    return previsaoFinal

### Execução do codigo de fato:
indicador = 'FEC'
conjunto = configuraConjuntoInicial(colunas=[indicador])
treino,teste = separaConjuntosDeTreinoETeste(conjunto=conjunto,dataReferencia='2013-01-01')
treinoNormalizado,scalerTreino = normalizaConjunto(conjunto=treino)
testeNormalizado,scalerTeste = normalizaConjunto(conjunto=teste)
mesCodificadoTreino = codificaMesesDoConjunto(treinoNormalizado)
mesCodificadoTeste = codificaMesesDoConjunto(testeNormalizado)
# treinoInput,treinoOutput = configuraConjuntosDeEntradaEDeReferencia(conjuntoDeDados=np.ndarray.flatten(treino.values),janelasDeObservacao=5,colunaAdicional=mesCodificadoTreino, labelColunaAdicional='mes_codificado')
# testeInput,testeOutput = configuraConjuntosDeEntradaEDeReferencia(conjuntoDeDados=np.ndarray.flatten(teste.values),janelasDeObservacao=5,colunaAdicional=mesCodificadoTeste, labelColunaAdicional='mes_codificado')
treinoInput,treinoOutput = configuraConjuntosDeEntradaEDeReferencia(conjuntoDeDados=np.ndarray.flatten(treinoNormalizado.values),janelasDeObservacao=12,atraso=1,colunaAdicional=mesCodificadoTreino, labelColunaAdicional='mes_codificado')
testeInput,testeOutput = configuraConjuntosDeEntradaEDeReferencia(conjuntoDeDados=np.ndarray.flatten(testeNormalizado.values),janelasDeObservacao=12,atraso=1,colunaAdicional=mesCodificadoTeste, labelColunaAdicional='mes_codificado')

modelo = Sequential()
modelo,historico = configuraRedeMLP(conjuntoEntrada=treinoInput,conjuntoSaida=treinoOutput,neuronios_1=12,neuronios_2=None,showPlot=True)

prev,real = executaPrevisaoEmMultiEstagios(modelo=modelo,conjuntoEntradaInicial=testeInput,numeroEstagios=testeOutput.size,conjuntoSaida=testeOutput,scalerSaida=scalerTeste,showPlot=True)
calculaErros(previsao=prev,verdadeiro=real)

# ## Previsão dos dados de treino:
# previsaoTreino, dadosReaisTreino = executaPrevisaoDoModelo(modelo=modelo,conjuntoEntrada=treinoInput,conjuntoSaida=treinoOutput,scalerSaida=scalerTreino,showPlot=True)
# calculaErros(previsao=previsaoTreino,verdadeiro=dadosReaisTreino)

# ## Previsão dos dados de teste:
# previsaoTeste, dadosReaisTeste = executaPrevisaoDoModelo(modelo=modelo,conjuntoEntrada=testeInput,conjuntoSaida=testeOutput,scalerSaida=scalerTeste,showPlot=True)
# calculaErros(previsao=previsaoTeste,verdadeiro=dadosReaisTeste)

conjunto = conjunto.set_index(pd.to_datetime(conjunto.index),drop=True)
previsao_final = pd.DataFrame(prev, columns=['Previsão do FEC'], index= pd.to_datetime(teste.iloc[-len(prev):].index))

ax = conjunto.plot(y=indicador,figsize=(12,7))
previsao_final.plot(y='Previsão do FEC',ax=ax)
plt.title('Resultado da previsão')
plt.grid(True)
plt.show()


    



