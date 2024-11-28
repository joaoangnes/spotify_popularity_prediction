import pickle
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn import pipeline
from sklearn import model_selection
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor

# Função para aplicar a transformação logarítmica nos dados
def log_transform(X):
    return np.log(X + 1e-8) # Evita log de zero adicionando uma pequena constante

# Classe para gerenciamento do modelo e seu pipeline
class ModelService:
    def __init__(self):
        # Caminho padrão para salvar o modelo
        self.model_path = 'app/models/default_model.pkl'
        
        # Carrega o pipeline e os parâmetros do modelo a partir de um arquivo salvo
        self.model_dict = self.get_model_dic('default_model')
        self.model_pipeline = self.model_dict['pipeline']
        self.model_params = None # Parâmetros do modelo (opcional)
        
        # Variável alvo e colunas descartadas durante o pré-processamento
        self.target = 'streams'
        self.drop_columns = [
            'track_name', 'artist(s)_name', 'released_year', 'in_apple_playlists', 
            'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts',
            'cover_url'
        ]

    # Método para realizar previsões nos dados fornecidos
    def predict(self, data):
        predictions = self.model_pipeline.predict(data) 
        formatted_predictions = [f"{prediction:,.0f}" for prediction in predictions]
        data['streams'] = formatted_predictions
        return data
    
    # Carrega o dicionário do modelo salvo em disco
    def get_model_dic(self, model_name):
        model_path = f'app/models/{model_name}.pkl'
        self.model_path = model_path
        
        # Tenta abrir e carregar o arquivo do modelo
        try:
            with open(model_path, 'rb') as file:
                model_dict = pickle.load(file)
        except FileNotFoundError: 
            # Retorna um dicionário vazio se o arquivo não for encontrado
            model_dict = {
                'name': None, 
                'pipeline': None, 
                'params': None
            }
        return model_dict
    
    # Salva o modelo treinado em disco
    def save_model(self):
        model_path = f'app/models/{self.model_name}.pkl'
        with open(model_path, 'wb') as file:
            # Salva o nome, o pipeline e os parâmetros do modelo
            pickle.dump({
                        'name': self.model_name,
                        'pipeline': self.model_pipeline,
                        'params': self.model_params
                    }, file)
    
    # Calcula e imprime as métricas de avaliação do modelo
    def get_metrics(self):
        r2_train = metrics.r2_score(self.y_train, self.y_pred_train)
        r2_teste = metrics.r2_score(self.y_test, self.y_pred_teste)
        self.R2_metrics = r2_teste
        
        print("R2 na base de treino:", r2_train)
        print("R2 na base de teste:", r2_teste)
    
    # Constrói o pipeline de pré-processamento dos dados
    def build_preprocessing_pipeline(self):
        # Pipeline para dados categóricos (imputação e codificação)
        cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
        
        # Pipeline para dados numéricos que precisam de transformação logarítmica
        log_pipeline = make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(log_transform, feature_names_out="one-to-one"), StandardScaler())
        
        # Pipeline padrão para dados numéricos
        default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

        # Combina os pipelines em um só
        preprocessing = ColumnTransformer([
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
            ("log", log_pipeline, ["artist_count", "acousticness_%", "instrumentalness_%", "liveness_%", "speechiness_%"]),
        ], remainder=default_num_pipeline) # Aplica o pipeline padrão ao restante das colunas
        
        return preprocessing
    
    # Limpa e prepara os dados antes do treinamento
    def clean_data(self, data):
        target = self.target
        # Remove as colunas especificadas
        data = data.drop(columns=self.drop_columns, axis=1)
        # Converte a variável alvo para numérico, tratando erros
        data[target] = pd.to_numeric(data[target], errors='coerce')
        # Remove linhas com valores nulos
        data = data.dropna()
        return data
    
    # Define os parâmetros do modelo com base no tipo de modelo
    def build_model_params(self, model_type):
        # Caso tenha sido informado parametros especificos para o treinamento do modelo, apenas retorna o que já foi inserido
        if self.model_params is not None:
            return self.model_params 
        
        # Caso não tenha sido informado os parametros, irá ser considerado parametros padrões para cada modelo
        match model_type:
            case 'RandomForestRegressor':
                model_params = {
                    'max_depth': 30, 
                    'min_samples_leaf': 4, 
                    'min_samples_split': 10, 
                    'n_estimators': 100
                }
            case 'GradientBoostingRegressor':    
                model_params = {
                    'learning_rate': 0.05, 
                    'max_depth': 3, 
                    'n_estimators': 100
                }
            case 'LightGBM':
                model_params = {
                    'learning_rate': 0.1, 
                    'max_depth': 3, 
                    'n_estimators': 50, 
                    'num_leaves': 31
                }
            case 'AdaBoostRegressor':
                model_params = {
                    'learning_rate': 0.01, 
                    'n_estimators': 100
                }
            case _:
                model_params = {}
                
        self.model_params = model_params
        return self.model_params
    
    # Constrói o modelo com base no tipo e nos parâmetros fornecidos
    def build_model(self, model_type):
        # Busca quais serão os parametros a serem considerados no treinamento
        model_params = self.build_model_params(model_type)
        
        # Realiza a instância do objeto do modelo a ser utilizado no treinamento
        match model_type:
            case 'RandomForestRegressor':
                model = RandomForestRegressor(**model_params)
            case 'GradientBoostingRegressor':    
                model = GradientBoostingRegressor(**model_params) 
            case 'LightGBM':
                model = LGBMRegressor(**model_params)
            case 'AdaBoostRegressor':
                model = AdaBoostRegressor(**model_params)
            case _:
                raise ValueError(f"Modelo desconhecido: {model_type}")
                
        return model
    
    # Constrói o pipeline completo com pré-processamento e modelo
    def build_model_pipeline(self):
        preprocessing = self.build_preprocessing_pipeline() # Pipeline de Pré Processamento dos Dados
        model = self.build_model(self.model_name) # Modelo a ser utilizado 
        
        # Combina o pré-processamento e o modelo em um único pipeline
        self.model_pipeline = pipeline.Pipeline([
            ("Pré Processamento dos Dados", preprocessing), 
            ("Modelo", model)
        ])
        
        return self.model_pipeline
    
    # Prepara os dados para o treinamento
    def prepare_data(self, data):
        data = self.clean_data(data) # Limpeza dos dados
        
        # Separa as variáveis preditoras (X) e a variável alvo (y)
        X = data.drop(columns=['streams'], axis=1)
        y = data[self.target]
        
        # Divide os dados em conjuntos de treino 70% e teste 30%
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                            test_size=0.3,
                                                                            random_state=42)
        return X_train, X_test, y_train, y_test
    
    # Treina o modelo com os dados fornecidos
    def train(self, data, model_name, params=None):
        if params is not None:
            self.model_params = params # Define parametros personalizados
            
        self.model_name = model_name # Define o nome do modelo
        X_train, X_test, y_train, y_test = self.prepare_data(data) # Prepara os dados
        
        # Salva os dados de treino e teste
        self.y_train = y_train
        self.y_test = y_test
        
        # Constrói e treina o pipeline do modelo
        model_pipeline = self.build_model_pipeline()
        model_pipeline.fit(X_train, y_train)
        
        # Realiza previsões nos dados de treino e teste
        self.y_pred_train = model_pipeline.predict(X_train)
        self.y_pred_teste = model_pipeline.predict(X_test)
        
        self.get_metrics() # Calcula e exibe as métricas de avaliação
        self.save_model() # Salva o modelo treinado