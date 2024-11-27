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

class ModelService:
    def __init__(self):
        self.model_path = None
        self.model_pipeline = self.get_model_pipeline('default_model')
        self.target = 'streams'
        self.drop_columns = [
            'track_name', 'artist(s)_name', 'released_year', 'in_apple_playlists', 
            'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts',
            'cover_url'
        ]

    def predict(self, data):
        predictions = self.model_pipeline.predict(data)
        formatted_predictions = [f"{prediction:,.0f}" for prediction in predictions]
        data['streams'] = formatted_predictions
        return data
    
    def get_model_pipeline(self, model_name):
        model_path = f'../app/models/{model_name}.pkl'
        self.model_path = model_path
        try:
            with open(model_path, 'rb') as file:
                model_pipeline = pickle.load(file)
        except FileNotFoundError:
            model_pipeline = None
        return model_pipeline
    
    def save_model(self):
        model = self.model_pipeline.named_steps['Modelo']
        with open(self.model_path, 'wb') as file:
            pickle.dump(model, file)
    
    def get_metrics(self):
        r2_train = metrics.r2_score(self.y_train, self.y_pred_train)
        r2_teste = metrics.r2_score(self.y_test, self.y_pred_teste)
        self.R2_metrics = r2_teste
        
        print("R2 na base de treino:", r2_train)
        print("R2 na base de teste:", r2_teste)
        
    def build_preprocessing_pipeline(self):
        def log_transform(X):
            return np.log(X + 1e-8)

        cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
        log_pipeline = make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(log_transform, feature_names_out="one-to-one"), StandardScaler())
        default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

        preprocessing = ColumnTransformer([
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
            ("log", log_pipeline, ["artist_count", "acousticness_%", "instrumentalness_%", "liveness_%", "speechiness_%"]),
        ], remainder=default_num_pipeline)
        
        return preprocessing
    
    def clean_data(self, data):
        target = self.target
        data = data.drop(columns=self.drop_columns, axis=1)
        data[target] = pd.to_numeric(data[target], errors='coerce')
        data = data.dropna()
        return data
    
    def build_model_params(self, model_type):
        match model_type:
            case 'RandomForest':
                model_params = {
                    'max_depth': None, 
                    'min_samples_leaf': 4, 
                    'min_samples_split': 10, 
                    'n_estimators': 100
                }
            case 'GradientBoosting':    
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
            case 'AdaBoost':
                model_params = {
                    'learning_rate': 0.01, 
                    'n_estimators': 100
                }
            case _:
                model_params = {}
                
        self.model_params = model_params
        return model_params
    
    def build_model(self, model_type):
        model_params = self.build_model_params(model_type)
        
        match model_type:
            case 'RandomForest':
                model = RandomForestRegressor(**model_params)
            case 'GradientBoosting':    
                model = GradientBoostingRegressor(**model_params) 
            case 'LightGBM':
                model = LGBMRegressor(**model_params)
            case 'AdaBoost':
                model = AdaBoostRegressor(**model_params)
            case _:
                raise ValueError(f"Modelo desconhecido: {model_type}")
                
        return model
    
    def build_model_pipeline(self):
        preprocessing = self.build_preprocessing_pipeline()
        # model = self.build_model('RandomForest')    
        model = self.build_model('GradientBoosting')    
        
        model_pipeline = pipeline.Pipeline([
            ("Pré Processamento dos Dados", preprocessing), 
            ("Modelo", model)
        ])
        self.model_pipeline = model_pipeline
        return model_pipeline
    
    def prepare_data(self, data):
        data = self.clean_data(data)
        
        X = data.drop(columns=['streams'], axis=1)
        y = data[self.target]
        
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                            test_size=0.2,
                                                                            random_state=42)
        return X_train, X_test, y_train, y_test
            
    def train(self, data):
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        self.y_train = y_train
        self.y_test = y_test
        
        model_pipeline = self.build_model_pipeline()
        model_pipeline.fit(X_train, y_train)
        
        self.y_pred_train = model_pipeline.predict(X_train)
        self.y_pred_teste = model_pipeline.predict(X_test)
        
        self.get_metrics()
        self.save_model()