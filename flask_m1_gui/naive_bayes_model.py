import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class BayesModel:
    def __init__(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self._process_data()
        self.model = self._train_model()

    def _process_data(self):
        #Load filtered_data 
        df = pd.read_csv('filtered_data.csv')

        #Select features & target
        features = ['LVEF','creatinine.enzymatic.method']
        X = df[features]
        y = df['death']

        #Split
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

        return X_train, X_test, y_train, y_test

    def _train_model(self):
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)

        return model
    
    def load_model(self):
        return self.model
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
