import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from joblib import load
from rich.console import Console
import time

console=Console()
scaler = StandardScaler()
iris = load_iris()
X = iris.data
y = iris.target
scaler.fit(X)

global model
flowertypes={0:'Setosa',1:'Versicolor',2:'Virignica'}

def get_name():
    console.print("Give your model file name: ",style="#F8FFD2")
    filename=input()
    return filename

def load_model(filename):
    model=load(f'{filename}.joblib')
    return model
    
def get_input():
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

def flower_predict(user_input,model):
    scaled_input=scaler.transform(user_input)
    pre_class=model.predict(scaled_input.reshape(1, -1))
    return pre_class[0]

def predict():

    filename=get_name()
    model=load_model(filename)
    user_input=get_input()
    pre_class=flower_predict(user_input,model)
    preflower=flowertypes.get(pre_class)
    console.print(f'Predicted Flower type: {preflower}',style="#F8FFD2")
