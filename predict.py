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
    console.print('Give your dimensions (in cm):',style="#FFF3CF")
    console.print('Sepal length ',style="#F8FFD2")
    sepal_length = float(input())
    console.print('Sepal width ',style="#F8FFD2")
    sepal_width = float(input())
    console.print('Petal length ',style="#F8FFD2")
    petal_length = float(input())
    console.print('Petal width ',style="#F8FFD2")
    petal_width = float(input())
    return np.array([[sepal_length, sepal_width, petal_length, petal_width]])

def flower_predict(user_input,model):
    scaled_input=scaler.transform(user_input)
    pre_class=model.predict(scaled_input.reshape(1, -1))
    return pre_class[0]

def predict(model_name,model_status):
    if(model_status):
       filename=get_name()
       model=load_model(filename)
    else:
        model=load_model(model_name)
    user_input=get_input()
    pre_class=flower_predict(user_input,model)
    preflower=flowertypes.get(pre_class)
    console.print(f'Predicted Flower type: [bold italic]{preflower}[/bold italic]',style="#F8FFD2")
