import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import rich
from rich.console import Console
from sklearn.datasets import load_iris
console=Console()
def make_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model = DecisionTreeClassifier(random_state=1)
    model.fit(X_train_scaled, y_train)
    console.print(f'The Model has been succesfully made!',style="italic #42855B")
    return model,X_test_scaled,y_test

def accuracy(model,X_test_scaled,y_test):
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    console.print(f"Accuracy: {accuracy}",style="#5272F2")
    console.print(f"[#6895D2]Classification Report:[/#6895D2]\n {report}",style="#F3B95F")
    matrix=confusion_matrix(y_test,y_pred)
    console.print(f"[#6895D2]Confusion Matrix:[/#6895D2]\n{matrix}")


def predict():
    model,X_test_scaled,y_test=make_model()
    s=input(f'Find accuracy (Y/n): ')
    if(s=='y' or s=='Y'):
        accuracy(model,X_test_scaled,y_test)
    else:
        console.print(f'Exiting....')
        exit()