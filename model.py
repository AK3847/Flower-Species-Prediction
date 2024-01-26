import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import os
from rich.console import Console
from sklearn.datasets import load_iris
console=Console()
state=0
def make_model():
    # global state
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model = DecisionTreeClassifier(random_state=state)
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

def save_state():
    if not os.path.exists('model_state.txt'):
        with open('model_state.txt','w') as file:
            file.write('None')
    with open('model_state.txt','a') as file:
        file.write(str(f'\nState:{state} at Date/Time: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}'))

def predict():
    global state
    state=np.random.randint(0,100000)
    model,X_test_scaled,y_test=make_model()
    s=input(f'Find accuracy (Y/n): ')
    if(s=='y' or s=='Y'):
        accuracy(model,X_test_scaled,y_test)
    elif (s!='n'):
        console.print(f'Wrong Choice!')
    console.print(f'Do you want to save the model state? (Y/N) ')
    s=input()
    if(s=='Y' or s=='y'):
        save_state()