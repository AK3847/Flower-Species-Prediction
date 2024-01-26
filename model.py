import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris

from rich.console import Console
from rich.progress import track

import os
import time
console=Console()
state=0
def make_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model = DecisionTreeClassifier(random_state=state)
    model.fit(X_train_scaled, y_train)
    console.print(f'The Model has been succesfully made!',style="italic #F8FFD2")
    return model,X_test_scaled,y_test

def accuracy(model,X_test_scaled,y_test):
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    console.print(f"Accuracy: {accuracy}",style="#6895D2")
    console.print(f"[#6895D2]Classification Report:[/#6895D2]\n {report}",style="#F3B95F")
    matrix=confusion_matrix(y_test,y_pred)
    console.print(f"[#6895D2]Confusion Matrix:[/#6895D2]\n{matrix}")

def save_state():
    if not os.path.exists('model_state.txt'):
        with open('model_state.txt','w') as file:
            file.write('')
    with open('model_state.txt','a') as file:
        file.write(str(f'\nState:{state} at Date/Time: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}'))

def predict():
    global state
    state=np.random.randint(0,100000)
    model,X_test_scaled,y_test=make_model()
    console.print(f'Find accuracy (Y/n): ',style="#F8FFD2")
    s=''
    while s not in ('y','n','Y','N'):
        s=input()
        if s in ('y','Y','n','N'):
            break
        console.print('Wrong Input Please Try again',style="#FF1700")
    if(s=='y' or s=='Y'):
        for x in track(range(100),description="Finding Accuracy..",style="#F8FFD2"):
            time.sleep(0.1)
        accuracy(model,X_test_scaled,y_test)
    console.print(f'Do you want to save the model state? (Y/n) ',style="#F8FFD2")
    s=''
    while s not in ('y','n','Y','N'):
        s=input()
        if s in ('y','Y','n','N'):
            break
        console.print('Wrong Input Please Try again',style="#FF1700")
    if s in ('y','Y'):
        console.print(f'The state : {state} has been saved in [italic] model_state.txt [/italic]',style="#F8FFD2")
        save_state()
    else:
        console.print('Exiting the program...',style="Italic #F8FFD2 ")
        time.sleep(1)
        exit()