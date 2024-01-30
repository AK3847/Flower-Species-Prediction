#Flower Species Classifier made by Abdul Kadir ;)
#Github profile: https://github.com/AK3847

from model import main_model
from rich.console import Console
import time
from rich.progress import track
from predict import predict
console=Console()

def new_model():
    main_model()

#main function
if __name__=='__main__':
    console.print('A Python script to predict the Flower Species',style="#42855B")
    console.print('Specifications:',style="Bold underline #42855B")
    console.print('Method Used: [#638889]Decision Tree Classifier[/#638889]',style="#42855B")
    console.print('Data Set Used: [#638889]Iris DataSet[/#638889]',style="#42855B")
    console.print('Test Size: [#638889]0.3/30%[/#638889]',style="#42855B")
    for x in track(range(100),description="Making the Model...",style="#F8FFD2"):
        time.sleep(0.08)
    console.print('Train a new model/Use a pre-trained model (T/N):')
    s=input()
    choice=['t','T','n','N']
    while s not in choice:
        console.print('Invalid Input, try again.')
        s=input()
        if s in choice:
            break
    if s=='t' or s=='T':
        model_name=main_model()
        choices=['y','Y','n','N']
        console.print('Want to predict a flower species based on this model? (y/n)')
        s=input()
        while s not in choices:
            console.print('Invalid Input, try again.')
            s=input()
            if s in choices:
                break
        if s in ['y','Y']:
            predict(model_name,False)
        else:   
            console.print('Exiting the Program')
    else:
        predict('',True)