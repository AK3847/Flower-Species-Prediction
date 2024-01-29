#Flower Species Classifier made by Abdul Kadir ;)
#Github profile: https://github.com/AK3847

from model import predict
from rich.console import Console
import time
from rich.progress import track
console=Console()

def start():
    predict()

#main function
if __name__=='__main__':
    console.print('A Python script to predict the Flower Species',style="#42855B")
    console.print('Specifications:',style="Bold underline #42855B")
    console.print('Method Used: [#638889]Decision Tree Classifier[/#638889]',style="#42855B")
    console.print('Test Size: [#638889]0.3/30%[/#638889]',style="#42855B")
    console.print('Data Set Used: [#638889]Iris DataSet[/#638889]',style="#42855B")
    for x in track(range(100),description="Making the Model...",style="#F8FFD2"):
        time.sleep(0.08)
    start()