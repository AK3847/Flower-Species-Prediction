from model import predict
from rich.console import Console
import time
console=Console()
def start():
    predict()

if __name__=='__main__':
    console.print('A Python script to predict the Flower Species',style="#42855B")
    console.print('Specifications:',style="Bold #42855B")
    console.print('Method Used: [#638889]Decision Tree Classifier[/#638889]',style="#42855B")
    console.print('Test Size: [#638889]0.3 or 30%[/#638889]',style="#42855B")
    console.print('Data Set Used: [#638889]Iris DataSet[/#638889]',style="#42855B")
    time.sleep(2)
    start()