from model import predict
from rich.console import Console
console=Console()
def start():
    predict()

if __name__=='__main__':
    console.print('A Python script to predict the Flower Species on the [italic]Iris DataSet[/italic] via [bold #638889]Decision Tree method.[/bold #638889]',style="#42855B")
    start()