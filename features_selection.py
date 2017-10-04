from  xgboost import XGBClassifier
from matplotlib import pyplot
from xgboost import plot_importance

def importance(X,Y):
    model=XGBClassifier()
    model.fit(X,Y)
    plot_importance(model)
    pyplot.show()

