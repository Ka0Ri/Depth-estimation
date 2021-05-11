import numpy as np

gt = np.random.randint(2, size=65000*5400)
predict = gt

TT = np.sum(gt * predict)
FF = np.sum((1 - gt) * (1 - predict))
TF = np.sum((1 - gt) * predict)
FT = np.sum((gt) * (1 - predict))
print("Confusion matrix: ")
print("TT: ", TT)
print("TF: ", TF)
print("FT: ", FT)
print("FF: ", FF)