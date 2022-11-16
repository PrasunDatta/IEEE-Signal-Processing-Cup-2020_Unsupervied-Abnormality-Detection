import numpy as np

def create_dataset(X, time_steps=1):
    Xs = []
    for i in range(len(X) - time_steps + 1):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        

    return np.array(Xs)