import numpy as np
def featureScale_train(train, col_list):

    zzz=train[[col_list[1], col_list[2], col_list[3]]]
    z = abs(zzz).max()
    max_ang_vel = z.max()

    zzz=train[[col_list[4], col_list[5], col_list[6]]]
    z = abs(zzz).max()
    max_acc = z.max()
        
    parameters = np.array([max_ang_vel, max_acc])
        
    return parameters


def applyFeatureScale_train(train, final_params):
    [a,b] = final_params
    full_params = np.array([a, a, a, b, b, b, 1, 1, 1, 1]) #ang_vel, acceleration, ori
    return train/ full_params[None, :]


