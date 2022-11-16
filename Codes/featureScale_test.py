import numpy as np

def featureScale_test(test,col_list, z):

    
    max_ang_vel = z[0]
    for i in range(1,4):
        test[col_list[i]] /= max_ang_vel
    
    
    max_acc = z[1]
    for i in range(4,7):
        test[col_list[i]] /= max_acc

        
    return test

