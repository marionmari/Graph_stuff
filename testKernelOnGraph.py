import numpy as np 
from Graph_stuff.graph_util import form_knn_graph,form_kernel_matrix
from Graph_stuff.kernels_on_graph import *
from pyGP_OO.Valid import valid
from pyGP_OO.Core import *
from Graph_stuff.usps_digit_data.loadBinaryUSPS import load_binary

SHUFFLE = True
ACC = []

# load small sample data with 2 classes
# digit 1 for +1 and digit 2 for -1
x,y = load_binary(1,2,reduce=True)

# shuffle the data if necessary
if SHUFFLE:
    data = np.append(x, y, axis=1)
    np.random.shuffle(data)
    x = data[:,:-1]
    y = data[:,-1:]

# form a knn graph 
A = form_knn_graph(x,2)

# use diffusion kernel to get precomputed matrix
# M = diffKernel(A)
M = regLapKernel(A)


# optional normalized the kernel matrix here
N = M.shape[0]

# cross validation
for indice_train,indice_test in valid.k_fold_indice(N, K=10):
    ''' switch train and test indices
    to train 20 examples only
    '''    
    n = len(indice_test)             # number of training data
    M1,M2 = form_kernel_matrix(M, indice_test, indice_train)

    k = cov.covPre(M1,M2)
    #k = cov.covSEiso([-1,0.])
    #k = k1+k2
    m = mean.meanZero()
    l = lik.likErf()
    i = inf.infLaplace()
    o = opt.Minimize()
    
    # if you only use precomputed kernel matrix, there is no x data
    # but you need to specify x to pass to gp (since the function still needs dimension of data)
    # you can create the following:
    # x_train = np.zeros((n,1))

    # if you use combination of precomputed matrix and normal kernel function,
    # this problem does not exsit
    # you can call gp in the normal way
    x_train = x[indice_test,:]
    y_train = y[indice_test,:]
    x_test = x[indice_train,:]
    y_test = y[indice_train,:]
    gp.train(i,m,k,l,x_train,y_train,o)
    out = gp.predict(i,m,k,l,x_train,y_train,x_test,None)

    # evaluation
    pred_class = np.sign(out[0])
    acc = valid.ACC(pred_class, y_test)
    ACC.append(acc)
    print 'Acc: %.3f' % acc

print 'Average Acc over k-fold: %.3f' % np.mean(ACC)


