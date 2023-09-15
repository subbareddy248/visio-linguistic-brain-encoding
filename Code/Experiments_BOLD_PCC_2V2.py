#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy import spatial
from numpy.linalg import inv, svd
from sklearn.linear_model import Ridge, RidgeCV
import time 
from scipy.stats import zscore
import pickle
import os
from joblib import Parallel, delayed
import argparse
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.decomposition import PCA


def corr(X,Y):
    return np.mean(zscore(X)*zscore(Y),0)

def R2(Pred,Real):
    SSres = np.mean((Real-Pred)**2,0)
    SStot = np.var(Real,0)
    return np.nan_to_num(1-SSres/SStot)

def R2r(Pred,Real):
    R2rs = R2(Pred,Real)
    ind_neg = R2rs<0
    R2rs = np.abs(R2rs)
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= - 1
    return R2rs

def ridge(X,Y,lmbda):
    return np.dot(inv(X.T.dot(X)+lmbda*np.eye(X.shape[1])),X.T.dot(Y))

def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def ridge_sk(X,Y,lmbda):
    rd = Ridge(alpha = lmbda)
    rd.fit(X,Y)
    return rd.coef_.T

def ridgeCV_sk(X,Y,lmbdas):
    rd = RidgeCV(alphas = lmbdas,solver = 'svd')
    rd.fit(X,Y)
    return rd.coef_.T

def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge_sk(X,Y,lmbda)
        error[idx] = 1 -  R2(np.dot(Xval,weights),Yval)
    return error

def ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))

def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def kernel_ridge(X,Y,lmbda):
    return np.dot(X.T.dot(inv(X.dot(X.T)+lmbda*np.eye(X.shape[0]))),Y)

def kernel_ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = kernel_ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def kernel_ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X.T, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)

def kernel_ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X.T, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def cross_val_ridge(train_features,train_data, n_splits = 10, 
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False):
    
    ridge_1 = dict(plain = ridge_by_lambda,
                   svd = ridge_by_lambda_svd,
                   kernel_ridge = kernel_ridge_by_lambda,
                   kernel_ridge_svd = kernel_ridge_by_lambda_svd,
                   ridge_sk = ridge_by_lambda_sk)[method]
    ridge_2 = dict(plain = ridge,
                   svd = ridge_svd,
                   kernel_ridge = kernel_ridge,
                   kernel_ridge_svd = kernel_ridge_svd,
                   ridge_sk = ridge_sk)[method]
    
    n_voxels = train_data.shape[1]
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
#         print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ridge_1(train_features[trn],train_data[trn],
                               train_features[val],train_data[val], 
                               lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost,aspect = 'auto')
        r_cv += cost
#         if icv%3 ==0:
#             print(icv)
#         print('average iteration length {}'.format((time.time()-start_t)/(icv+1)))
    if do_plot:
        plt.figure()
        plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

    argmin_lambda = np.argmin(r_cv,axis = 0)
    weights = np.zeros((train_features.shape[1],train_data.shape[1]))
    for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
    if do_plot:
        plt.figure()
        plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);

    return weights, np.array([lambdas[i] for i in argmin_lambda])




def GCV_ridge(train_features,train_data,lambdas = np.array([10**i for i in range(-6,10)])):
    
    n_lambdas = lambdas.shape[0]
    n_voxels = train_data.shape[1]
    n_time = train_data.shape[0]
    n_p = train_features.shape[1]

    CVerr = np.zeros((n_lambdas, n_voxels))

    # % If we do an eigendecomp first we can quickly compute the inverse for many different values
    # % of lambda. SVD uses X = UDV' form.
    # % First compute K0 = (X'X + lambda*I) where lambda = 0.
    #K0 = np.dot(train_features,train_features.T)
    print('Running svd',)
    start_time = time.time()
    [U,D,Vt] = svd(train_features,full_matrices=False)
    V = Vt.T
    print(U.shape,D.shape,Vt.shape)
    print('svd time: {}'.format(time.time() - start_time))

    for i,regularizationParam in enumerate(lambdas):
        regularizationParam = lambdas[i]
        print('CVLoop: Testing regularization param: {}'.format(regularizationParam))

        #Now we can obtain Kinv for any lambda doing Kinv = V * (D + lambda*I)^-1 U'
        dlambda = D**2 + np.eye(n_p)*regularizationParam
        dlambdaInv = np.diag(D / np.diag(dlambda))
        KlambdaInv = V.dot(dlambdaInv).dot(U.T)
        
        # Compute S matrix of Hastie Trick  H = X(XT X + lambdaI)-1XT
        S = np.dot(U, np.diag(D * np.diag(dlambdaInv))).dot(U.T)
        denum = 1-np.trace(S)/n_time
        
        # Solve for weight matrix so we can compute residual
        weightMatrix = KlambdaInv.dot(train_data);


#         Snorm = np.tile(1 - np.diag(S) , (n_voxels, 1)).T
        YdiffMat = (train_data - (train_features.dot(weightMatrix)));
        YdiffMat = YdiffMat / denum;
        CVerr[i,:] = (1/n_time)*np.sum(YdiffMat * YdiffMat,0);


    # try using min of avg err
    minerrIndex = np.argmin(CVerr,axis = 0);
    r=np.zeros((n_voxels));

    for nPar,regularizationParam in enumerate(lambdas):
        ind = np.where(minerrIndex==nPar)[0];
        if len(ind)>0:
            r[ind] = regularizationParam;
            print('{}% of outputs with regularization param: {}'.format(int(len(ind)/n_voxels*100),
                                                                        regularizationParam))
            # got good param, now obtain weights
            dlambda = D**2 + np.eye(n_p)*regularizationParam
            dlambdaInv = np.diag(D / np.diag(dlambda))
            KlambdaInv = V.dot(dlambdaInv).dot(U.T)

            weightMatrix[:,ind] = KlambdaInv.dot(train_data[:,ind]);


    return weightMatrix, r

def pairwise_accuracy(actual, predicted):
    true = 0
    total = 0
    for i in range(0,len(actual)):
#         print(i)
        for j in range(i+1, len(actual)):
            total += 1

            s1 = actual[i]
            s2 = actual[j]
            b1 = predicted[i]
            b2 = predicted[j]

            result1 = spatial.distance.cosine(s1, b1)
            result2 = spatial.distance.cosine(s2, b2)
            result3 = spatial.distance.cosine(s1, b2)
            result4 = spatial.distance.cosine(s2, b1)

            if(result1 + result2 < result3 + result4):
                true += 1

    return(true/total)

def pearcorr(actual, predicted):
    corr = []
    for i in range(0, len(actual)):
        corr.append(np.corrcoef(actual[i],predicted[i])[0][1])
    return np.mean(corr)


kf = KFold(n_splits=10)


def train(vectors, voxels):
    
    dataset_X = np.array(voxels.copy())
    
    dataset_Y = np.array(vectors.copy())
    
    actual = []
    predicted = []
    pairwise_2v2 = []
    final_corr = []

    cnt = 0
    for train_index, test_index in kf.split(dataset_X):

        X_train, X_test = dataset_X[train_index], dataset_X[test_index]
        y_train, y_test = dataset_Y[train_index], dataset_Y[test_index]
           
        weights, lbda = cross_val_ridge(X_train,y_train) 
        y_pred = np.dot(X_test,weights)
        
        #pairwise_2v2.append(pairwise_accuracy(y_test,y_pred))
        final_corr.append(pearcorr(y_test,y_pred))
        actual.extend(y_test)
        predicted.extend(y_pred)
        #print(pairwise_2v2[cnt],final_corr[cnt],rdm_acc[cnt])
        cnt += 1
        #print(cnt)
    
    fin_acc = pairwise_accuracy(actual,predicted)
    return fin_acc, np.mean(final_corr)


# In[13]:

img_feat = np.load('./DEit_img_feat.npy')
img_feat = np.mean(img_feat, axis=2)
img_feat = np.reshape(img_feat, (img_feat.shape[0], img_feat.shape[2]))
print(img_feat.shape)

file = open('stim_list/stim_lists/CSI01_stim_lists.txt','r')
lines = file.readlines()


ROIS = ['LHPPA', 'RHPPA', 'LHLOC', 'RHLOC', 'LHEarlyVis', 'RHEarlyVis', 'LHOPA', 'RHOPA',  'LHRSC','RHRSC']

for roi in ROIS:
    print(roi)
    print()
    
    for i in np.arange(1,5):
        fmri_data1 = loadmat('s'+str(i)+'mat/mat/CSI'+str(i)+'_ROIs_TR1.mat')
        fmri_data2 = loadmat('s'+str(i)+'mat/mat/CSI'+str(i)+'_ROIs_TR2.mat')
        fmri_data3 = loadmat('s'+str(i)+'mat/mat/CSI'+str(i)+'_ROIs_TR3.mat')
        fmri_data4 = loadmat('s'+str(i)+'mat/mat/CSI'+str(i)+'_ROIs_TR34.mat')
        fmri_data5 = loadmat('s'+str(i)+'mat/mat/CSI'+str(i)+'_ROIs_TR4.mat')
        fmri_data6 = loadmat('s'+str(i)+'mat/mat/CSI'+str(i)+'_ROIs_TR5.mat')
        
        roi_fmri = np.mean([fmri_data1[roi],fmri_data2[roi],fmri_data3[roi],fmri_data4[roi],
            fmri_data5[roi],fmri_data6[roi]], axis=0)
        if i!=1:
            file1 = open('stim_list/stim_lists/CSI0'+str(i)+'_stim_lists.txt','r')
            lines_sub3 = file1.readlines()
            indices = []
            for j in lines_sub3:
                try:
                    indices.append(lines.index(j))
                except:
                    continue
                    
            d,c = train(roi_fmri,img_feat[indices])
            print(np.round(d,3),np.round(c,3))
        else:
            d,c = train(roi_fmri,img_feat)
            print(np.round(d,3),np.round(c,3))