# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 03:36:05 2022

@author: Haris
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 18:58:05 2022

@author: Haris
"""

import sys, os
file_path=os.path.dirname(os.path.abspath(__file__))
sys.path
sys.path.append(file_path)
os.chdir(file_path)


#%%

import numpy as np
#np.random.seed(0)
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import fairlearn
from fairlearn.metrics import MetricFrame
import csv

import random
#random.seed(0)

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from nba_libararies_LG import (sort_correlation,sep_input_output_data,
                            normalize_dataframe, nba_data_processing,
                            learn_logistic_regression,fairness_measures,
                            binary_conf_mat,MCAR,MAR,MNAR,imp_IAPD,train_node2vec_nba,train_node2vec_german,
                            train_node2vec_bail,train_node2vec_credit)

from nba_libararies_LG import *

from fancyimpute import (
    BiScaler,
    KNN,
    NuclearNormMinimization,
    SoftImpute,
    SimpleFill,
    IterativeImputer,
    IterativeSVD,
    MatrixFactorization
)

pd.options.mode.chained_assignment = None  # default='warn'
#%%

algorithm=None
Dataset=None
path_csv=None


#%%
sensitive_attr=None
x_data=None
y_data=None
df_edge_list=None

def train_lg(x_train=None,y_train=None,x_test=None,y_test=None,sensitive_attr=sensitive_attr,ttrain_index=None, ttest_index=None):   
    
    
    #x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = test_size,stratify=y_data)
    
    xx_data=pd.concat([x_train, x_test], axis=0)
    yy_data=pd.concat([y_train, y_test], axis=0)
    #x_data1=normalize_dataframe(x_data, Scaler=scaler)
    
    
    data_index=pd.concat([xx_data, yy_data], axis=1)
    #data_no_index=pd.concat([x_data1, y_data], axis=1)
    #x_train1, x_test1, y_train1, y_test1 = train_test_split(xx_data, yy_data, test_size = test_size,shuffle=False)
        
    
    if algorithm=='LR':
        
        if Dataset=='NBA':
    
            [pred,report,conf_mat,model_LG_nba]=learn_logistic_regression(x_train, x_test, y_train, y_test)
            
        elif Dataset=='german':
    
            [pred,report,conf_mat,model_LG_nba]=learn_logistic_regression(x_train, x_test, y_train, y_test)
            
        elif Dataset=='credit': 
            [pred,report,conf_mat,model_LG_nba]=learn_logistic_regression(x_train, x_test, y_train, y_test)
            
        elif Dataset=='bail': 
            [pred,report,conf_mat,model_LG_nba]=learn_logistic_regression(x_train, x_test, y_train, y_test)
            
        elif Dataset=='pokec_z': 
            [pred,report,conf_mat,model_LG_nba]=learn_logistic_regression(x_train, x_test, y_train, y_test)
            
        elif Dataset=='pokec_n': 
            [pred,report,conf_mat,model_LG_nba]=learn_logistic_regression(x_train, x_test, y_train, y_test)
    
    elif algorithm=='Node2Vec':
        
        if Dataset=='NBA':
        
            [pred,report,conf_mat,model_LG_nba]=train_node2vec_nba(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list, \
                                                                   train_index=ttrain_index, test_index=ttest_index)
    
        elif Dataset=='german': 
            [pred,report,conf_mat,model_LG_nba]=train_node2vec_german(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list, \
                                                                   train_index=ttrain_index, test_index=ttest_index)
    
        elif Dataset=='credit': 
            [pred,report,conf_mat,model_LG_nba]=train_node2vec_credit(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list, \
                                                                   train_index=ttrain_index, test_index=ttest_index)
                
        elif Dataset=='bail': 
            [pred,report,conf_mat,model_LG_nba]=train_node2vec_bail(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list, \
                                                                   train_index=ttrain_index, test_index=ttest_index)
    
        elif Dataset=='pokec_z': 
            [pred,report,conf_mat,model_LG_nba]=train_node2vec_pokec_z(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list, \
                                                                   train_index=ttrain_index, test_index=ttest_index)
                
        elif Dataset=='pokec_n': 
            [pred,report,conf_mat,model_LG_nba]=train_node2vec_pokec_n(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list, \
                                                                   train_index=ttrain_index, test_index=ttest_index)
        
    elif algorithm=='fairwalk':
        
        if Dataset=='NBA':
            [pred,report,conf_mat,model_LG_nba]=train_failwalk(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list,train_index=ttrain_index,test_index=ttest_index,d_name=Dataset)
   
        elif Dataset=='german': 
            [pred,report,conf_mat,model_LG_nba]=train_failwalk(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list,train_index=ttrain_index,test_index=ttest_index,d_name=Dataset)
       
        elif Dataset=='credit': 
            [pred,report,conf_mat,model_LG_nba]=train_failwalk(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list,train_index=ttrain_index,test_index=ttest_index,d_name=Dataset)
                   
        elif Dataset=='bail': 
            [pred,report,conf_mat,model_LG_nba]=train_failwalk(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list,train_index=ttrain_index,test_index=ttest_index,d_name=Dataset)

        elif Dataset=='pokec_z':
            [pred,report,conf_mat,model_LG_nba]=train_failwalk(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list,train_index=ttrain_index,test_index=ttest_index,d_name=Dataset)
        
        elif Dataset=='pokec_n':
            [pred,report,conf_mat,model_LG_nba]=train_failwalk(X_data=xx_data,Y_data=yy_data,df_edge_list=df_edge_list,train_index=ttrain_index,test_index=ttest_index,d_name=Dataset)
   
    
    elif algorithm=='GCN':
        
        if Dataset=='NBA':
        
            [pred,report,conf_mat, model]=train_gcn_nba(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
   
        elif Dataset=='german': 
            [pred,report,conf_mat, model]=train_gcn_german(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
    
        elif Dataset=='credit': 
            [pred,report,conf_mat, model]=train_gcn_credit(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
                
        elif Dataset=='bail': 
            [pred,report,conf_mat, model]=train_gcn_bail(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
    
        elif Dataset=='pokec_z': 
            [pred,report,conf_mat, model]=train_gcn_pokec_z(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
                
        elif Dataset=='pokec_n': 
            [pred,report,conf_mat, model]=train_gcn_pokec_n(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
    
    elif algorithm=='GAT':
        
        if Dataset=='NBA':
        
            [pred,report,conf_mat, model]=train_gat_nba(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
   
        elif Dataset=='german': 
            [pred,report,conf_mat, model]=train_gat_german(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
    
        elif Dataset=='credit': 
            [pred,report,conf_mat, model]=train_gat_credit(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
                
        elif Dataset=='bail': 
            [pred,report,conf_mat, model]=train_gat_bail(Dataset=data_index, df_edge_list=df_edge_list, X_train=x_train, \
                                            X_test=x_test, Y_train=y_train, Y_test=y_test)
    
    
    
    fm=fairness_measures(sensitive_attr,x_test,y_test,pred)
    
    
    if len(fm.by_group)==1:
        if fm.by_group.index==0:
            
            fm.by_group.loc[1] = np.zeros(len(fm.by_group.columns))
            
        elif fm.by_group.index==1:
            
            fm.by_group.loc[0] = np.zeros(len(fm.by_group.columns))
    
    a=fm.by_group.reset_index()
    a=a.sort_values(a.columns[0])
    a=a.set_index(sensitive_attr)
    
    fm_group=pd.concat([a,fm.overall.to_frame().T],ignore_index=True)    
    
    #by convension parity differnece is from first row of fairness_measures - second row
    
    sens_attr = x_test[sensitive_attr]
    
    # by convension(sp  0 - sp 1)
    sp_diff=fairlearn.metrics.demographic_parity_difference(y_true=y_test,y_pred=pred,sensitive_features=sens_attr)
    #print("parity differnece:",sp_diff)
    
    # by convension(equalized odd 0 - equalized odd 1)
    eo_diff=fairlearn.metrics.equalized_odds_difference(y_true=y_test,y_pred=pred,sensitive_features=sens_attr)
    #print("Equalized Odd difference:",eo_diff)
        
    
    [conf_mat,sp_a,sp_b,eq_opp]=binary_conf_mat(sen_attr=sens_attr, test_true_labels=y_test, test_pred_labels=pred)
    
    conf_mat['SP_1']=sp_a
    conf_mat['SP_0']=sp_b
    conf_mat['EO']=eq_opp
    conf_mat['sp_diff']=sp_diff
    
    conf_mat=pd.DataFrame(conf_mat,index=[0])
    
    return conf_mat,fm_group

fm_group_list=None
conf_mat_list=None
       
def average_results(conf_mat_list=conf_mat_list,fm_group_list=fm_group_list):
    
    
    
    conf_mat_all = pd.concat(conf_mat_list) 
    fm_group_all = pd.concat(fm_group_list)
    
    #if len(conf_mat_all)==1:
    #    return conf_mat_all,fm_group_all,conf_mat_all,fm_group_all
    
    #else:
    conf_mat_all=conf_mat_all.apply(pd.to_numeric)
    fm_group_all=fm_group_all.apply(pd.to_numeric)
    
    by_row_index = conf_mat_all.groupby(conf_mat_all.index)
    conf_mat_all_avr = by_row_index.mean()
      
    by_row_index = fm_group_all.groupby(fm_group_all.index)
    fm_group_all_avr = by_row_index.mean()
        
    return conf_mat_all,fm_group_all,conf_mat_all_avr,fm_group_all_avr

  
def write_csv_MCAR(Dataset=None,algorithm=None,\
  missing_data_type=None,imputation=None,exp_no=None,test_size=None,\
      t_MCAR=None,t_k_MCAR=None,k_MCAR=None,conf_mat_all_avr=None,\
      fm_group_all_avr=None,path_csv=None,alpha1_MAR=None,alpha2_MAR=None,\
    beta1_MAR=None,beta2_MAR=None,A=None,alpha1_MNAR=None,alpha2_MNAR=None,\
    beta1_MNAR=None,beta2_MNAR=None, imp_IAPD=None):
    

    a=['DataSet', 
    'Algorithm', 
    'MissingData', 
    'Imputataion',
    'No of exp',
    'Test Set',
    'Theta_MCAR',
    'Theta_K_MCAR',
        'K',
     'alpha1_MAR',
     'beta1_MAR',
     'alpha2_MAR',
     'beta2_MAR',
     'alpha1_MNAR',
     'beta1_MNAR',
     'alpha2_MNAR',
     'beta2_MNAR',
     'Type MAR/MNAR',
     'imp_IAPD']

    
    
    d1=[Dataset,algorithm,missing_data_type,imputation,exp_no,test_size\
        ,t_MCAR,t_k_MCAR,k_MCAR,alpha1_MAR,beta1_MAR,alpha2_MAR,beta2_MAR,alpha1_MNAR,
        beta1_MNAR,alpha2_MNAR,beta2_MNAR,A,  imp_IAPD]
    
        
  
    if fm_group_all_avr is not None:    
        d2=list(conf_mat_all_avr.iloc[0])
        
        d3=list(fm_group_all_avr.iloc[0])
        d4=list(fm_group_all_avr.iloc[1])
        d5=list(fm_group_all_avr.iloc[2])
    
        data=d1+d2+d3+d4+d5
    else:
        data=d1+([None]*(12+24))
    
    if missing_data_type=='None':
        
        b=list(conf_mat_all_avr.columns)

        c=list(fm_group_all_avr.columns)

        string = '_0'
        c1 = [x + string for x in c]

        string = '_1'
        c2 = [x + string for x in c]

        string = '_all'
        c3 = [x + string for x in c]

        header=a+b+c1+c2+c3
        
        with open(path_csv, 'a+',newline ='') as f:
      
            # using csv.writer method from CSV package
            write = csv.writer(f)
              
            write.writerow(header)
            write.writerow(data)
            
    elif missing_data_type!='None':
        
        
        with open(path_csv, 'a+',newline ='') as f:
      
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(data)
    
attr_type=None
theta_MCAR_k=None
k1=None
theta_MCAR=None
imputation=None
attr_type=None
no=None
output_attr=None


# X_test, X_train, Y_test, Y_train

def MCAR_imputation(XX_data=None,YY_data=None,k1=None,theta_MCAR_k=None,\
                    theta_MCAR=None,imputation=None,attr_type=None,no=None,\
                    output_attr=None,sensitive_attr=None):
     
    
    
    for j in missing_data_type:
        
        
        if j=='None':
            
            i='None'
            alpha1='None'
            alpha2='None'
            beta1='None'
            beta2='None'
            A1='None'
            
            
            conf_mat_list=[]
            fm_group_list=[]
            
            X_train, X_test, Y_train, Y_test = train_test_split(XX_data, YY_data, test_size = test_size,stratify=YY_data)
                        
            train_index=X_train.index.values.tolist()

            test_index=X_test.index.values.tolist()
            
            
            [conf_mat,fm_group]=train_lg(x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test\
                                         ,sensitive_attr=sensitive_attr,ttrain_index=train_index, ttest_index=test_index)
            
            conf_mat_list.append(conf_mat)
            fm_group_list.append(fm_group)
            
            [conf_mat_all,fm_group_all,conf_mat_all_avr,fm_group_all_avr]=\
                average_results(conf_mat_list=conf_mat_list, fm_group_list=fm_group_list)
            
            write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
              missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                  t_MCAR='None',t_k_MCAR='None',k_MCAR='None',conf_mat_all_avr=conf_mat_all_avr,\
                  fm_group_all_avr=fm_group_all_avr,path_csv=path_csv,alpha1_MAR=alpha1,alpha2_MAR=alpha2,\
                beta1_MAR=beta1,beta2_MAR=beta2,A=A1)
                    
                #write_results_no_missing()
            
        elif j=='MCAR':
            
            print('MCAR')
            
            for a in attr_type:
                if a=='sensitive attribute only':
                    for i in imputation:
                        for t in theta_MCAR:
                       
                            
                            #k='None'
                            #t_k='None'
                            
                            conf_mat_list=[]
                            fm_group_list=[]
                            imp_iapd_list=[]
                            
                            #z=True
                            
                            for y in range(no):
                                
                                #if z==False:
                                    #continue
                                X_train, X_test, Y_train, Y_test = train_test_split(XX_data, YY_data, test_size = test_size,stratify=YY_data)
                                                
                                train_index=X_train.index.values.tolist()

                                test_index=X_test.index.values.tolist()
                                
                                result2 = pd.concat([X_train, Y_train], axis=1, join='inner')
                                
                                result1=MCAR(Dataset=result2,Theta=t, Attr_type=a,Sens_attr_name=sensitive_attr,K='None', theta_k='None')
                                
                                #z=del_data_info(dataset=result1,output_attr=output_attr,sensitive_attr=sensitive_attr)
                                
                                #if z == False:
                                    #continue
                                
                                result=perform_imputation(df=result1,I=i)
                                result.index=result2.index
                                
                                #result2.reset_index( inplace=True,drop=True)
                                #result1.reset_index( inplace=True,drop=True)
                                #result.reset_index( inplace=True,drop=True)
                                
                                imp_iapd=imp_IAPD(dataset_original=result2,dataset_missing=result1,\
                                                  dataset_imp=result,sens_attr=sensitive_attr,I=i,scaler=scaler)
                               
                                imp_iapd_list.append(imp_iapd)
                                    
                                [y_train_imp,x_train_imp]=sep_input_output_data(result, attr=output_attr)
                                
                                #try:
                                    
                                [conf_mat,fm_group]=train_lg(x_train=x_train_imp,y_train=y_train_imp,x_test=X_test,y_test=Y_test\
                                                             ,sensitive_attr=sensitive_attr,ttrain_index=train_index, ttest_index=test_index)
                                    
                                    
                                #except:
                                    #continue
                            
                                conf_mat_list.append(conf_mat)
                                fm_group_list.append(fm_group)
                            
                            imp_iapd=Average(imp_iapd_list)
                            
                            if len(fm_group_list) !=0:
                                
                                [conf_mat_all,fm_group_all,conf_mat_all_avr,fm_group_all_avr]=\
                                    average_results(conf_mat_list=conf_mat_list, fm_group_list=fm_group_list)
                                
                                write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                  missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                      t_MCAR=t,t_k_MCAR='None',k_MCAR='None',conf_mat_all_avr=conf_mat_all_avr,\
                                      fm_group_all_avr=fm_group_all_avr,path_csv=path_csv, imp_IAPD=imp_iapd)
                            else:
                                write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                  missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                      t_MCAR=t,t_k_MCAR='None',k_MCAR='None',path_csv=path_csv, imp_IAPD=imp_iapd)
                            
                
                elif a=='from k attributes':
                    
                    #theta_MCAR_k=theta_MCAR_k # when k is more than one, missing data in many attributes
            
                    #K=K  #number of sttributes from which data is removed for missingness
                    for i in imputation:
                        for k in k1:
                    
                            for t_k in theta_MCAR_k:
                                
                                conf_mat_list=[]
                                fm_group_list=[]
                                imp_iapd_list=[]
                                
                                t='None'
                                #z=True
                                
                                for y in range(no):
                                    X_train, X_test, Y_train, Y_test = train_test_split(XX_data, YY_data, test_size = test_size,stratify=YY_data)
                                                
                                    train_index=X_train.index.values.tolist()

                                    test_index=X_test.index.values.tolist()
                                    
                                    #if z==False:
                                        #continue
                                    
                                    result2 = pd.concat([X_train, Y_train], axis=1, join='inner')
                                    
                                    result1=MCAR(Dataset=result2,Theta=t, Attr_type=a,Sens_attr_name=sensitive_attr,K=k, theta_k=t_k)
                                    
                                    z=del_data_info(dataset=result1,output_attr=output_attr,sensitive_attr=sensitive_attr)
                                    
                                    #if z == False:
                                        #continue
                                    
                                    
                                    result=perform_imputation(df=result1,I=i)
                                    result.index=result2.index
                                    #result2.reset_index( inplace=True,drop=True)
                                    #result1.reset_index( inplace=True,drop=True)
                                    #result.reset_index( inplace=True,drop=True)
                                    
                                    imp_iapd=imp_IAPD(dataset_original=result2,dataset_missing=result1,\
                                                      dataset_imp=result,sens_attr=sensitive_attr,I=i,scaler=scaler)
                                    
                                    imp_iapd_list.append(imp_iapd)    
                                        
                                    [y_train_imp,x_train_imp]=sep_input_output_data(result, attr=output_attr)
                                    
                                    #try:
                                        
                                    [conf_mat,fm_group]=train_lg(x_train=x_train_imp,y_train=y_train_imp,x_test=X_test,y_test=Y_test\
                                                                 ,sensitive_attr=sensitive_attr,ttrain_index=train_index, ttest_index=test_index)
                                    #except:
                                        #continue
                                    
                                    conf_mat_list.append(conf_mat)
                                    fm_group_list.append(fm_group)
                                    
                                imp_iapd=Average(imp_iapd_list)
                                
                                if len(fm_group_list) !=0:
                                    
                                    [conf_mat_all,fm_group_all,conf_mat_all_avr,fm_group_all_avr]=\
                                        average_results(conf_mat_list=conf_mat_list, fm_group_list=fm_group_list)
                                    
                                    write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                      missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                          t_MCAR=t,t_k_MCAR=t_k,k_MCAR=k,conf_mat_all_avr=conf_mat_all_avr,\
                                          fm_group_all_avr=fm_group_all_avr,path_csv=path_csv, imp_IAPD=imp_iapd)
                                
                                else:
                                    write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                      missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                          t_MCAR=t,t_k_MCAR=t_k,k_MCAR=k,path_csv=path_csv, imp_IAPD=imp_iapd)
                        

k11=None
alpha1_MAR=None
alpha2_MAR=None
beta1_MAR=None
beta2_MAR=None

def MAR_imputation(x_data=x_data,y_data=y_data,k1=k11,imputation=imputation,attr_type=attr_type,no=no,\
                    output_attr=output_attr,sensitive_attr=sensitive_attr, alpha1_MAR=alpha1_MAR,\
                    alpha2_MAR=alpha2_MAR,beta1_MAR=beta1_MAR,beta2_MAR=beta2_MAR):
    
    
    for a in attr_type:
        if a=='sensitive attribute only':
            for i in imputation:  
                
                for A1 in [0,1]: #first two cases
                    
                    for alpha1 in alpha1_MAR:
                        for beta1 in beta1_MAR:
                            
                            alpha2='None'
                            beta2='None'
                            k11='None'
                            
                            par=[alpha1,beta1,alpha2,beta2]
                            
                            conf_mat_list=[]
                            fm_group_list=[]
                            imp_iapd_list=[]
                            
                            for y in range(no):  
            
                                result2 = pd.concat([x_data, y_data], axis=1, join='inner')
                                
                                result1=MAR(Dataset=result2, A=A1, Par=par, Attr_type=a,\
                                         sens_attr_name=sensitive_attr,K=k11)
                                
                                
                                z=del_data_info(dataset=result1,output_attr=output_attr,sensitive_attr=sensitive_attr)
                                
                                if z == False:
                                    continue    
                                    
                                    
                                result=perform_imputation(df=result1,I=i)
                                
                               
                                
                                imp_iapd=imp_IAPD(dataset_original=result2,dataset_missing=result1,\
                                                  dataset_imp=result,sens_attr=sensitive_attr,I=i)
                                
                                imp_iapd_list.append(imp_iapd)
                                
                                [y_data1,x_data1]=sep_input_output_data(result, attr=output_attr)
                                
                                try:
                                    
                                    [conf_mat,fm_group]=train_lg(x_data=x_data1,y_data=y_data1\
                                                             ,sensitive_attr=sensitive_attr)
                                except:
                                    continue
                                
                                conf_mat_list.append(conf_mat)
                                fm_group_list.append(fm_group)
                                
                            imp_iapd=Average(imp_iapd_list)
                            if len(fm_group_list) !=0:
                                
                                [conf_mat_all,fm_group_all,conf_mat_all_avr,fm_group_all_avr]=\
                                    average_results(conf_mat_list=conf_mat_list, fm_group_list=fm_group_list)
                                
                                write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                  missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                      t_MCAR='None',t_k_MCAR='None',k_MCAR='None',conf_mat_all_avr=conf_mat_all_avr,\
                                      fm_group_all_avr=fm_group_all_avr,path_csv=path_csv,alpha1_MAR=alpha1,alpha2_MAR=alpha2,\
                                    beta1_MAR=beta1,beta2_MAR=beta2,A=A1, imp_IAPD=imp_iapd)
                            
                            else:
                                write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                  missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                    path_csv=path_csv,alpha1_MAR=alpha1,alpha2_MAR=alpha2,\
                                    beta1_MAR=beta1,beta2_MAR=beta2,A=A1, imp_IAPD=imp_iapd)
                                


        
        elif a=='from k attributes':
            
            for i in imputation:
                
                for k11 in k1:
            
                    for A1 in [2,3]: #first two cases
                        
                        for alpha2 in alpha2_MAR:
                            for beta2 in beta2_MAR:
                            
                                alpha1='None'
                                beta1='None'
                                
                                par=[alpha1,beta1,alpha2,beta2]
                                
                                conf_mat_list=[]
                                fm_group_list=[]
                                imp_iapd_list=[]
                        
                                for y in range(no):
                                    
                                    
                                    result2 = pd.concat([x_data, y_data], axis=1, join='inner')
                                    
                                    result1=MAR(Dataset=result2, A=A1, Par=par, Attr_type=a,\
                                             sens_attr_name=sensitive_attr,K=k11)
                                    
                                    
                                    z=del_data_info(dataset=result1,output_attr=output_attr,sensitive_attr=sensitive_attr)
                                    
                                    if z == False:
                                        continue    
                                        
                                        
                                    result=perform_imputation(df=result1,I=i)
                                    
                                    
                                    imp_iapd=imp_IAPD(dataset_original=result2,dataset_missing=result1,\
                                                      dataset_imp=result,sens_attr=sensitive_attr,I=i)
                                    imp_iapd_list.append(imp_iapd)    
                                    [y_data1,x_data1]=sep_input_output_data(result, attr=output_attr)
                                    try:
                                        
                                        [conf_mat,fm_group]=train_lg(x_data=x_data1,y_data=y_data1\
                                                                 ,sensitive_attr=sensitive_attr)
                                    except:
                                        continue
                                    
                                    conf_mat_list.append(conf_mat)
                                    fm_group_list.append(fm_group)
                                    

                                imp_iapd=Average(imp_iapd_list) 
                                if len(fm_group_list) !=0:
                                    
                                    [conf_mat_all,fm_group_all,conf_mat_all_avr,fm_group_all_avr]=\
                                        average_results(conf_mat_list=conf_mat_list, fm_group_list=fm_group_list)
                                    
                                    write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                      missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                          t_MCAR='None',t_k_MCAR='None',k_MCAR=k11,conf_mat_all_avr=conf_mat_all_avr,\
                                          fm_group_all_avr=fm_group_all_avr,path_csv=path_csv,alpha1_MAR=alpha1,alpha2_MAR=alpha2,\
                                        beta1_MAR=beta1,beta2_MAR=beta2,A=A1, imp_IAPD=imp_iapd)
                                
                                else:
                                    
                                    write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                      missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                        k_MCAR=k11,path_csv=path_csv,alpha1_MAR=alpha1,alpha2_MAR=alpha2,\
                                        beta1_MAR=beta1,beta2_MAR=beta2,A=A1, imp_IAPD=imp_iapd)

alpha1_MNAR=None
alpha2_MNAR=None
beta1_MNAR=None
beta2_MNAR=None

def MNAR_imputation(x_data=x_data,y_data=y_data,k1=k11,imputation=imputation,attr_type=attr_type,no=no,\
                    output_attr=output_attr,sensitive_attr=sensitive_attr, alpha1_MNAR=alpha1_MNAR,\
                    alpha2_MNAR=alpha2_MNAR,beta1_MNAR=beta1_MNAR,beta2_MNAR=beta2_MNAR):
    
    for a in attr_type:
        if a=='sensitive attribute only':
            for i in imputation:  
                
                for A1 in [0]: #first two cases
                    
                    for alpha1 in alpha1_MNAR:
                        for beta1 in beta1_MNAR:
                            
                            alpha2=None
                            beta2=None
                            k11=None
                            
                            par=[alpha1,beta1,alpha2,beta2]
                            
                            conf_mat_list=[]
                            fm_group_list=[]
                            imp_iapd_list=[]
                            
                            for y in range(no):  
            
                                result2 = pd.concat([x_data, y_data], axis=1, join='inner')
                                
                                
                                result1=MNAR(Dataset=result2, A=A1, Par=par, Attr_type=a,\
                                         Sens_attr_name=sensitive_attr,K=k11)    
                                
                                
                                z=del_data_info(dataset=result1,output_attr=output_attr,sensitive_attr=sensitive_attr)
                                
                                if z == False:
                                    continue    
                                    
                                result=perform_imputation(df=result1,I=i)
                                
                                
                                imp_iapd=imp_IAPD(dataset_original=result2,dataset_missing=result1,\
                                                  dataset_imp=result,sens_attr=sensitive_attr,I=i)
                                imp_iapd_list.append(imp_iapd)
                                [y_data1,x_data1]=sep_input_output_data(result, attr=output_attr)
                                
                                try:
                                    [conf_mat,fm_group]=train_lg(x_data=x_data1,y_data=y_data1\
                                                             ,sensitive_attr=sensitive_attr)
                                except:
                                    continue
                                conf_mat_list.append(conf_mat)
                                fm_group_list.append(fm_group)
                                
                            imp_iapd=Average(imp_iapd_list)
                            if len(fm_group_list) !=0:
                                
                                [conf_mat_all,fm_group_all,conf_mat_all_avr,fm_group_all_avr]=\
                                    average_results(conf_mat_list=conf_mat_list, fm_group_list=fm_group_list)
                                
                                write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                      missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                          k_MCAR=k11,conf_mat_all_avr=conf_mat_all_avr,\
                                          fm_group_all_avr=fm_group_all_avr,path_csv=path_csv,A=A1,\
                                        alpha1_MNAR=alpha1,alpha2_MNAR=alpha2,\
                                        beta1_MNAR=beta1,beta2_MNAR=beta2, imp_IAPD=imp_iapd)
                            
                            else:
                                write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                      missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                          k_MCAR=k11,path_csv=path_csv,A=A1,\
                                        alpha1_MNAR=alpha1,alpha2_MNAR=alpha2,\
                                        beta1_MNAR=beta1,beta2_MNAR=beta2, imp_IAPD=imp_iapd)


        
        elif a=='from k attributes':
            
            for i in imputation:
                
                for k11 in k1:
            
                    for A1 in [1]: #first two cases
                        
                        for alpha2 in alpha2_MNAR:
                            for beta2 in beta2_MNAR:
                            
                                alpha1=None
                                beta1=None
                                
                                par=[alpha1,beta1,alpha2,beta2]
                                
                                conf_mat_list=[]
                                fm_group_list=[]
                                imp_iapd_list=[]
                        
                                for y in range(no):
                                    
                                    
                                    result2 = pd.concat([x_data, y_data], axis=1, join='inner')
                                    
                                    result1=MNAR(Dataset=result2, A=A1, Par=par, Attr_type=a,\
                                             Sens_attr_name=sensitive_attr,K=k11) 
                                    
                                        
                                    z=del_data_info(dataset=result1,output_attr=output_attr,sensitive_attr=sensitive_attr)
                                    
                                    if z == False:
                                        continue    
                                        
                                    result=perform_imputation(df=result1,I=i)
                                    
                                    
                                    
                                    imp_iapd=imp_IAPD(dataset_original=result2,dataset_missing=result1,\
                                                      dataset_imp=result,sens_attr=sensitive_attr,I=i)
                                    imp_iapd_list.append(imp_iapd)
                                    [y_data1,x_data1]=sep_input_output_data(result, attr=output_attr)
                                    try:
                                        [conf_mat,fm_group]=train_lg(x_data=x_data1,y_data=y_data1\
                                                                 ,sensitive_attr=sensitive_attr)
                                    except:
                                        continue
                                    conf_mat_list.append(conf_mat)
                                    fm_group_list.append(fm_group)
                                    

                                imp_iapd=Average(imp_iapd_list) 
                                if len(fm_group_list) !=0:
                                    
                                    [conf_mat_all,fm_group_all,conf_mat_all_avr,fm_group_all_avr]=\
                                        average_results(conf_mat_list=conf_mat_list, fm_group_list=fm_group_list)
                                    
                                    write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                      missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                          k_MCAR=k11,conf_mat_all_avr=conf_mat_all_avr,\
                                          fm_group_all_avr=fm_group_all_avr,path_csv=path_csv,A=A1,\
                                        alpha1_MNAR=alpha1,alpha2_MNAR=alpha2,\
                                        beta1_MNAR=beta1,beta2_MNAR=beta2, imp_IAPD=imp_iapd)
                                else:
                                    write_csv_MCAR(Dataset=Dataset,algorithm=algorithm,\
                                      missing_data_type=j,imputation=i,exp_no=no,test_size=test_size,\
                                          k_MCAR=k11,path_csv=path_csv,A=A1,\
                                        alpha1_MNAR=alpha1,alpha2_MNAR=alpha2,\
                                        beta1_MNAR=beta1,beta2_MNAR=beta2, imp_IAPD=imp_iapd)

def imp_del(dataset=None):
    """
    Description:
        delete rows with nan 
    Input:
        pandas df
    Output:
        dataset with deleted rows

    """
    dataset_copy=dataset.copy(deep=True)
    
    dataset_copy=dataset_copy.dropna()    
    return dataset_copy


def imp_knn(dataset=None,k_knn=None):
    """
    Description:
        KNN imputation
    Input:
        pandas df nan as missing data
        k= KNN parameter
    Output:
        dataset with imputed data

    """
    dataset_copy=dataset.copy(deep=True)
    
    column=dataset_copy.columns.values.tolist()
    
    dataset_copy=dataset_copy.to_numpy()
    
    knnImpute = KNN(k=k_knn)
    X_filled_knn = knnImpute.fit_transform(dataset_copy)
    
    df = pd.DataFrame(X_filled_knn, columns = column)
         
    return df



def imp_NNM(dataset=None):
    """
    Description:
        nuclear norm minimizer imputation
    Input:
        pandas df nan as missing data
    Output:
        dataset with imputed data

    """
    dataset_copy=dataset.copy(deep=True)
    
    column=dataset_copy.columns.values.tolist()
    
    dataset_copy=dataset_copy.to_numpy()
    
    X_filled_knn = NuclearNormMinimization().fit_transform(dataset_copy)
    
    df = pd.DataFrame(X_filled_knn, columns = column)
         
    return df


def imp_softImpute(dataset=None):
    """
    Description:
        Soft SVD imputation
    Input:
        pandas df nan as missing data

    Output:
        dataset with imputed data

    """
    dataset_copy=dataset.copy(deep=True)
    
    column=dataset_copy.columns.values.tolist()
    
    dataset_copy=dataset_copy.to_numpy()
    
    softImpute = SoftImpute()
    
    X_filled_knn = softImpute.fit_transform(dataset_copy)
    
    df = pd.DataFrame(X_filled_knn, columns = column)
         
    return df


def imp_iterativeImputer(dataset=None):
    """
    Description:
        Soft SVD imputation
    Input:
        pandas df nan as missing data

    Output:
        dataset with imputed data

    """
    dataset_copy=dataset.copy(deep=True)
    
    column=dataset_copy.columns.values.tolist()
    
    dataset_copy=dataset_copy.to_numpy()
    
    
    X_filled_knn = IterativeImputer().fit_transform(dataset_copy)
    
    df = pd.DataFrame(X_filled_knn, columns = column)
         
    return df

    

def imp_iterativeSVD(dataset=None):
    """
    Description:
        iterative SVD imputation
    Input:
        pandas df nan as missing data

    Output:
        dataset with imputed data

    """
    dataset_copy=dataset.copy(deep=True)
    
    column=dataset_copy.columns.values.tolist()
    
    dataset_copy=dataset_copy.to_numpy()
    
    
    X_filled_knn = IterativeSVD().fit_transform(dataset_copy)
    
    df = pd.DataFrame(X_filled_knn, columns = column)
         
    return df


def imp_MatrixFactorization(dataset=None):
    """
    Description:
        Soft SVD imputation
    Input:
        pandas df nan as missing data

    Output:
        dataset with imputed data

    """
    dataset_copy=dataset.copy(deep=True)
    
    column=dataset_copy.columns.values.tolist()
    
    dataset_copy=dataset_copy.to_numpy()
    
    
    X_filled_knn = MatrixFactorization().fit_transform(dataset_copy)
    
    df = pd.DataFrame(X_filled_knn, columns = column)
         
    return df


def is_categorical(array_like):
    """
    if a df column is categorical or not
    
    Output:
        true if categorical otherwise False
    """

    return array_like.dtype.name == 'category'

def is_binary(series, allow_na=True):
    
    """
    test if a series is binary or not
    Input:
        df column as series, df_nba['country']
    Output:
        true if binary
    """
    if allow_na:
        a=series.dropna(inplace=False)
    return sorted(series.unique()) == [0, 1]


def imp_mean(dataset=None):
    
    """
    impute nan values with mean of the column, if column is binary replace with most frequent values
    Input:
        df with missing values
    Output:
        df with missing values repalced with mean
    """
    
    dataset_copy=dataset.copy(deep=True)
      
    for col in dataset_copy:
        #print(col)
        
        if dataset_copy[col].isnull().values.any()==True:
            #print(col,'yes')
            
            if is_binary(dataset_copy[col]):
                #print('is_binary=yes',col)
                count_1 = dataset_copy[col][dataset_copy[col]==1].count()
                count_0 = dataset_copy[col][dataset_copy[col]==0].count()
                
                if count_0 > count_1:
                    dataset_copy[col].fillna(0,inplace=True)
                else:
                    dataset_copy[col].fillna(1,inplace=True)
                    
            dataset_copy[col].fillna(dataset_copy[col].mean(),inplace=True)
    return dataset_copy

  

def del_data_info(dataset=None,output_attr=None,sensitive_attr=None):
    
    """
    provide inforamtion about the missing df, weather output attribute has retained binary value, or length of the
    
    df is less than 2, or sensitive attribute has retained binary value after deleting data
    
    """
    dataset_copy=dataset.copy(deep=True)

    d1=dataset_copy[output_attr].dropna()
    d1=d1.to_frame()
    d2=dataset_copy[sensitive_attr].dropna()
    d2=d2.to_frame()
    
    #if len(dataset_copy) < 2: # if dataset has less than 2 values after deleting missing data
        #a=False
    if len(d1[output_attr].unique())==1: # if output attribute is non binary after missing data deletion
        a=False
    elif len(d2[sensitive_attr].unique())==1: # if sensitive attribute is non binary after missing data deletion
        a=False
    else:
        a=True
    return a

def del_data_info1(dataset=None,output_attr=None,sensitive_attr=None):
    
    """
    provide inforamtion about the missing df, weather output attribute has retained binary value, or length of the
    
    df is less than 2, or sensitive attribute has retained binary value after deleting data
    
    """
    dataset_copy=dataset.copy(deep=True)

    d1=dataset_copy[output_attr].dropna()
    d1=d1.to_frame()
    d2=dataset_copy[sensitive_attr].dropna()
    d2=d2.to_frame()
    
    #if len(dataset_copy) < 2: # if dataset has less than 2 values after deleting missing data
        #a=False
    if len(d1[output_attr].unique())==1: # if output attribute is non binary after missing data deletion
        a=False
    #elif len(d2[sensitive_attr].unique())==1: # if sensitive attribute is non binary after missing data deletion
        #a=False
    else:
        a=True
    return a



def Average(lst):
    lst1 = [i for i in lst if i]
    if len(lst1)!=0:
        return (sum(lst1) / len(lst1))
    else:
        return None

def perform_imputation(df=None,I=None):
    
    
    if I=='del': 
        result=imp_del(dataset=df)
    elif I=='imp_mean':
        result=imp_mean(dataset=df)
    elif I=='knn':
        result=imp_knn(dataset=df)
    elif I=='NNM':
        result=imp_NNM(dataset=df)
        
    elif I=='soft impute':
        result=imp_softImpute(dataset=df)
        
    elif I=='iterative imputer':
        result=imp_iterativeImputer(dataset=df)
    
    elif I=='iterative SVD':
        result=imp_iterativeSVD(dataset=df)

    elif I=='matrix factorization':
        result=imp_MatrixFactorization(dataset=df)        
        
    return result
    
    
#%% initialize variables
#algorithm 


#algorithm='GAT'
#algorithm='Node2Vec'  #logistic regression 
#algorithm='GCN'
#algorithm='LR'  
algorithm='fairwalk'  
#########################################
#dataset


Dataset='german'
#Dataset='NBA'  
#Dataset='credit'
#Dataset='bail'
#Dataset='pokec_z'
#Dataset='pokec_n'
#########################################
#missing_type


missing_type='MCAR'
#missing_type='MAR'  
#missing_type='MNAR'

########################################

no=5 # no of times experiemnt is repeated

test_size=0.3

scaler = preprocessing.MinMaxScaler()

base_path='C:\\Users\\HarisM\\Desktop'


if Dataset=='NBA':
    
    
    path_csv=".//results//"+ algorithm+' '+Dataset +' ' + missing_type +" results.csv"

    output_attr='SALARY'

    sensitive_attr='country'


    nba_data_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\NBA\\nba_processed.csv'
    
    #nba_data_path="C:\\Users\\haris\\Downloads\\fairness_graph_data_imputation\\datasets\\NBA\\nba.csv"
    
    df_nba=df = pd.read_csv(nba_data_path)
    
    edge_list_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\NBA\\nba_edges_processed.txt'
    
    df_edge_list=pd.read_csv(edge_list_path, sep=' ', header=None, names=["source", "target"])
    
    # data analysis with pandas
    
    #read this article:
    #    https://www.kaggle.com/code/kashnitsky/topic-1-exploratory-data-analysis-with-pandas
    
    
    #import seaborn as sns
    
    
    #f1=sns.countplot(x=output_attr, data=df_nba)
    
    #f2=sns.countplot(x=sensitive_attr, data=df_nba)
    
    #print(df_nba.info())
    
    #print('df_Shape=',df_nba.shape)
    
    #print(df_nba[output_attr].value_counts(normalize=True)) #salary values percentage
    
    #print(df_nba[sensitive_attr].value_counts(normalize=True)) #country values percentage
    
    #print(pd.crosstab(df_nba[output_attr], df_nba[sensitive_attr])) # cross table salary vs percentage
    
    
    # sort data with correlation wrt salary this is necessary, because we are assuming missingness from 
    # k most correltaed attributes and thye should be first k attributes, so we have to sort this by correlation
    
    [sorted_df,corr_df]=sort_correlation(df_nba, attr=output_attr,sensitive_attr=sensitive_attr)
    
    # input output attributes split
    [y_data,x_data]=sep_input_output_data(sorted_df, attr=output_attr)
    
    x_data=normalize_dataframe(x_data, Scaler=scaler)
        
    #X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size = test_size,stratify=y_data)
    
    #Y_train = Y_train.astype(int)
    #Y_test = Y_test.astype(int)
    
    #train_index=X_train.index.values.tolist()

    #test_index=X_test.index.values.tolist()
    
    #missing_data_type=['None','MCAR','MAR','MNAR']
    missing_data_type=['None',missing_type]

    #imputation=['del','imp_mean','knn','soft impute','iterative imputer','iterative SVD']

    imputation=['imp_mean','knn','soft impute','iterative imputer','iterative SVD']
    #imputation=['del']
    ###############################################
    #MCAR parameters

    theta_MCAR=np.linspace(0.1, 0.9, num=5) #MCAR probability
    theta_MCAR_k=np.linspace(0.1, 0.9, num=5) # when k is more than one, missing data in many attributes
    k11=[ 5,10, 15,20, 25,30]  #number of sttributes from which data is removed for missingness

    attr_type=['sensitive attribute only', 'from k attributes']
    # k1 number of sttributes from which data is removed for missingness

    ###

    #alpha1_MAR=np.linspace(-1, 1, num=42)
    #alpha2_MAR=np.linspace(-1, 1, num=42)
    #beta1_MAR=np.linspace(-1, 1, num=42)
    #beta2_MAR=np.linspace(0, 1, num=21)


    ################################################# 
    #MNAR parameters

    #alpha1_MNAR=[-0.1,0,0.1] #MCAR probability
    #beta1_MNAR=[-0.1,0,0.1] 
    #alpha2_MNAR=[-0.1,0,0.1] 
    #beta2_MNAR=[-0.1,0,0.1] 


    #alpha1_MNAR=np.linspace(-1, 1, num=42)
    #alpha2_MNAR=np.linspace(-1, 1, num=42)
    #beta1_MNAR=np.linspace(-1, 1, num=42)
    #beta2_MNAR=np.linspace(-1, 1, num=42)
    # k1 number of attributes from which data is removed for missingness


if Dataset=='german':
    
    
    path_csv=".//results//"+ algorithm+' '+Dataset +' ' + missing_type +" results.csv"

    output_attr='GoodCustomer'
    
    sensitive_attr='Gender'

    
    data_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\german\\german_processed.csv'

    edge_list_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\german\\german_edges_processed.txt'
    
    
    df=df = pd.read_csv(data_path)
    
    
    df_edge_list=pd.read_csv(edge_list_path, sep=' ', header=None, names=["source", "target"])
    
    
    [sorted_df,corr_df]=sort_correlation(df, attr=output_attr,sensitive_attr=sensitive_attr)
    
    # input output attributes split
    [y_data,x_data]=sep_input_output_data(sorted_df, attr=output_attr)
    
    x_data=normalize_dataframe(x_data, Scaler=scaler)
        
    #X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size = test_size,stratify=y_data)
    
    #Y_train = Y_train.astype(int)
    #Y_test = Y_test.astype(int)
    
    #train_index=X_train.index.values.tolist()

    #test_index=X_test.index.values.tolist()
    
    #missing_data_type=['None','MCAR','MAR','MNAR']
    missing_data_type=['None',missing_type]

    #imputation=['del','imp_mean','knn','soft impute','iterative imputer','iterative SVD']
    
    imputation=['imp_mean','knn','iterative SVD','soft impute']
    #imputation=['imp_mean','knn']
    #imputation=['del']
    ###############################################
    #MCAR parameters

    theta_MCAR=np.linspace(0.1, 0.9, num=5) #MCAR probability
    theta_MCAR_k=np.linspace(0.1, 0.9, num=5) # when k is more than one, missing data in many attributes
    k11=[ 2,10, 20]  #number of sttributes from which data is removed for missingness

    attr_type=['sensitive attribute only', 'from k attributes']
    # k1 number of sttributes from which data is removed for missingness

    ###

    #alpha1_MAR=np.linspace(-1, 1, num=42)
    #alpha2_MAR=np.linspace(-1, 1, num=42)
    #beta1_MAR=np.linspace(-1, 1, num=42)
    #beta2_MAR=np.linspace(0, 1, num=21)


    ################################################# 
    #MNAR parameters

    #alpha1_MNAR=[-0.1,0,0.1] #MCAR probability
    #beta1_MNAR=[-0.1,0,0.1] 
    #alpha2_MNAR=[-0.1,0,0.1] 
    #beta2_MNAR=[-0.1,0,0.1] 


    #alpha1_MNAR=np.linspace(-1, 1, num=42)
    #alpha2_MNAR=np.linspace(-1, 1, num=42)
    #beta1_MNAR=np.linspace(-1, 1, num=42)
    #beta2_MNAR=np.linspace(-1, 1, num=42)
    # k1 number of attributes from which data is removed for missingness
    
    
if Dataset=='credit':
    
    
    path_csv=".//results//"+ algorithm+' '+Dataset +' ' + missing_type +" results.csv"
    
    sensitive_attr="Age"

    output_attr="NoDefaultNextMonth"


    data_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\credit\\credit_processed.csv'

    edge_list_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\credit\\credit_edges_processed.txt'

    
    df=df = pd.read_csv(data_path)
    
    
    df_edge_list=pd.read_csv(edge_list_path, sep=' ', header=None, names=["source", "target"])
    
    
    [sorted_df,corr_df]=sort_correlation(df, attr=output_attr,sensitive_attr=sensitive_attr)
    
    # input output attributes split
    [y_data,x_data]=sep_input_output_data(sorted_df, attr=output_attr)
    
    x_data=normalize_dataframe(x_data, Scaler=scaler)
        
    #X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size = test_size,stratify=y_data)
    
    #Y_train = Y_train.astype(int)
    #Y_test = Y_test.astype(int)
    
    #train_index=X_train.index.values.tolist()

    #test_index=X_test.index.values.tolist()
    
    #missing_data_type=['None','MCAR','MAR','MNAR']
    missing_data_type=['None',missing_type]

    #imputation=['del','imp_mean','knn','soft impute','iterative imputer','iterative SVD']
    
    imputation=['imp_mean','knn','iterative SVD','soft impute']
    #imputation=['imp_mean','knn']
    #imputation=['del']
    ###############################################
    #MCAR parameters

    theta_MCAR=np.linspace(0.1, 0.9, num=5) #MCAR probability
    theta_MCAR_k=np.linspace(0.1, 0.9, num=5) # when k is more than one, missing data in many attributes
    k11=[ 2,10, 20]  #number of sttributes from which data is removed for missingness

    attr_type=['sensitive attribute only', 'from k attributes']
    # k1 number of sttributes from which data is removed for missingness

    ###

    #alpha1_MAR=np.linspace(-1, 1, num=42)
    #alpha2_MAR=np.linspace(-1, 1, num=42)
    #beta1_MAR=np.linspace(-1, 1, num=42)
    #beta2_MAR=np.linspace(0, 1, num=21)


    ################################################# 
    #MNAR parameters

    #alpha1_MNAR=[-0.1,0,0.1] #MCAR probability
    #beta1_MNAR=[-0.1,0,0.1] 
    #alpha2_MNAR=[-0.1,0,0.1] 
    #beta2_MNAR=[-0.1,0,0.1] 


    #alpha1_MNAR=np.linspace(-1, 1, num=42)
    #alpha2_MNAR=np.linspace(-1, 1, num=42)
    #beta1_MNAR=np.linspace(-1, 1, num=42)
    #beta2_MNAR=np.linspace(-1, 1, num=42)
    # k1 number of attributes from which data is removed for missingness
    
if Dataset=='bail':
        
    path_csv=".//results//"+ algorithm+' '+Dataset +' ' + missing_type +" results.csv"

    sensitive_attr="WHITE"
    
    output_attr="RECID"

    data_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\bail\\bail_processed.csv'

    edge_list_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\bail\\bail_edges_processed.txt'

    
    df=df = pd.read_csv(data_path)
    
    
    df_edge_list=pd.read_csv(edge_list_path, sep=' ', header=None, names=["source", "target"])
    
    
    [sorted_df,corr_df]=sort_correlation(df, attr=output_attr,sensitive_attr=sensitive_attr)
    
    # input output attributes split
    [y_data,x_data]=sep_input_output_data(sorted_df, attr=output_attr)
    
    x_data=normalize_dataframe(x_data, Scaler=scaler)
        
    #X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size = test_size,stratify=y_data)
    
    #Y_train = Y_train.astype(int)
    #Y_test = Y_test.astype(int)
    
    #train_index=X_train.index.values.tolist()

    #test_index=X_test.index.values.tolist()
    
    #missing_data_type=['None','MCAR','MAR','MNAR']
    missing_data_type=['None',missing_type]

    #imputation=['del','imp_mean','knn','soft impute','iterative imputer','iterative SVD']

    imputation=['imp_mean','knn','iterative SVD','soft impute']
    #imputation=['imp_mean','knn']
    #imputation=['del']
    ###############################################
    #MCAR parameters

    theta_MCAR=np.linspace(0.1, 0.9, num=5) #MCAR probability
    theta_MCAR_k=np.linspace(0.1, 0.9, num=5) # when k is more than one, missing data in many attributes
    k11=[ 2,9]  #number of sttributes from which data is removed for missingness

    attr_type=['sensitive attribute only', 'from k attributes']
    # k1 number of sttributes from which data is removed for missingness

    ###

    #alpha1_MAR=np.linspace(-1, 1, num=42)
    #alpha2_MAR=np.linspace(-1, 1, num=42)
    #beta1_MAR=np.linspace(-1, 1, num=42)
    #beta2_MAR=np.linspace(0, 1, num=21)


    ################################################# 
    #MNAR parameters

    #alpha1_MNAR=[-0.1,0,0.1] #MCAR probability
    #beta1_MNAR=[-0.1,0,0.1] 
    #alpha2_MNAR=[-0.1,0,0.1] 
    #beta2_MNAR=[-0.1,0,0.1] 


    #alpha1_MNAR=np.linspace(-1, 1, num=42)
    #alpha2_MNAR=np.linspace(-1, 1, num=42)
    #beta1_MNAR=np.linspace(-1, 1, num=42)
    #beta2_MNAR=np.linspace(-1, 1, num=42)
    # k1 number of attributes from which data is removed for missingness
    
if Dataset=='pokec_z':
        
    path_csv=".//results//"+ algorithm+' '+Dataset +' ' + missing_type +" results.csv"

    sensitive_attr="region"

    output_attr="I_am_working_in_field"


    data_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\pokec_z\\pokec_z_processed.csv'

    edge_list_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\pokec_z\\pokec_z_edges_processed.txt'

    df=df = pd.read_csv(data_path)
    
    df_edge_list=pd.read_csv(edge_list_path, sep=' ', header=None, names=["source", "target"])
    
    [sorted_df,corr_df]=sort_correlation(df, attr=output_attr,sensitive_attr=sensitive_attr)
    
    # input output attributes split
    [y_data,x_data]=sep_input_output_data(sorted_df, attr=output_attr)
    
    #x_data=normalize_dataframe(x_data_original, Scaler=scaler)
    
    x_scaled = scaler.fit_transform(x_data)
        
    #X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size = test_size,stratify=y_data)
    
    #Y_train = Y_train.astype(int)
    #Y_test = Y_test.astype(int)
    
    #train_index=X_train.index.values.tolist()

    #test_index=X_test.index.values.tolist()
    
    #missing_data_type=['None','MCAR','MAR','MNAR']
    missing_data_type=['None',missing_type]

    #imputation=['del','imp_mean','knn','soft impute','iterative imputer','iterative SVD']

    imputation=['imp_mean','knn','iterative SVD','soft impute']
    #imputation=['imp_mean','knn']
    #imputation=['del']
    ###############################################
    #MCAR parameters

    theta_MCAR=np.linspace(0.1, 0.9, num=5) #MCAR probability
    theta_MCAR_k=np.linspace(0.1, 0.9, num=5) # when k is more than one, missing data in many attributes
    k11=[ 13,26,39,52]  #number of sttributes from which data is removed for missingness

    attr_type=['sensitive attribute only', 'from k attributes']
    # k1 number of sttributes from which data is removed for missingness

    ###

    #alpha1_MAR=np.linspace(-1, 1, num=42)
    #alpha2_MAR=np.linspace(-1, 1, num=42)
    #beta1_MAR=np.linspace(-1, 1, num=42)
    #beta2_MAR=np.linspace(0, 1, num=21)


    ################################################# 
    #MNAR parameters

    #alpha1_MNAR=[-0.1,0,0.1] #MCAR probability
    #beta1_MNAR=[-0.1,0,0.1] 
    #alpha2_MNAR=[-0.1,0,0.1] 
    #beta2_MNAR=[-0.1,0,0.1] 


    #alpha1_MNAR=np.linspace(-1, 1, num=42)
    #alpha2_MNAR=np.linspace(-1, 1, num=42)
    #beta1_MNAR=np.linspace(-1, 1, num=42)
    #beta2_MNAR=np.linspace(-1, 1, num=42)
    # k1 number of attributes from which data is removed for missingness


if Dataset=='pokec_n':
        
    path_csv=".//results//"+ algorithm+' '+Dataset +' ' + missing_type +" results.csv"

    sensitive_attr="region"

    output_attr="I_am_working_in_field"

    data_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\pokec_n\\pokec_n_processed.csv'

    edge_list_path=base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\pokec_n\\pokec_n_edges_processed.txt'

    df=df = pd.read_csv(data_path)
    
    df_edge_list=pd.read_csv(edge_list_path, sep=' ', header=None, names=["source", "target"])
    
    [sorted_df,corr_df]=sort_correlation(df, attr=output_attr,sensitive_attr=sensitive_attr)
    
    # input output attributes split
    [y_data,x_data]=sep_input_output_data(sorted_df, attr=output_attr)
    
    x_data=normalize_dataframe(x_data, Scaler=scaler)
        
    #X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size = test_size,stratify=y_data)
    
    #Y_train = Y_train.astype(int)
    #Y_test = Y_test.astype(int)
    
    #train_index=X_train.index.values.tolist()

    #test_index=X_test.index.values.tolist()
    
    #missing_data_type=['None','MCAR','MAR','MNAR']
    missing_data_type=['None',missing_type]

    #imputation=['del','imp_mean','knn','soft impute','iterative imputer','iterative SVD']

    imputation=['imp_mean','knn','iterative SVD','soft impute']
    #imputation=['imp_mean','knn']
    #imputation=['del']
    ###############################################
    #MCAR parameters
    
    theta_MCAR=np.linspace(0.1, 0.9, num=5) #MCAR probability
    theta_MCAR_k=np.linspace(0.1, 0.9, num=5) # when k is more than one, missing data in many attributes
    k11=[ 13,26,39,52]  #number of sttributes from which data is removed for missingness

    attr_type=['sensitive attribute only', 'from k attributes']
    # k1 number of sttributes from which data is removed for missingness

    ###

    #alpha1_MAR=np.linspace(-1, 1, num=42)
    #alpha2_MAR=np.linspace(-1, 1, num=42)
    #beta1_MAR=np.linspace(-1, 1, num=42)
    #beta2_MAR=np.linspace(0, 1, num=21)


    ################################################# 
    #MNAR parameters

    #alpha1_MNAR=[-0.1,0,0.1] #MCAR probability
    #beta1_MNAR=[-0.1,0,0.1] 
    #alpha2_MNAR=[-0.1,0,0.1] 
    #beta2_MNAR=[-0.1,0,0.1] 


    #alpha1_MNAR=np.linspace(-1, 1, num=42)
    #alpha2_MNAR=np.linspace(-1, 1, num=42)
    #beta1_MNAR=np.linspace(-1, 1, num=42)
    #beta2_MNAR=np.linspace(-1, 1, num=42)
    # k1 number of attributes from which data is removed for missingness


#%%



for j in missing_data_type:
    
        
    if j=='MCAR':
        
        print('MCAR')
        
        MCAR_imputation(XX_data=x_data,YY_data=y_data,k1=k11,theta_MCAR_k=theta_MCAR_k,\
                            theta_MCAR=theta_MCAR,imputation=imputation,attr_type=attr_type,no=no,\
                            output_attr=output_attr,sensitive_attr=sensitive_attr)

    elif j=='MAR':
        print('MAR')
        
        MAR_imputation(x_data=x_data,y_data=y_data,k1=k1,imputation=imputation,attr_type=attr_type,no=no,\
                            output_attr=output_attr,sensitive_attr=sensitive_attr, alpha1_MAR=alpha1_MAR,\
                            alpha2_MAR=alpha2_MAR,beta1_MAR=beta1_MAR,beta2_MAR=beta2_MAR)
    elif j=='MNAR':
        
        print('MNAR')
        
        MNAR_imputation(x_data=x_data,y_data=y_data,k1=k1,imputation=imputation,attr_type=attr_type,no=no,\
                            output_attr=output_attr,sensitive_attr=sensitive_attr, alpha1_MNAR=alpha1_MNAR,\
                            alpha2_MNAR=alpha2_MNAR,beta1_MNAR=beta1_MNAR,beta2_MNAR=beta2_MNAR)


