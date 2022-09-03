# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 19:00:27 2022

@author: Haris
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:59:19 2022

@author: haris
"""

import sys, os
file_path=os.path.dirname(os.path.abspath(__file__))
sys.path
sys.path.append(file_path)
os.chdir(file_path)

#%%
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import fairlearn
from fairlearn.metrics import MetricFrame


import random
from sklearn.metrics import accuracy_score, recall_score,precision_score

import sys
#if 'google.colab' in sys.modules:
#   %pip install -q stellargraph[demos]==1.2.1

# verify that we're using the correct version of StellarGraph for this notebook
import stellargraph as sg

try:
    sg.utils.validate_notebook_version("1.2.1")
except AttributeError:
    raise ValueError(
        f"This notebook requires StellarGraph version 1.2.1, but a different version {sg.__version__} is installed.  Please see <https://github.com/stellargraph/stellargraph/issues/1172>."
    ) from None

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph


import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN, GAT

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt


#%% fair walk
base_path='C:\\Users\\HarisM\\Desktop'


def train_failwalk(X_data=None,Y_data=None,df_edge_list=None,train_index=None,test_index=None,d_name=None):
    """## Downstream task
    
    The node embeddings calculated using `Word2Vec` can be used as feature vectors in a downstream task such as node attribute inference. 
    
    In this example, we will use the `fairwalk` node embeddings to train a classifier to predict the subject of a paper in Cora.
    """
    #[graph_dataset,y]=stellargraph_nba(dataframe=df,dataframe_edge_list=df_edge_list)

    

    if d_name=='NBA':

        X = np.load(base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\NBA\\nba_fairwalk_embeddings.npy')
        
    elif d_name=='german':

        X = np.load(base_path+'\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\german\\german_fairwalk_embeddings.npy')

        
    elif d_name=='credit': 
        None
        
    elif d_name=='bail': 
        None        
        
    elif d_name=='pokec_z': 
        None        
        
    elif d_name=='pokec_n': 
        None
    # X will hold the 128-dimensional input features
 
        
    # y holds the corresponding target values
    #y = np.array(node_targets)
    
    #x_training_data=X.iloc[train_index] 
    #x_test_data=X.iloc[test_index]
    
    x_training_data=X[[train_index]]
    x_test_data=X[[test_index]]
    
    
    y_training_data=Y_data.iloc[train_index]
    y_test_data=Y_data.iloc[test_index]
    
    #x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(X, y, test_size=0.2,shuffle=False)
    


    
    model = LogisticRegression()

    model.fit(x_training_data, y_training_data)

    predictions = model.predict(x_test_data)
    
    print(classification_report(y_test_data, predictions))

    report=classification_report(y_test_data, predictions,output_dict=True)

    conf_mat=confusion_matrix(y_test_data, predictions)
    
    return predictions,report,conf_mat, model






#%% node2vec nba




def node2Vec_emb(graph=None,y=None, save=None, emb_save_path=None):
    
    G=graph
    node_subjects =y 
    
    #print(G.info())
    

    
    rw = BiasedRandomWalk(G)
    
    walks = rw.run(
        nodes=list(G.nodes()),  # root nodes
        length=100,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    )
    print("Number of random walks: {}".format(len(walks)))
    
    
    from gensim.models import Word2Vec
    
    str_walks = [[str(n) for n in walk] for walk in walks]
    
    model = Word2Vec(str_walks, vector_size =128, window=5, min_count=0, sg=1, workers=2, epochs=1)
    
    
    # Retrieve node embeddings and corresponding subjects
    node_ids = model.wv.index_to_key  # list of node IDs
    node_embeddings = (
        model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    node_targets = node_subjects[[int(node_id) for node_id in node_ids]]
    
    # Apply t-SNE transformation on node embeddings
    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(node_embeddings)
    
    # draw the points
    alpha = 0.7
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colours = [label_map[target] for target in node_targets]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=node_colours,
        cmap="jet",
        alpha=alpha,
    )


    if save==True:
        np.save(emb_save_path,node_embeddings, allow_pickle=False, fix_imports=False)
        
    return node_embeddings,node_targets
 
def stellargraph_pokec_z(dataframe=None,dataframe_edge_list=None):
    
    #df = pd.read_csv(path_df)
    
    df=dataframe.copy(deep=True)
    
    #df_edge_list=pd.read_csv(path_edge_list, sep='\t', header=None, names=["source", "target"])
    
    df_edge_list=dataframe_edge_list.copy(deep=True)
    
    output=df['I_am_working_in_field']
    
    df.drop('I_am_working_in_field', inplace=True, axis=1)
    
    #graph = StellarGraph({"players": df}, {"player_edges": df_edge_list})
    graph=StellarGraph(df, df_edge_list)
    
    print(graph.info())
    
    return graph,output

def stellargraph_pokec_n(dataframe=None,dataframe_edge_list=None):
    
    #df = pd.read_csv(path_df)
    
    df=dataframe.copy(deep=True)
    
    df_edge_list=dataframe_edge_list.copy(deep=True)

    output=df['I_am_working_in_field']
    
    df.drop('I_am_working_in_field', inplace=True, axis=1)
    
    #graph = StellarGraph({"players": df}, {"player_edges": df_edge_list})
    graph=StellarGraph(df, df_edge_list)
    
    print(graph.info())
    
    return graph,output       

def stellargraph_nba(dataframe=None,dataframe_edge_list=None):
    
    #df = pd.read_csv(path_df)
    
    df=dataframe.copy(deep=True)
    
    #df_edge_list=pd.read_csv(path_edge_list, sep='\t', header=None, names=["source", "target"])
    
    df_edge_list=dataframe_edge_list.copy(deep=True)
    
    #df = df.set_index("user_id")
    #a=df[df['SALARY'] == -1].index.values
    #df_edge_list = df_edge_list[~df_edge_list['target'].isin(a)]
    #df_edge_list = df_edge_list[~df_edge_list['source'].isin(a)]
    
    #df.drop(df.index[df['SALARY'] == -1], inplace=True)
    
    output=df['SALARY']
    
    df.drop('SALARY', inplace=True, axis=1)
    
    graph = StellarGraph({"players": df}, {"player_edges": df_edge_list})
    
    
    print(graph.info())
    
    return graph,output


def train_node2vec_nba(X_data=None,Y_data=None,df_edge_list=None,train_index=None,test_index=None):
    """## Downstream task
    
    The node embeddings calculated using `Word2Vec` can be used as feature vectors in a downstream task such as node attribute inference. 
    
    In this example, we will use the `Node2Vec` node embeddings to train a classifier to predict the subject of a paper in Cora.
    """
    #[graph_dataset,y]=stellargraph_nba(dataframe=df,dataframe_edge_list=df_edge_list)



    #[node_embeddings,node_targets]=node2Vec_emb(graph=graph_dataset,y=y, save=False, emb_save_path=None)

    # X will hold the 128-dimensional input features
    X = np.load('C:\\Users\\haris\\Downloads\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\NBA\\nba_node2vec_embeddings.npy')
    # y holds the corresponding target values
    #y = np.array(node_targets)
    
    #x_training_data=X.iloc[train_index] 
    #x_test_data=X.iloc[test_index]
    
    x_training_data=X[[train_index]]
    x_test_data=X[[test_index]]
    
    
    y_training_data=Y_data.iloc[train_index]
    y_test_data=Y_data.iloc[test_index]
    
    #x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(X, y, test_size=0.2,shuffle=False)
    


    
    model = LogisticRegression()

    model.fit(x_training_data, y_training_data)

    predictions = model.predict(x_test_data)
    
    print(classification_report(y_test_data, predictions))

    report=classification_report(y_test_data, predictions,output_dict=True)

    conf_mat=confusion_matrix(y_test_data, predictions)
    
    return predictions,report,conf_mat, model

def train_node2vec_pokec_z(X_data=None,Y_data=None,df_edge_list=None,train_index=None,test_index=None):
    """## Downstream task
    
    The node embeddings calculated using `Word2Vec` can be used as feature vectors in a downstream task such as node attribute inference. 
    
    In this example, we will use the `Node2Vec` node embeddings to train a classifier to predict the subject of a paper in Cora.
    """
    #[graph_dataset,y]=stellargraph_pokec_z(dataframe=df,dataframe_edge_list=df_edge_list)

    #[node_embeddings,node_targets]=node2Vec_emb(graph=graph_dataset,y=y, save=True, emb_save_path='./nba_node2vec_embeddings.npy')

    # X will hold the 128-dimensional input features
    X = np.load('C:\\Users\\haris\\Downloads\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\pokec_z\pokec_z_node2vec_embeddings.npy')
    # y holds the corresponding target values
    x_training_data=X[[train_index]]
    x_test_data=X[[test_index]]
    
    
    y_training_data=Y_data.iloc[train_index]
    y_test_data=Y_data.iloc[test_index]
    
    #x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(X, y, test_size=0.2,shuffle=False)
    
    model = LogisticRegression()

    model.fit(x_training_data, y_training_data)

    predictions = model.predict(x_test_data)
    
    print(classification_report(y_test_data, predictions))

    report=classification_report(y_test_data, predictions,output_dict=True)

    conf_mat=confusion_matrix(y_test_data, predictions)
    
    return predictions,report,conf_mat, model

def train_node2vec_pokec_n(X_data=None,Y_data=None,df_edge_list=None,train_index=None,test_index=None):
    """## Downstream task
    
    The node embeddings calculated using `Word2Vec` can be used as feature vectors in a downstream task such as node attribute inference. 
    
    In this example, we will use the `Node2Vec` node embeddings to train a classifier to predict the subject of a paper in Cora.
    """
    #[graph_dataset,y]=stellargraph_pokec_n(dataframe=df,dataframe_edge_list=df_edge_list)

    #[node_embeddings,node_targets]=node2Vec_emb(graph=graph_dataset,y=y, save=True, emb_save_path='./nba_node2vec_embeddings.npy')

    # X will hold the 128-dimensional input features
    X = np.load('C:\\Users\\haris\\Downloads\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\pokec_n\pokec_n_node2vec_embeddings.npy')
    # y holds the corresponding target values
    x_training_data=X[[train_index]]
    x_test_data=X[[test_index]]
    
    
    y_training_data=Y_data.iloc[train_index]
    y_test_data=Y_data.iloc[test_index]
    
    model = LogisticRegression()

    model.fit(x_training_data, y_training_data)

    predictions = model.predict(x_test_data)
    
    print(classification_report(y_test_data, predictions))

    report=classification_report(y_test_data, predictions,output_dict=True)

    conf_mat=confusion_matrix(y_test_data, predictions)
    
    return predictions,report,conf_mat, model
################################################node 2 vec german

def stellargraph_german(dataframe=None,dataframe_edge_list=None):
    
    df=dataframe.copy(deep=True)
    
    #df_edge_list=pd.read_csv(path_edge_list, sep='\t', header=None, names=["source", "target"])
    
    df_edge_list=dataframe_edge_list.copy(deep=True)
    
    output=df['GoodCustomer']
    
    df.drop('GoodCustomer', inplace=True, axis=1)
    
    graph = StellarGraph({"customer": df}, {"customer_edges": df_edge_list})
    
    
    print(graph.info())
    
    return graph,output



def train_node2vec_german(X_data=None,Y_data=None,df_edge_list=None,train_index=None,test_index=None):
    """## Downstream task
    
    The node embeddings calculated using `Word2Vec` can be used as feature vectors in a downstream task such as node attribute inference. 
    
    In this example, we will use the `Node2Vec` node embeddings to train a classifier to predict the subject of a paper in Cora.
    """
    #[graph_dataset,y]=stellargraph_german(dataframe=df,dataframe_edge_list=df_edge_list)



    #[node_embeddings,node_targets]=node2Vec_emb(graph=graph_dataset,y=y, save=False, emb_save_path=None)

    # X will hold the 128-dimensional input features
    #X = node_embeddings
    # y holds the corresponding target values
   # y = np.array(node_targets)
    
    #x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(X, y, test_size=0.2,shuffle=False)
    
    X = np.load('C:\\Users\\haris\\Downloads\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\german\\german_node2vec_embeddings.npy')
    # y holds the corresponding target values
    #y = np.array(node_targets)
    
    #x_training_data=X.iloc[train_index] 
    #x_test_data=X.iloc[test_index]
    
    x_training_data=X[[train_index]]
    x_test_data=X[[test_index]]
    
    
    y_training_data=Y_data.iloc[train_index]
    y_test_data=Y_data.iloc[test_index]
    
    
    
    model = LogisticRegression()
    
    model.fit(x_training_data, y_training_data)
    
    predictions = model.predict(x_test_data)
    
    print(classification_report(y_test_data, predictions))
    
    report=classification_report(y_test_data, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(y_test_data, predictions)
    
    
    
    return predictions,report,conf_mat, model
#%%node2vec credit


def stellargraph_credit(dataframe=None,dataframe_edge_list=None):
    
    df=dataframe.copy(deep=True)
    
    #df_edge_list=pd.read_csv(path_edge_list, sep='\t', header=None, names=["source", "target"]) 
    
    df_edge_list=dataframe_edge_list.copy(deep=True)
    
    output=df['NoDefaultNextMonth']
    
    df.drop('NoDefaultNextMonth', inplace=True, axis=1)
    
    graph = StellarGraph({"customer": df}, {"customer_edges": df_edge_list})
    
    
    print(graph.info())
    
    return graph,output


def train_node2vec_credit(X_data=None,Y_data=None,df_edge_list=None,train_index=None,test_index=None):
    """## Downstream task
    
    The node embeddings calculated using `Word2Vec` can be used as feature vectors in a downstream task such as node attribute inference. 
    
    In this example, we will use the `Node2Vec` node embeddings to train a classifier to predict the subject of a paper in Cora.
    
    """
    
    
    #[graph_dataset,y]=stellargraph_credit(dataframe=df,dataframe_edge_list=df_edge_list)

    #[node_embeddings,node_targets]=node2Vec_emb(graph=graph_dataset,y=y, save=True, emb_save_path='./credit_node2vec_embeddings.npy')

    # X will hold the 128-dimensional input features
    X = np.load('C:\\Users\\haris\\Downloads\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\credit\\credit_node2vec_embeddings.npy')
    # y holds the corresponding target values
    #y = np.array(node_targets)
    
    #x_training_data=X.iloc[train_index] 
    #x_test_data=X.iloc[test_index]
    
    x_training_data=X[[train_index]]
    x_test_data=X[[test_index]]
    
    y_training_data=Y_data.iloc[train_index]
    y_test_data=Y_data.iloc[test_index]
    
    
    model = LogisticRegression()
    
    model.fit(x_training_data, y_training_data)
    
    predictions = model.predict(x_test_data)
    
    print(classification_report(y_test_data, predictions))
    
    report=classification_report(y_test_data, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(y_test_data, predictions)
    
    
    
    return predictions,report,conf_mat, model

#%% node2vec bail

def stellargraph_bail(dataframe=None,dataframe_edge_list=None):
    
    df=dataframe.copy(deep=True)
    
    #df_edge_list=pd.read_csv(path_edge_list, sep='\t', header=None, names=["source", "target"]) 
    
    df_edge_list=dataframe_edge_list.copy(deep=True)
    
    output=df['RECID']
    
    df.drop('RECID', inplace=True, axis=1)
    
    graph = StellarGraph({"customer": df}, {"customer_edges": df_edge_list})
    
    
    print(graph.info())
    
    return graph,output

def train_node2vec_bail(X_data=None,Y_data=None,df_edge_list=None,train_index=None,test_index=None):
    """## Downstream task
    
    The node embeddings calculated using `Word2Vec` can be used as feature vectors in a downstream task such as node attribute inference. 
    
    In this example, we will use the `Node2Vec` node embeddings to train a classifier to predict the subject of a paper in Cora.
    
    """
    
    
   # [graph_dataset,y]=stellargraph_bail(dataframe=df,dataframe_edge_list=df_edge_list)

   # [node_embeddings,node_targets]=node2Vec_emb(graph=graph_dataset,y=y, save=True, emb_save_path='./bail_node2vec_embeddings.npy')

    # X will hold the 128-dimensional input features
    X = np.load('C:\\Users\\haris\\Downloads\\fairness_graph_data_imputation\\fairness_graph_data_imputation\\datasets\\bail\\bail_node2vec_embeddings.npy')
    # y holds the corresponding target values
    #y = np.array(node_targets)
    
    #x_training_data=X.iloc[train_index] 
    #x_test_data=X.iloc[test_index]
    
    x_training_data=X[[train_index]]
    x_test_data=X[[test_index]]
    
    y_training_data=Y_data.iloc[train_index]
    y_test_data=Y_data.iloc[test_index]
    
    
    
    model = LogisticRegression()
    
    model.fit(x_training_data, y_training_data)
    
    predictions = model.predict(x_test_data)
    
    print(classification_report(y_test_data, predictions))
    
    report=classification_report(y_test_data, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(y_test_data, predictions)
    
    
    
    return predictions,report,conf_mat, model

#%% gcn nba

def train_gcn_pokec_z(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):

    [G,node_subjects]=stellargraph_pokec_z(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(
        Y_train, test_size=0.3, stratify=Y_train)
    
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    
    generator = FullBatchNodeGenerator(G, method="gcn")
    
    
    train_gen = generator.flow(train_subjects.index, train_targets)
    
    
    gcn = GCN(
        layer_sizes=[8, 8], activations=["relu", "relu"], generator=generator, dropout=0.5
    )
    
    
    x_inp, x_out = gcn.in_out_tensors()
    
    #x_out
    
    
    predictions = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)
    
    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    
    from tensorflow.keras.callbacks import EarlyStopping
    
    es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
    
    
    history = model.fit(
        train_gen,
        epochs=2000,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback],
    )
    
    
    sg.utils.plot_history(history)
    
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    
    all_predictions = model.predict(test_gen)
    
    
    predictions = target_encoding.inverse_transform(all_predictions.squeeze())

        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model

def train_gcn_pokec_n(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):

    [G,node_subjects]=stellargraph_pokec_n(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(
        Y_train, test_size=0.3, stratify=Y_train)
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    generator = FullBatchNodeGenerator(G, method="gcn")
    
    train_gen = generator.flow(train_subjects.index, train_targets)
    
    gcn = GCN(
        layer_sizes=[8, 8], activations=["relu", "relu"], generator=generator, dropout=0.5
    )
    
    x_inp, x_out = gcn.in_out_tensors()
    
    predictions = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)
    
    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    
    from tensorflow.keras.callbacks import EarlyStopping
    
    es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
    
    
    history = model.fit(
        train_gen,
        epochs=2000,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback],
    )
    
    
    sg.utils.plot_history(history)
    
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    
    all_predictions = model.predict(test_gen)
    
    
    predictions = target_encoding.inverse_transform(all_predictions.squeeze())

        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model

def train_gcn_nba(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):

    [G,node_subjects]=stellargraph_nba(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(
        Y_train, test_size=0.3, stratify=Y_train)
    
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    
    generator = FullBatchNodeGenerator(G, method="gcn")
    
    
    train_gen = generator.flow(train_subjects.index, train_targets)
    
    
    gcn = GCN(
        layer_sizes=[8, 8], activations=["relu", "relu"], generator=generator, dropout=0.5
    )
    
    
    x_inp, x_out = gcn.in_out_tensors()
    
    #x_out
    
    
    predictions = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)
    
    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    
    from tensorflow.keras.callbacks import EarlyStopping
    
    es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
    
    
    history = model.fit(
        train_gen,
        epochs=2000,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback],
    )
    
    
    sg.utils.plot_history(history)
    
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    
    all_predictions = model.predict(test_gen)
    
    
    predictions = target_encoding.inverse_transform(all_predictions.squeeze())

        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model


def train_gcn_german(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):

    [G,node_subjects]=stellargraph_german(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(
        Y_train, test_size=0.3, stratify=Y_train)
    
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    generator = FullBatchNodeGenerator(G, method="gcn")
    
    train_gen = generator.flow(train_subjects.index, train_targets)
    
    gcn = GCN(
        layer_sizes=[8, 8], activations=["relu", "relu"], generator=generator, dropout=0.5
        #layer_sizes=[8, 8], activations=["sigmoid", "sigmoid"], generator=generator, dropout=0.5

    )
    
    x_inp, x_out = gcn.in_out_tensors()
    
    predictions = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)
    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    from tensorflow.keras.callbacks import EarlyStopping
    
    es_callback = EarlyStopping(monitor="val_acc", patience=100, restore_best_weights=True)
    
    history = model.fit(
        train_gen,
        epochs=2000,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback])
    
    sg.utils.plot_history(history)
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    all_predictions = model.predict(test_gen)
    
    predictions = target_encoding.inverse_transform(all_predictions.squeeze())
        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model


def train_gcn_credit(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):

    [G,node_subjects]=stellargraph_credit(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(
        Y_train, test_size=0.2, stratify=Y_train)
    
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    
    generator = FullBatchNodeGenerator(G, method="gcn")
    
    
    train_gen = generator.flow(train_subjects.index, train_targets)
    
    
    gcn = GCN(
        layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
    )
    
    
    x_inp, x_out = gcn.in_out_tensors()
    
    #x_out
    
    
    predictions = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)
    
    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    
    from tensorflow.keras.callbacks import EarlyStopping
    
    es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
    
    
    history = model.fit(
        train_gen,
        epochs=2000,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback],
    )
    
    
    sg.utils.plot_history(history)
    
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    
    all_predictions = model.predict(test_gen)
    
    
    predictions = target_encoding.inverse_transform(all_predictions.squeeze())

        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model

def train_gcn_bail(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):

    [G,node_subjects]=stellargraph_bail(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(
        Y_train, test_size=0.2, stratify=Y_train)
    
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    
    generator = FullBatchNodeGenerator(G, method="gcn")
    
    
    train_gen = generator.flow(train_subjects.index, train_targets)
    
    
    gcn = GCN(
        layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.5
    )
    
    
    x_inp, x_out = gcn.in_out_tensors()
    
    #x_out
    
    
    predictions = layers.Dense(units=train_targets.shape[1], activation="sigmoid")(x_out)
    
    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    
    from tensorflow.keras.callbacks import EarlyStopping
    
    es_callback = EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
    
    
    history = model.fit(
        train_gen,
        epochs=2000,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback],
    )
    
    
    sg.utils.plot_history(history)
    
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    
    all_predictions = model.predict(test_gen)
    
    
    predictions = target_encoding.inverse_transform(all_predictions.squeeze())

        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model

def train_gat_nba(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):


    [G,node_subjects]=stellargraph_nba(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(Y_train, test_size=0.5, stratify=Y_train)
    
    
    #train_subjects, val_subjects = model_selection.train_test_split(Y_train, test_size=0.7)
    
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    
    
    generator = FullBatchNodeGenerator(G, method="gat")
    

    train_gen = generator.flow(train_subjects.index, train_targets)
    
    """Now we can specify our machine learning model, we need a few more parameters for this:
    
     * the `layer_sizes` is a list of hidden feature sizes of each layer in the model. In this example we use two GAT layers with 8-dimensional hidden 
     node features for the first layer and the 7 class classification output for the second layer.
     * `attn_heads` is the number of attention heads in all but the last GAT layer in the model
     * `activations` is a list of activations applied to each layer's output
     * Arguments such as `bias`, `in_dropout`, `attn_dropout` are internal parameters of the model, execute `?GAT` for details.
    
    To follow the GAT model architecture used for Cora dataset in the original paper [Graph Attention Networks. P. 
    Veličković et al. ICLR 2018 https://arxiv.org/abs/1710.10903], let's build a 2-layer GAT model, with the second layer 
    being the classifier that predicts paper subject: it thus should have the output size of `train_targets.shape[1]` (7 subjects) and a softmax activation.
    """
    
    gat = GAT(
        layer_sizes=[16,16, train_targets.shape[1]],
        activations=["elu", "elu","sigmoid"],
        attn_heads=8,
        generator=generator,
        in_dropout=0.5,
        attn_dropout=0.5,
        normalize=None,
    )
    
    
    x_inp, predictions = gat.in_out_tensors()
    

    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    """Train the model, keeping track of its loss and accuracy on the training set, and its generalisation performance on the validation set (we need to create another generator over the validation data for this)"""
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    """Create callbacks for early stopping (if validation accuracy stops improving) and best model checkpoint saving:"""
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    es_callback = EarlyStopping(
        monitor="val_acc", patience=100
    )  # patience is the number of epochs to wait before early stopping in case of no further improvement
    mc_callback = ModelCheckpoint(
        "logs/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True
    )
    
    """Train the model"""
    
    history = model.fit(
        train_gen,
        epochs=500,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback, mc_callback],
    )
    
    """Plot the training history:"""
    
    sg.utils.plot_history(history)
    
    """Reload the saved weights of the best model found during the training (according to validation accuracy)"""
    
    model.load_weights("logs/best_model.h5")
    
    """Evaluate the best model on the test set"""
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    """### Making predictions with the model
    
    Now let's get the predictions for all nodes:
    """
    
    test_predictions = model.predict(test_gen)
    
    
    predictions = target_encoding.inverse_transform(test_predictions.squeeze())

        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model


def train_gat_german(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):


    [G,node_subjects]=stellargraph_german(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(Y_train, test_size=0.5, stratify=Y_train)
    
    
    #train_subjects, val_subjects = model_selection.train_test_split(Y_train, test_size=0.7)
    
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    
    
    generator = FullBatchNodeGenerator(G, method="gat")
    

    train_gen = generator.flow(train_subjects.index, train_targets)
    
    """Now we can specify our machine learning model, we need a few more parameters for this:
    
     * the `layer_sizes` is a list of hidden feature sizes of each layer in the model. In this example we use two GAT layers with 8-dimensional hidden 
     node features for the first layer and the 7 class classification output for the second layer.
     * `attn_heads` is the number of attention heads in all but the last GAT layer in the model
     * `activations` is a list of activations applied to each layer's output
     * Arguments such as `bias`, `in_dropout`, `attn_dropout` are internal parameters of the model, execute `?GAT` for details.
    
    To follow the GAT model architecture used for Cora dataset in the original paper [Graph Attention Networks. P. 
    Veličković et al. ICLR 2018 https://arxiv.org/abs/1710.10903], let's build a 2-layer GAT model, with the second layer 
    being the classifier that predicts paper subject: it thus should have the output size of `train_targets.shape[1]` (7 subjects) and a softmax activation.
    """
    
    gat = GAT(
        layer_sizes=[16,16, train_targets.shape[1]],
        activations=["elu", "elu","sigmoid"],
        attn_heads=8,
        generator=generator,
        in_dropout=0.5,
        attn_dropout=0.5,
        normalize=None,
    )
    
    
    x_inp, predictions = gat.in_out_tensors()
    

    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    """Train the model, keeping track of its loss and accuracy on the training set, and its generalisation performance on the validation set (we need to create another generator over the validation data for this)"""
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    """Create callbacks for early stopping (if validation accuracy stops improving) and best model checkpoint saving:"""
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    es_callback = EarlyStopping(
        monitor="val_acc", patience=100
    )  # patience is the number of epochs to wait before early stopping in case of no further improvement
    mc_callback = ModelCheckpoint(
        "logs/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True
    )
    
    """Train the model"""
    
    history = model.fit(
        train_gen,
        epochs=500,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback, mc_callback],
    )
    
    """Plot the training history:"""
    
    sg.utils.plot_history(history)
    
    """Reload the saved weights of the best model found during the training (according to validation accuracy)"""
    
    model.load_weights("logs/best_model.h5")
    
    """Evaluate the best model on the test set"""
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    """### Making predictions with the model
    
    Now let's get the predictions for all nodes:
    """
    
    test_predictions = model.predict(test_gen)
    
    
    predictions = target_encoding.inverse_transform(test_predictions.squeeze())

        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model


def train_gat_credit(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):


    [G,node_subjects]=stellargraph_credit(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(Y_train, test_size=0.5, stratify=Y_train)
    
    
    #train_subjects, val_subjects = model_selection.train_test_split(Y_train, test_size=0.7)
    
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    
    
    generator = FullBatchNodeGenerator(G, method="gat")
    

    train_gen = generator.flow(train_subjects.index, train_targets)
    
    """Now we can specify our machine learning model, we need a few more parameters for this:
    
     * the `layer_sizes` is a list of hidden feature sizes of each layer in the model. In this example we use two GAT layers with 8-dimensional hidden 
     node features for the first layer and the 7 class classification output for the second layer.
     * `attn_heads` is the number of attention heads in all but the last GAT layer in the model
     * `activations` is a list of activations applied to each layer's output
     * Arguments such as `bias`, `in_dropout`, `attn_dropout` are internal parameters of the model, execute `?GAT` for details.
    
    To follow the GAT model architecture used for Cora dataset in the original paper [Graph Attention Networks. P. 
    Veličković et al. ICLR 2018 https://arxiv.org/abs/1710.10903], let's build a 2-layer GAT model, with the second layer 
    being the classifier that predicts paper subject: it thus should have the output size of `train_targets.shape[1]` (7 subjects) and a softmax activation.
    """
    
    gat = GAT(
        layer_sizes=[16,16, train_targets.shape[1]],
        activations=["elu", "elu","sigmoid"],
        attn_heads=8,
        generator=generator,
        in_dropout=0.5,
        attn_dropout=0.5,
        normalize=None,
    )
    
    
    x_inp, predictions = gat.in_out_tensors()
    

    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    """Train the model, keeping track of its loss and accuracy on the training set, and its generalisation performance on the validation set (we need to create another generator over the validation data for this)"""
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    """Create callbacks for early stopping (if validation accuracy stops improving) and best model checkpoint saving:"""
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    es_callback = EarlyStopping(
        monitor="val_acc", patience=100
    )  # patience is the number of epochs to wait before early stopping in case of no further improvement
    mc_callback = ModelCheckpoint(
        "logs/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True
    )
    
    """Train the model"""
    
    history = model.fit(
        train_gen,
        epochs=500,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback, mc_callback],
    )
    
    """Plot the training history:"""
    
    sg.utils.plot_history(history)
    
    """Reload the saved weights of the best model found during the training (according to validation accuracy)"""
    
    model.load_weights("logs/best_model.h5")
    
    """Evaluate the best model on the test set"""
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    """### Making predictions with the model
    
    Now let's get the predictions for all nodes:
    """
    
    test_predictions = model.predict(test_gen)
    
    
    predictions = target_encoding.inverse_transform(test_predictions.squeeze())

        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model

def train_gat_bail(Dataset=None, df_edge_list=None, X_train=None, X_test=None, Y_train=None, Y_test=None):


    [G,node_subjects]=stellargraph_bail(dataframe=Dataset,dataframe_edge_list=df_edge_list)
    
    
    test_subjects = Y_test
    
    train_subjects, val_subjects = model_selection.train_test_split(Y_train, test_size=0.5, stratify=Y_train)
    
    
    #train_subjects, val_subjects = model_selection.train_test_split(Y_train, test_size=0.7)
    
    
    target_encoding = preprocessing.LabelBinarizer()
    
    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)
    
    
    
    generator = FullBatchNodeGenerator(G, method="gat")
    

    train_gen = generator.flow(train_subjects.index, train_targets)
    
    """Now we can specify our machine learning model, we need a few more parameters for this:
    
     * the `layer_sizes` is a list of hidden feature sizes of each layer in the model. In this example we use two GAT layers with 8-dimensional hidden 
     node features for the first layer and the 7 class classification output for the second layer.
     * `attn_heads` is the number of attention heads in all but the last GAT layer in the model
     * `activations` is a list of activations applied to each layer's output
     * Arguments such as `bias`, `in_dropout`, `attn_dropout` are internal parameters of the model, execute `?GAT` for details.
    
    To follow the GAT model architecture used for Cora dataset in the original paper [Graph Attention Networks. P. 
    Veličković et al. ICLR 2018 https://arxiv.org/abs/1710.10903], let's build a 2-layer GAT model, with the second layer 
    being the classifier that predicts paper subject: it thus should have the output size of `train_targets.shape[1]` (7 subjects) and a softmax activation.
    """
    
    gat = GAT(
        layer_sizes=[16,16, train_targets.shape[1]],
        activations=["elu", "elu","sigmoid"],
        attn_heads=8,
        generator=generator,
        in_dropout=0.5,
        attn_dropout=0.5,
        normalize=None,
    )
    
    
    x_inp, predictions = gat.in_out_tensors()
    

    
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.binary_crossentropy,
        metrics=["acc"],
    )
    
    """Train the model, keeping track of its loss and accuracy on the training set, and its generalisation performance on the validation set (we need to create another generator over the validation data for this)"""
    
    val_gen = generator.flow(val_subjects.index, val_targets)
    
    """Create callbacks for early stopping (if validation accuracy stops improving) and best model checkpoint saving:"""
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    es_callback = EarlyStopping(
        monitor="val_acc", patience=100
    )  # patience is the number of epochs to wait before early stopping in case of no further improvement
    mc_callback = ModelCheckpoint(
        "logs/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True
    )
    
    """Train the model"""
    
    history = model.fit(
        train_gen,
        epochs=500,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback, mc_callback],
    )
    
    """Plot the training history:"""
    
    sg.utils.plot_history(history)
    
    """Reload the saved weights of the best model found during the training (according to validation accuracy)"""
    
    model.load_weights("logs/best_model.h5")
    
    """Evaluate the best model on the test set"""
    
    test_gen = generator.flow(test_subjects.index, test_targets)
    
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))
    
    """### Making predictions with the model
    
    Now let's get the predictions for all nodes:
    """
    
    test_predictions = model.predict(test_gen)
    
    
    predictions = target_encoding.inverse_transform(test_predictions.squeeze())

        
    print(classification_report(Y_test, predictions))
    
    report=classification_report(Y_test, predictions,output_dict=True)
    
    conf_mat=confusion_matrix(Y_test, predictions)
    
    return predictions,report,conf_mat, model




#%%




def sort_correlation(dataframe, attr='SALARY',sensitive_attr=None):
    
    """
    given dataframe df and an attribute attr, sort the dataframe columns with increasing 
    correaltion(absolute) to the attr. 
    """
    df=dataframe.copy(deep=True)
    
    corr=df.corr().abs()
    ix = df.corr().abs().sort_values(attr, ascending=False).index
    df=df.loc[:, ix]
    
    a=len(df.columns)
    
    first_column = df.pop(sensitive_attr)
  
    # insert column using insert(position,column_name,
    # first_column) function
    #df.insert(a-1, sensitive_attr, first_column)
    df.insert(0, sensitive_attr, first_column)
    return df, corr


def sep_input_output_data(df, attr=None):
    
    """
    seperating training and testing data, given input dataframe and output 
    attribute. Output is y_data and x_data
    """
    y_data = df[attr] #output is salary
    x_data=df.copy(deep=True) 
    x_data.drop(attr, inplace=True, axis=1)
    
    return y_data,x_data



def normalize_dataframe(Dataframe, Scaler):
    
    """
    scale a df if not scaled, Dataframe is the unscaled dataframe and scaler is 
    the class of sklearn, which you want to apply
    
    """

    x = Dataframe.values #returns a numpy 
    name=list(Dataframe.columns)
    #Scaler = preprocessing.MinMaxScaler()
    x_scaled = Scaler.fit_transform(x)
    Dataframe = pd.DataFrame(x_scaled)
    Dataframe.columns = name
    return Dataframe



def nba_data_processing(path):
    """
    Input:
        path: path to nba dataset

    Output:
        original and clean dataset in pandas df type
    """
    
    
    df_nba = pd.read_csv(path)
    
    df_nba_original=df_nba.copy(deep=True)
    
    """
    the dataset has salary as predictive attribute, to make it a classification task we 
    have divided it into either 50m=0, greater than 50m=1, salary=-1  means unlabeled data
    
    but since we want a binary classification task we further process it to make it
    greater than or equal to 50m=1, less than 50m=0
    
    the sentitive atttribute is country which is either USA or not, 
    USA=0
    Non-USA=1
    
    but since we want a binary classification task we further process it to make it
    greater than or equal to 50m=1, less than 50m=0
    we remove the missing data salary=-1
    """
    df_nba.drop(df_nba.index[df_nba['SALARY'] == -1], inplace=True) 
    #remove salary=-1 rows, they are missing 
    
    
    df_nba.drop('user_id', inplace=True, axis=1) # drop user_id as it is of no use

    df_nba.reset_index( inplace=True,drop=True)

    return df_nba_original, df_nba
    

def learn_logistic_regression(x_training_data, x_test_data, y_training_data, y_test_data):


    model = LogisticRegression()

    model.fit(x_training_data, y_training_data)

    predictions = model.predict(x_test_data)
    
    print(classification_report(y_test_data, predictions))

    report=classification_report(y_test_data, predictions,output_dict=True)

    conf_mat=confusion_matrix(y_test_data, predictions)
    
    return predictions,report,conf_mat, model


def fairness_measures(sensitive_attr,x_test,y_test,y_pred):
    """
    measures the pairwise fairness 
    
    Input:
        
        sensitive_attr: name of senstitive attribute column in df
        x_test: test df
        y_test: actual labels df
        y_pred: predicted label df
        
    Ouputs:
        
        class of measures the pairwise fairness 
    
    """
    
    sens_attr = x_test[sensitive_attr]

    metrics_dict = {

        "acc": accuracy_score,
        "prec": precision_score,
        'recall': recall_score,
        "FPR": fairlearn.metrics.false_positive_rate,
        "FNR": fairlearn.metrics.false_negative_rate,
        "TPR": fairlearn.metrics.true_positive_rate,
        "TNR": fairlearn.metrics.true_negative_rate,
        "count": fairlearn.metrics.count
                   }
    mf = MetricFrame(
         metrics=metrics_dict,
         y_true=y_test,
         y_pred=y_pred,
         sensitive_features=sens_attr)

    return mf
    

pred=None
y_test=None

def binary_conf_mat(sen_attr, test_true_labels=y_test, test_pred_labels=pred):
    
    """
    input:
        sen_attr as a dataframe (it can be from train or test data)
        test_true_labels= true labels 
        test_pred_labels= predicted labels
        
    Output:
        conf_mat: binary confusion matrix 
        sp_a: statistical parity of sensitive attribute=a
        sp_b: statistical parity of sensitive attribute=b
        eq_opp: equal oppertunity
        
    
    by convention we take 1 implies sensitive_attribute=a 
    and 0 impies sensitive_attribute=b
    """
    
    TP_a = 0
    FP_a = 0
    TP_b = 0
    FP_b = 0
    TN_a = 0
    FN_a = 0
    TN_b = 0
    FN_b = 0
    
    sen_attr=np.array(sen_attr)
    test_true_labels=np.array(test_true_labels)
    test_pred_labels=np.array(test_pred_labels)

    for index in range(len(test_pred_labels)):
        
       
        # print(i)
        if (sen_attr[index] == 1 and test_true_labels[index] == 1 and test_pred_labels[index] == 1):
          TP_a = TP_a + 1
          # print(TP_a)
      
        if (sen_attr[index]==0 and test_true_labels[index] == 1 and test_pred_labels[index] == 1):
          TP_b = TP_b + 1
      
        if (sen_attr[index]==1 and not(test_true_labels[index]) and test_pred_labels[index] == 1):
          FP_a = FP_a + 1
      
        if (sen_attr[index] == 0 and not(test_true_labels[index]) and test_pred_labels[index] == 1):
          FP_b = FP_b + 1
      
        if (sen_attr[index] == 1 and not(test_true_labels[index]) and not(test_pred_labels[index])):
          TN_a = TN_a + 1
          # print(TN_a)
        
        if (sen_attr[index] == 0 and not(test_true_labels[index]) and not(test_pred_labels[index])):
          TN_b = TN_b + 1
          # print(TN_b)
        
        if (sen_attr[index] == 1 and test_true_labels[index] and not(test_pred_labels[index])):
          # print(FN_a)
          FN_a = FN_a + 1
          
        if (sen_attr[index] == 0 and test_true_labels[index] and not(test_pred_labels[index])):
          FN_b = FN_b + 1
          # print(FN_b)

    
    conf_mat={'TP_1':TP_a,
              'TP_0':TP_b,
              'FP_1':FP_a,
              'FP_0':FP_b,
              'TN_1':TN_a,
              'TN_0':TN_b,
              'FN_1':FN_a,
              'FN_0':FN_b}

    if (TP_a + FP_a + TN_a+ FN_a)==0:
        sp_a=None
    else:    
        sp_a = abs((TP_a + FP_a)/(TP_a + FP_a + TN_a+ FN_a))
    
    if (TP_b + FP_b + TN_b + FN_b)==0:
        sp_b=None
    else:
        sp_b = abs((TP_b + FP_b)/(TP_b + FP_b + TN_b+ FN_b))

    if ((TP_a + FN_a)==0) or ((TP_b + FN_b)==0):
        eq_opp = None
    else:
        eq_opp = abs(((FN_a/(TP_a + FN_a)) - (FN_b/(TP_b + FN_b))))
                     

    return conf_mat,sp_a,sp_b,eq_opp


#%%

theta_MCAR=None #MCAR probability


k=None  #number of sttributes from which data is removed for missingness


attr_type=['sensitive attribute only', 'from k attributes']

#attr_type=['sensitive sttribute only'] #for MCAR all attribute types produces similar results

#sens_attr_name='sens_attr' #sensitive attribute column name in dataframe
sens_attr_name=None #sensitive attribute column name in dataframe

theta_k=None # theta k is less than theta mcar because it produces missign data in many columns

df=None

def MCAR(Dataset=df,Theta=theta_MCAR, Attr_type=attr_type[0],\
         Sens_attr_name=sens_attr_name,K=k, theta_k=theta_k):
    
    """
    Description:
        given datasete dataframe, produce missingness in data attributes as NaN
        k: if 1 only from sensitive attribute, otherwise from first K attributes independently
        theta probability of missingness (0-1), 1 means surely missing 
        sensitive attribute is the name of sensitive attribute in df
    
    input:
        Dataset: input dataset in pandas dataframe form
        Theta: missing probability if missiness from only sensitive attribute
        Attr_type: missingness from either sensitive or fron K attributes 
        Sens_attr_name: Name of sensitive attribute in Dataset daraframe
        K: number of attributes from which missing data is introduced
        theta_k: missing probability if missiness from k attribute
        
    Outout:
    return dataframe with missign values as NaN
    
    """
    
    if Dataset.__class__.__name__ != 'DataFrame':
    
        print('Data type is not Dataframe')
        
        sys.exit() #stop program from execution
    
    print('Data type is Dataframe')
    
    copy=Dataset.copy(deep=True)
    
    if Attr_type=='sensitive attribute only':
        
        for i, row_value in copy[Sens_attr_name].iteritems():
            if random.uniform(0,1) < Theta:
                copy[Sens_attr_name][i] = np.nan
                    

    if Attr_type=='from k attributes':
        
        for i in range(K):
            for j in range(len(copy)):
                if random.uniform(0,1) < theta_k:
                    copy.iloc[[j], [i]] = np.nan
        
        #index=[i for i in range(len(copy)) if random.uniform(0, 1)<Theta]
        
        #copy.drop(index, axis=0, inplace=True)
    
    #arr = np.delete(Adj_mat, index,axis=0)
    #arr = np.delete(arr, index,axis=1)
    
    return copy



"""
MAR missing at random

we are considering 4 types. For every z(z1,z2,z3....zp) sample with p 
attribute alpha and beta are numbers. The probability of missing zj is

0 alpha1+beta1*1(if female =true) from sensitive attribute only
1 alpha1+beta1*1(if male =true)  from sensitive attribute only
2 alpha2+beta2*z(K+j) from K attributes
3 alpha2-beta2*z(K+j) from K attributes

these are encoded as a=[0,1,2,3]

"""

alpha1_MAR=None #MCAR probability
beta1_MAR=None
alpha2_MAR=None
beta2_MAR=None #-0.5

par=[alpha1_MAR,beta1_MAR,alpha2_MAR,beta2_MAR]

 # when k is more than one, missing data in many attributes

k=None  #number of attributes from which data is removed for missingness

a=[0,1,2,3]

attr_type=['sensitive attribute only', 'from k attributes']

#attr_type=['sensitive sttribute only'] #for MCAR all attribute types produces similar results

sensitive_attr=None #sensitive attribute column anem in dataframe


def MAR(Dataset=df, A=a[0], Par=par, Attr_type=attr_type[0],\
         sens_attr_name=sensitive_attr,K=k):
    
    """
    Description:
        given datasete dataframe, produce missingness in data attributes as NaN
        using missing at random (MAR).
        
        we are considering 4 types. For every z(z1,z2,z3....zp) sample with p 
        attribute alpha and beta are numbers. The probability of missing zj is

        0 alpha1+beta1*1(if sensitive attribute=0, else alpha1) from sensitive attribute only
        1 alpha1+beta1*1(if sensitive attribute=1 else alpha1)  from sensitive attribute only
        2 alpha2+beta2*z(K+j) from K attributes
        3 alpha2-beta2*z(K+j) from K attributes

        these are encoded as a=[0,1,2,3]
    
    input:
        Dataset: input dataset in pandas dataframe form
        Par: Parameters of missing probability, first two are used if
            only sensitive attribute missing, last two for k missing attributes  
        Attr_type: missingness from either sensitive or fron K attributes 
        Sens_attr_name: Name of sensitive attribute in Dataset daraframe
        K: number of attributes from which missing data is introduced
        A: missingness types
        
    Outout:
    return dataframe with missign values as NaN

"""
    
    
    if Dataset.__class__.__name__ != 'DataFrame':
    
        print('Data type is not Dataframe')
        
        sys.exit() #stop program from execution
    
    print('Data type is Dataframe')
    
    copy=Dataset.copy(deep=True)
    
    """
    if Attr_type=='sensitive attribute only':
        
        if A==0:
        
            for i, row_value in copy[sens_attr_name].iteritems():
                
                t=0
                
                if copy[sens_attr_name][i]==0: #based on female
                
                    t=Par[0]+Par[1]
                
                if copy[sens_attr_name][i]==1: # based on female but male is present
                    
                    t=Par[0]
                
                if random.uniform(0,1) < t:
                    copy[sens_attr_name][i] = np.nan
    
    
    if Attr_type=='sensitive attribute only':
        
        if A==1: #based on male attribute only
        
        
            for i, row_value in copy[sens_attr_name].iteritems():
                
                t=0
                
                if copy[sens_attr_name][i]==1: #based on male, and male=1
                
                    t=Par[0]+Par[1]
                
                if copy[sens_attr_name][i]==0: # based on male but female is present, male=0
                    
                    t=Par[0]
                
                if random.uniform(0,1) < t:
                    copy[sens_attr_name][i] = np.nan
                    
       """
             
    if Attr_type=='from k attributes':
        
        if A==2:
        
            for i in range(K):
                
                for j in range(len(copy)):
                    
                    t=Par[2]+(Par[3]*copy.iloc[j][i+K])
                    
                    if random.uniform(0,1) < t:
                        copy.iloc[[j], [i]] = np.nan
                

    if Attr_type=='from k attributes':
        
        if A==3:
        
            for i in range(K):
                
                for j in range(len(copy)):
                    
                    t=Par[2]-(Par[3]*copy.iloc[j][i+K])
                    
                    if random.uniform(0,1) < t:
                        copy.iloc[[j], [i]] = np.nan
                
    
    if Attr_type=='from k attributes':
        
        if A==0:
        
            for i in range(K):
                
                for j in range(len(copy)):
                    
                    t=0
                    
                    if copy[sens_attr_name][i]==0: #based on female
                    
                        t=Par[0]+Par[1]
                    
                    if copy[sens_attr_name][i]==1: # based on female but male is present
                        
                        t=Par[0]
                    
                    if random.uniform(0,1) < t:
                        
                        copy.iloc[[j], [i]] = np.nan
                        
                

    if Attr_type=='from k attributes':
        
        if A==1:
        
            for i in range(K):
                
                for j in range(len(copy)):
                    
                    t=0
                    
                    if copy[sens_attr_name][i]==1: #based on male
                    
                        t=Par[0]+Par[1]
                    
                    if copy[sens_attr_name][i]==0: # based on male but female is present
                        
                        t=Par[0]
                    
                    if random.uniform(0,1) < t:
                        
                        copy.iloc[[j], [i]] = np.nan
        
        #index=[i for i in range(len(copy)) if random.uniform(0, 1)<Theta]
        
        #copy.drop(index, axis=0, inplace=True)
    
    #arr = np.delete(Adj_mat, index,axis=0)
    #arr = np.delete(arr, index,axis=1)
    
    return copy



"""
MNAR missing not at random

we are considering 2 types. For every z(z1,z2,z3....zp) sample with p 
attribute (alpha and beta are numbers). The probability of missing zj is

0 alpha1+beta1*zj from sensitive attribute only

1 alpha2+beta2*zj from K attributes


these are encoded as a=[0,1]

"""

alpha1_MNAR=None #MCAR probability
beta1_MNAR=None
alpha2_MNAR=None
beta2_MNAR=None #-0.5

par_MNAR=[alpha1_MNAR,beta1_MNAR,alpha2_MNAR,beta2_MNAR]

 # when k is more than one, missing data in many attributes

k=None  #number of attributes from which data is removed for missingness

a=[0,1]

attr_type=['sensitive attribute only', 'from k attributes']

#attr_type=['sensitive sttribute only'] #for MCAR all attribute types produces similar results

sens_attr_name=None #sensitive attribute column anem in dataframe


def MNAR(Dataset=df, A=a[0], Par=par_MNAR, Attr_type=attr_type[0],\
         Sens_attr_name=sens_attr_name,K=k):
    
    """
    Description:
        given datasete dataframe, produce missingness in data attributes as NaN
        using missing at random (MNAR).
        
        we are considering 2 types. For every z(z1,z2,z3....zp) sample with p 
        attribute alpha and beta are numbers. The probability of missing zj is
  
        0 alpha1+beta*zj from sensitive attribute only
  
        1 alpha2+beta2*zj from K attributes
  
  
        these are encoded as a=[0,1]
     
    
    input:
        Dataset: input dataset in pandas dataframe form
        Par: Parameters of missing probability, first two are used if
            only sensitive attribute missing, last two for k missing attributes  
        Attr_type: missingness from either sensitive or fron K attributes 
        Sens_attr_name: Name of sensitive attribute in Dataset daraframe
        K: number of attributes from which missing data is introduced
        A: missingness types
        
    Outout:
    return dataframe with missign values as NaN
"""
    
    if Dataset.__class__.__name__ != 'DataFrame':
    
        print('Data type is not Dataframe')
        
        sys.exit() #stop program from execution
    
    print('Data type is Dataframe')
    
    copy=Dataset.copy(deep=True)
    
    if Attr_type=='sensitive attribute only':
        
        if A==0:
        
            for i, row_value in copy[Sens_attr_name].iteritems():
                
                t=Par[0]+(Par[1]*copy[Sens_attr_name][i])
                
                
                if random.uniform(0,1) < t:
                    copy[Sens_attr_name][i] = np.nan
    
                    
    if Attr_type=='from k attributes':
        
        if A==1:
        
            for i in range(K):
                
                for j in range(len(copy)):
                    
                    t=Par[2]+(Par[3]*copy.iloc[j][i])
                    
                    if random.uniform(0,1) < t:
                        copy.iloc[[j], [i]] = np.nan
                
    
    return copy



def Average(lst):
    if len(lst)!=0:
        return sum(lst) / len(lst)
    else:
        return None

def imp_IAPD(dataset_original=None,dataset_missing=None,dataset_imp=None,sens_attr=None,I=None,scaler=None):
    
    """
    compute imputation accuracy parity differnece (IAPD), which shows fairness of imputation.
    
    Input:
        dataset_Original: orignal dataset in df
        dataset_missing: dataset with missing values as nan in df
        imp_Original: imputated dataset with nan values repalced in df
        sens_attr= name of sensitve attribute column in df
    Output:
        IAPD
        
    """
    copy_original=dataset_original.copy(deep=True)
    copy_missing=dataset_missing.copy(deep=True)
    copy_imp=dataset_imp.copy(deep=True)
    
    
    copy_original.reset_index( inplace=True,drop=True)
    copy_missing.reset_index( inplace=True,drop=True)
    copy_imp.reset_index( inplace=True,drop=True)
    
    #copy_original=scaler.inverse_transform(copy_original)
    #copy_missing=scaler.inverse_transform(copy_missing)
    #copy_imp=scaler.inverse_transform(copy_imp)
    
    if I=='del':
        return None
    
    else:
    
        l_0=[]
        l_1=[]
        
        for i in range(len(copy_original)):
            for col in copy_original:
                #print(i,col)
                if np.isnan(copy_missing.loc[i,col]):
                    
                    if copy_original.loc[i,sens_attr]==1:
                        l_1.append((copy_original.loc[i,col]-copy_imp.loc[i,col])*(copy_original.loc[i,col]-copy_imp.loc[i,col]))            
                    else:
                        l_0.append((copy_original.loc[i,col]-copy_imp.loc[i,col])*(copy_original.loc[i,col]-copy_imp.loc[i,col]))
        
        if len(l_0)==0:
            l_0=0
        else:    
            l_0=Average(l_0)
        
        if len(l_1)==0:
            l_1=0
        else:    
            l_1=Average(l_1)
        
        if (l_0==None) or (l_1==None):
            
            return None
        else:
            
            return (l_0-l_1)
    

#%%
