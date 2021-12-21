from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from hyperopt import fmin, tpe, anneal, hp, Trials, space_eval
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from functools import partial

def surgery_preprocess(split=True): # If split=True, return the split; else, return all data
    
    
    ### Load/subset/merge NHANES data ###
    
    # Read in the datasets
    spirometry = pd.read_sas('SPX_G.XPT')
    demographics = pd.read_sas('DEMO_G.XPT')
    body = pd.read_sas('BMX_G.XPT')
    enx = pd.read_sas('ENX_G.XPT')
    bpx = pd.read_sas('BPX_G.XPT')

    
    # Subset the datasets
    subset = spirometry[['SPXNFVC', 'SPXNFEV1', 'SEQN', 
                             'ENQ010', 'ENQ020', 'SPQ040', 'ENQ100', 
                             'SPQ050', 'SPQ060','SPQ070B', 'SPQ100']]
    demographics = demographics[['RIDAGEYR', 'RIAGENDR', 'SEQN']]
    body = body[['BMXHT', 'BMXWT', 'SEQN']]
    enx = enx[['ENQ040','ENQ090','ENXTR4Q','SEQN']]
    bpx = bpx[['BPQ150D','SEQN']]
    
    # Merge the datasets by SEQN
    subset = pd.merge(subset, demographics, 'left', 'SEQN')
    subset = pd.merge(subset, body, 'left', 'SEQN')
    subset = pd.merge(subset, enx, 'left', 'SEQN')
    subset = pd.merge(subset, bpx, 'left', 'SEQN')
    subset.rename(columns = {'SPXNFVC': 'FVC', 'SPXNFEV1': 'FEV1',
                            'RIDAGEYR': 'Age', 'RIAGENDR': 'Sex', 
                            'BMXHT': 'Height', 'BMXWT': 'Weight'}, inplace=True)
    
    # entries prior to filtering: 7495
    subset = subset[(subset['ENQ010'] !=1) & (subset['ENQ020']!=1) &
                    (subset['SPQ040']!=1) & (subset['SPQ050']!=1) & (subset['SPQ060']!=1) &
                    (subset['SPQ070B']!=2) & ( subset['SPQ100']!=1) & (subset['ENQ100']!=1) &
                    (subset['ENQ040']!=1) & (subset['ENQ090']!=1)]
    
    subset.dropna(axis=0, subset=['FVC','Age','FVC'])
    # 4950 entries
    
    # Get mean FVC and FEV1 per age
    estimates = subset.groupby('Age')[['FVC', 'FEV1']].mean()
    estimates.rename(columns={'FVC':'mean_FVC', 'FEV1':'mean_FEV1'}, inplace=True) 
    
    ### Load and clean surgery data, and merge variables from NHANES data ###
    
    # Read in surgery data
    surgery = pd.read_csv('ThoracicSurgery.csv', index_col = 0)
    
    # Rename variables
    surgery.rename(columns = {'DNG': 'Diagnosis', 'PRE4': 'FVC',
                            'PRE5': 'FEV1', 'PRE6': 'Performance',
                            'PRE7': 'Pain', 'PRE8': 'Haemoptysis',
                            'PRE9': 'Dyspnoea', 'PRE10': 'Cough',
                            'PRE11': 'Weakness', 'PRE14': 'Tumor_size',
                            'PRE17': 'Type2_diabetes', 'PRE19': 'MI_6months',
                            'PRE25': 'PAD', 'PRE30': 'Smoking',
                            'PRE32': 'Asthma', 'AGE': 'Age'}, inplace=True)
    
    # Back to NHANES data for a bit...
    # Impute mean FVC and FEV1 for ages 80+ using a simple linear regression
    # We only have 8 people of ages 80+ in our data. (Ages: 80,81,87)

    # To get a good linear fit, zoom into the age range where both mean FVC and FEV1 start to steadily decline
    lr_estimates = estimates[estimates.index > 25]
    
    # Fit linear regression to FVC and FEV1
    lr_FVC = LinearRegression(fit_intercept=True,normalize=True).fit(lr_estimates.index.values.reshape(-1, 1),lr_estimates['mean_FVC'])
    lr_FEV1 = LinearRegression(fit_intercept=True,normalize=True).fit(lr_estimates.index.values.reshape(-1, 1),lr_estimates['mean_FEV1'])
    
    # Impute the mean FVC and FEV1 for ages 80 - maximum age in the surgery dataset and append
    pred_ages = np.arange(80,surgery.Age.max()+1).reshape(-1,1)
    lr_estimates = pd.DataFrame([lr_FVC.predict(pred_ages).flatten(),lr_FEV1.predict(pred_ages).flatten()]).T
    lr_estimates.columns = estimates.columns
    lr_estimates.index = pred_ages.flatten()
    estimates = estimates.append(lr_estimates)
    
        # Fix errorneous FEV1 values (greater than corresponding FVC)
    errs = surgery.loc[(surgery.FEV1 > surgery.FVC),:].index
    for i in errs:
        patient = surgery.iloc[i,:]
        near_age = surgery.loc[(surgery.Age >= patient.Age-2) & (surgery.Age <= patient.Age+2),:]
        near_age = near_age.loc[~near_age.index.isin(errs)]
        surgery.loc[i,'FEV1'] = near_age.FEV1.mean()
        
    # Compute new features from surgery/NHANES data
    surgery['FEV1/FVC'] = surgery['FEV1'] / surgery['FVC']
    surgery['expected_FVC'] = estimates.loc[surgery.Age, 'mean_FVC'].values / 1000
    surgery['expected_FEV1'] = estimates.loc[surgery.Age, 'mean_FEV1'].values / 1000
    surgery['expected_FEV1/FVC'] = surgery['expected_FEV1'] / surgery['expected_FVC']
    surgery['FEV1_deficit'] = (surgery['expected_FEV1']-surgery['FEV1'])/surgery['expected_FEV1']
    surgery['FVC_deficit'] = (surgery['expected_FVC']-surgery['FVC'])/surgery['expected_FVC']
    surgery['FEV1/FVC_deficit'] = (surgery['expected_FEV1/FVC']-surgery['FEV1/FVC'])/surgery['expected_FEV1/FVC']
    surgery['FEV1^2'] = surgery['FEV1']**2
    surgery['FVC^2'] = surgery['FVC']**2
    surgery['Age*FVC'] = surgery['Age']*surgery['FVC']
    surgery['Age*FEV1'] = surgery['Age']*surgery['FEV1']
    surgery['FVC*FEV1'] = surgery['FVC']*surgery['FEV1']
    surgery['FVC^2*FEV1'] = surgery['FEV1']*(surgery['FVC']**2)
    surgery['FVC*FEV1^2'] = surgery['FVC']*(surgery['FEV1']**2)
    
    # Won't be needing these for modeling... already got what we wanted to compute from these
    surgery.drop(columns=['expected_FVC', 'expected_FEV1', 'expected_FEV1/FVC'], inplace=True)

    # Select this right after engineering new features, but
    # before changing F/T to 0/1, so we don't scale ordinal/binary features
    numeric_cols = surgery.select_dtypes('number').columns
    
    # Clean up string/bool data
    surgery.loc[:,surgery.dtypes=='object']= surgery.loc[:,surgery.dtypes=='object'].apply(lambda s: (s.str.replace("\'", "").str.replace('b', '')))
    surgery.replace(to_replace = ['F', 'T'], value=[0,1], inplace=True)

    # Ordinal encoding
    ord_cols = ['Performance', 'Tumor_size']
    for col in ord_cols:
        surgery[col] = surgery[col].str.strip().str[-1].astype(int)
        
    # Remove diagnoses with too little data (n=1,4,2)
    surgery = surgery[~(surgery.DGN.isin(['DGN1', 'DGN6', 'DGN8']))]
    # Drop categorical feature with too few 1's for the algorithm to learn anything reliable
    surgery.drop(columns=['PAD', 'Asthma', 'MI_6months'], inplace=True)
    # Turn our categorical DGN feature to boolean _DGNx features
    surgery = pd.get_dummies(surgery, prefix=[''], columns=['DGN'])
    
    ### Return the preprocessed data ###
    if split == False:
        return surgery
    else:
        ### Split the data ###
        X_train, X_test, y_train, y_test = train_test_split(surgery.drop('Risk1Yr', axis=1),
                                                            surgery['Risk1Yr'], test_size=0.2,
                                                            shuffle=True,stratify=surgery['Risk1Yr'],
                                                            random_state=0)
        ### Normalize numeric features ###
        scaler = StandardScaler()
        # Fit the test data in the same scale as the training data
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        ### Return the split train/test data ###
        return X_train, X_test, y_train, y_test
    
def split(surgery, drop_cols=[], idx=[]):
    
    numeric_cols=['FVC', 'FEV1', 'Age', 'FEV1/FVC', 'FVC_deficit', 'FEV1_deficit',
       'FEV1/FVC_deficit', 'FEV1^2', 'FVC^2', 'Age*FVC', 'Age*FEV1',
       'FVC*FEV1', 'FVC^2*FEV1', 'FVC*FEV1^2']
    d = drop_cols.copy()
    num_col = numeric_cols.copy()
    if len(d) > 0:
        for el in d:
            if el not in num_col:
                continue
            else:
                num_col.remove(el)
    d.append('Risk1Yr')
    d.append('Risk1Yr')
    
    if len(idx)==0:
        X_train, X_test, y_train, y_test = train_test_split(surgery.drop(columns=d),
                                                        surgery['Risk1Yr'], 
                                                        test_size=0.2,
                                                        shuffle=True,
                                                        stratify=surgery['Risk1Yr'],)
    else:
        X_train = surgery.loc[idx].drop(columns=d)
        X_test = surgery.loc[~surgery.index.isin(idx)].drop(columns=d)
        y_train = surgery.loc[idx, 'Risk1Yr']
        y_test = surgery.loc[~surgery.index.isin(idx),'Risk1Yr']
        
    ### Normalize numeric features ###
    scaler = StandardScaler()
    # Fit the test data in the same scale as the training data
    X_train[num_col] = scaler.fit_transform(X_train[num_col])
    X_test[num_col] = scaler.transform(X_test[num_col])

    ### Return the split train/test data ###
    return X_train, X_test, y_train, y_test

def get_scores(pipe, train_X, test_X, train_y, test_y, scores):
    model_scores = {}
                
    pipe.fit(train_X, train_y)
    
    preds_train = pipe.predict(train_X)
    proba_train = pipe.predict_proba(train_X)[:,1]
    preds_test = pipe.predict(test_X)
    proba_test = pipe.predict_proba(test_X)[:,1]
    for s in scores:
        if s in ['roc_auc_score', 'average_precision_score']:
            model_scores['train_'+s] = eval('metrics.'+s+'(train_y, proba_train)')
            model_scores['test_'+s] = eval('metrics.'+s+'(test_y, proba_test)')
        else:
            model_scores['train_'+s] = eval('metrics.'+s+'(train_y, preds_train)')
            model_scores['test_'+s] = eval('metrics.'+s+'(test_y, preds_test)')
                    
    return model_scores


def plot_scores(model_scores, scores=None):

    if scores == None:
        scores = list(model_scores.keys())
    if len(scores) > 5:
        nrow = len(scores)//2 - 1
        ncol = len(scores)//2 + len(scores)%2
    else:
        nrow = len(scores)//2 + len(scores)%2
        ncol = 2
    fig, ax = plt.subplots(nrow,ncol,figsize=(12,12))
    matplotlib.rc('xtick', labelsize=13) 
    matplotlib.rc('ytick', labelsize=13) 
    num = model_scores[scores[0]].shape[0]
    i = 0
    for s in scores:
        ax[i//ncol,i%ncol].bar(range(1,num+1), model_scores[s])
        ax[i//ncol,i%ncol].hlines(model_scores[s].mean(),1,num)
        ax[i//ncol,i%ncol].set_xlabel('Trial', size=14)
        ax[i//ncol,i%ncol].set_ylabel(s, size=14)
        #ax[i//2,i%2].set_title(names[i], size=14)
        i += 1

def build_model_t(model, params, obj, scores, data, k=8, n=10, evals=70, scoring='roc_auc',
                drop=[], return_idx=False, **model_args):
    scores_dict={}
    param_list=[]
    if return_idx:
        idx_list = []
    for s in scores:
        scores_dict[s] = np.zeros(k)
        
    for i in range(k):
        X_train, X_test, y_train, y_test = split(data, drop_cols=drop)
        mod = model(**model_args)
        pipe = Pipeline([('upsample',ros),('model',mod)])
        trials = Trials()
        # Partial make its own callable function. When you call partial() you supply
        # a function with n args and supply k of them. Then, when you call the function
        # that partial creates, you supply the remaining n-k. These n-k serve as the
        # FIRST arguments to the function of which you made a partial
        fmin_func = partial(obj, pipe=pipe, train_X=X_train, train_y=y_train,
                            scoring=scoring)
        best = fmin(fn=fmin_func, 
                    space=params, algo=anneal.suggest, max_evals=evals,
                    trials=trials)
        best_params = space_eval(params,best)
        
        pipe.fit(X_train,y_train)
        pipe.set_params(**best_params)
        model_scores = get_scores(pipe, X_train, X_test, y_train, y_test,
                                  scores, n=n)
        for s in scores:
            scores_dict[s][i] = model_scores[s]
        param_list.append(best_params)
        if return_idx:
            idx_list.append(X_train.index)
        
    if return_idx:
        return scores_dict, param_list, idx_list
    return scores_dict, param_list