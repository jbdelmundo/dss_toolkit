import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import LabelBinarizer


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss

from scipy.stats import ks_2samp

class CustomClassifier(BaseEstimator):

    def __init__(
        self, 
        estimator = LogisticRegression(),
    ):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """ 

        self.estimator = estimator
        self.cv_results = None


    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self


    def predict(self, X, y=None):
        return self.estimator.predict(X)


    def predict_proba(self, X):
        return self.estimator.predict_proba(X)


    def score(self, X, y):
        return self.estimator.score(X, y)
    
    def set_cv_results(self, cv_results_):
        self.cv_results = cv_results_
        
    def get_cv_results(self):
        return self.cv_results

class ModelClassifier:

    def __init__(self, 
                 train, 
                 validation, 
                 label, 
                 label_values, 
                 features, 
                 log_transform_features = [],
                 log_transform = False,
                 label_binarizer = False, 
                 train_dev_set = False, 
                 train_dev = None, 
                 test_set = False, 
                 test = None):
        
        self.train                  = train
        self.validation             = validation
        
        self.test                   = test
        self.train_dev              = train_dev
        
        self.train_dev_set          = train_dev_set
        self.test_set               = test_set
        
        self.label_values           = label_values

        self.train_features         = self.data_prepare(self.train, features)
        self.validation_features    = self.data_prepare(self.validation, features)

        self.train_labels           = self.data_prepare(self.train, label)
        self.validation_labels      = self.data_prepare(self.validation, label)

        self.scaler                 = None
        self.decomposer             = None
        self.grid_search            = None

        self.filter_low_variance    = False
        self.var_threshold          = 0.05
        
        self.train_dev_features     = None
        self.train_dev_labels       = None

        self.test_features          = None
        self.test_labels            = None

        
        if label_binarizer:

            self.train_labels           = self.label_binarizer(self.train_labels[[label]])
            self.validation_labels      = self.label_binarizer(self.validation_labels[[label]])
            
        if log_transform:
            self.train_features         = self.data_prepare(self.train_features, log_transform_features)
            self.validation_features    = self.data_prepare(self.validation_features, log_transform_features)
        
        if self.train_dev_set:
            self.train_dev_features     = self.data_prepare(self.train_dev, features)
            self.train_dev_labels       = self.data_prepare(self.train_dev, label)
            
            if label_binarizer:
                self.train_dev_labels   = self.label_binarizer(self.train_dev_labels[[label]])
                
            if log_transform:
                self.train_dev_features = self.data_prepare(self.train_dev_features, log_transform_features)

        if self.test_set:
            self.test_features          = self.data_prepare(self.test, features)
            self.test_labels            = self.data_prepare(self.test, label)
            
            if label_binarizer:
                self.test_labels        = self.label_binarizer(self.test_labels[[label]])

            if log_transform:
                self.test_features      = self.data_prepare(self.test_features, log_transform_features)
            

    def data_prepare(self, data, fields):
        return data[fields]

    def log_transform(self, data, fields):
        data[fields] = np.log(data[fields] + 1)
        return data
        
    def label_binarizer(self, data):
        lb = LabelBinarizer()
        return lb.fit_transform(data)

    def get_filtered_low_variance_features(self, data):
        selector = VarianceThreshold(threshold = (self.var_threshold * (1 - self.var_threshold)))
        data = data[data.columns[selector.get_support(indices=True)]] 
        return data

    def set_decomposer(self, pca_params):
        self.decomposer = PCA(**pca_params)

    def get_decomposer(self):
        return self.decomposer
    
    def set_scaler(self, scaler_type = 'standard', scaler = StandardScaler()):

        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'quantile':
            self.scaler = QuantileTransformer()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'custom':
            self.scaler = scaler
        else:
            self.scaler = StandardScaler()
        
    def get_scaler(self):
        return self.scaler

    def set_grid_search(self, grid_search_type = 'gridsearchcv', grid_search = GridSearchCV):

        if grid_search_type == 'gridsearchcv':
            self.grid_search = GridSearchCV
        elif grid_search_type == 'randomsearchcv':
            self.grid_search = RandomizedSearchCV
        elif grid_search_type == 'custom':
            self.grid_search = grid_search
        else:
            self.grid_search = GridSearchCV

    def get_grid_search(self, pipeline_params, grid_search_params, scaler = False, pca = False):
        pipeline = self.pipeline_generator(scaler = scaler, pca = pca)

        return self.grid_search(pipeline, pipeline_params, **grid_search_params)

    def pipeline_generator(self, scaler = False, pca = False):
        
        pipes = []

        if scaler is not None:
            pipes.append(('scaler', self.get_scaler()))
        
        if pca:
            pipes.append(('pca', self.get_decomposer()))
            
        pipes.append(('clf', CustomClassifier()))
                    
        return Pipeline(pipes)


    def classifier(self, pipeline_params, grid_search_params, scaler = False, pca = False):
                
        gscv = self.get_grid_search(pipeline_params, grid_search_params, scaler, pca)
        gscv.fit(self.train_features, self.train_labels)

        print("Best parameter (CV score = %0.3f):" % gscv.best_score_)
        print(gscv.best_params_)

        model = gscv.best_estimator_
        model[-1].set_cv_results(gscv.cv_results_)
           
        return model


    def classifier_metrics(self, model, confusion_matrix = False, return_pred = False):
        y_train_pred, y_val_pred, y_train_dev_pred, y_test_pred = None, None, None, None
        
        y_train_pred = model.predict(self.train_features)
        y_train_pred_proba = model.predict_proba(self.train_features)[::,1]

        y_val_pred   = model.predict(self.validation_features)
        y_val_pred_proba = model.predict_proba(self.validation_features)[::,1]
        
        print("\nTraining Performance:")
        self.print_evaluation_scores(self.train_labels, y_train_pred, y_train_pred_proba)
        

        print("\nValidation Performance:")
        self.print_evaluation_scores(self.validation_labels, y_val_pred, y_val_pred_proba)

        if confusion_matrix: 
            self.get_confusion_matrix(self.validation_labels, y_val_pred)
        
        
        if self.train_dev_set:

            print("\nTrain-Dev Performance:")
            y_train_dev_pred   = model.predict(self.train_dev_features)
            y_train_dev_pred_proba = model.predict_proba(self.train_dev_features)[::,1]
            
            self.print_evaluation_scores(self.train_dev_labels, y_train_dev_pred, y_train_dev_pred_proba)

            if confusion_matrix: 
                self.get_confusion_matrix(self.train_dev_labels, y_train_dev_pred)

        if self.test_set:

            print("\nTest Performance:")
            y_test_pred   = model.predict(self.test_features)
            y_test_pred_proba = model.predict_proba(self.test_features)[::,1]
            
            self.print_evaluation_scores(self.test_labels, y_test_pred, y_test_pred_proba)

            if confusion_matrix: 
                self.get_confusion_matrix(self.test_labels, y_test_pred)
         
        if return_pred:
            return (y_train_pred, y_val_pred, y_train_dev_pred, y_test_pred)
        

    def evaluation(self, metric, y, y_pred, avg):
        return metric(y, y_pred, average = avg)

    def print_evaluation_scores(self, y, y_pred, y_pred_proba):
        
        print(f'AUC score: {roc_auc_score(y, y_pred_proba)}')      
        print(f'LogLoss score: {log_loss(y, y_pred_proba)}')      
        print(f'Accuracy: {accuracy_score(y,y_pred)}')
        print('f1 macro: {}'.format(self.evaluation(f1_score,y,y_pred,'macro')))
        print('f1 micro: {}'.format(self.evaluation(f1_score,y,y_pred,'micro')))
        print('f1 weighted: {}'.format(self.evaluation(f1_score,y,y_pred,'weighted')))
        print('Precision macro: {}'.format(self.evaluation(average_precision_score,y,y_pred,'macro')))
        print('Precision micro: {}'.format(self.evaluation(average_precision_score,y,y_pred,'micro')))
        print('Precision weighted: {}'.format(self.evaluation(average_precision_score,y,y_pred,'weighted')))
        print('Recall macro: {}'.format(self.evaluation(recall_score,y,y_pred,'macro')))
        print('Recall micro: {}'.format(self.evaluation(recall_score,y,y_pred,'micro')))
        print('Recall weighted: {}'.format(self.evaluation(recall_score,y,y_pred,'weighted')))
        
    def training_evaluation_scores(self, model):
        y_train_pred, y_val_pred, y_train_dev_pred, y_test_pred = None, None, None, None
        
        y_train_pred = model.predict(self.train_features)
        y_train_pred_proba = model.predict_proba(self.train_features)[::,1]
        
        training_eval_scores = self.evaluation_scores(self.train_labels, y_train_pred, y_train_pred_proba)
        
        return training_eval_scores
    
    
    def validation_evaluation_scores(self, model):
        y_train_pred, y_val_pred, y_train_dev_pred, y_test_pred = None, None, None, None
        
        y_val_pred   = model.predict(self.validation_features)
        y_val_pred_proba = model.predict_proba(self.validation_features)[::,1]
        
        validation_eval_scores = self.evaluation_scores(self.validation_labels, y_val_pred, y_val_pred_proba)
        
        return validation_eval_scores

    def test_evaluation_scores(self, model):
                
        if self.test_features is None:
            
            self.validation_evaluation_scores(model)
            
        else:
        
            y_train_pred, y_val_pred, y_train_dev_pred, y_test_pred = None, None, None, None

            y_test_pred   = model.predict(self.test_features)
            y_test_pred_proba = model.predict_proba(self.test_features)[::,1]

            test_eval_scores = self.evaluation_scores(self.test_labels, y_test_pred, y_test_pred_proba)

            return test_eval_scores

    def all_evaluation_scores(self, model):
        
        return (self.training_evaluation_scores(model), self.validation_evaluation_scores(model))
    
    def ks_stat(self, y, yhat):
        return ks_2samp(yhat[y==1], yhat[y!=1]).statistic
        
    def evaluation_scores(self, y, y_pred, y_pred_proba):
        cm = self.get_confusion_matrix_instance(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        
        return {
            "auc_score":         roc_auc_score(y, y_pred_proba),
            "logloss":           log_loss(y, y_pred_proba),
            "gini":              2 * auc - 1,
            "accuracy":          accuracy_score(y,y_pred),
            "f1_macro":          self.evaluation(f1_score,y,y_pred,'macro'),
            "f1_micro":          self.evaluation(f1_score,y,y_pred,'micro'),
            "f1_weighted":       self.evaluation(f1_score,y,y_pred,'weighted'),
            "precision_macro":   self.evaluation(average_precision_score,y,y_pred,'macro'),
            "precision_micro":   self.evaluation(average_precision_score,y,y_pred,'micro'),
            "precision_weighted":self.evaluation(average_precision_score,y,y_pred,'weighted'),
            "recall_macro":      self.evaluation(recall_score,y,y_pred,'macro'),
            "recall_micro":      self.evaluation(recall_score,y,y_pred,'micro'),
            "recall_weighted":   self.evaluation(recall_score,y,y_pred,'weighted'),
            "true_positive":     cm[1][1],
            "false_positive":    cm[0][1],            
            "true_negative":     cm[0][0],
            "false_negative":    cm[1][0],
            "ks_test":           self.ks_stat(y,y_pred),
            "balanced_accuracy": balanced_accuracy_score(y,y_pred),
            "specificity":       cm[0][0]/(cm[0][0] + cm[0][1])
        }
        
    def get_roc_curve(self, y, y_pred, n_classes):
        roc_curve(y, y_pred, n_classes)
        
    def get_confusion_matrix_instance(self, y, y_pred):
        return confusion_matrix(y, y_pred)

    def get_train_get_decile_analysis(self, model):

        predictions = model.predict_proba(self.train_features)[::,1]

        return self.get_decile_analysis(self.train_labels, predictions)

    def get_validation_get_decile_analysis(self, model):

        predictions = model.predict_proba(self.validation_features)[::,1]

        return self.get_decile_analysis(self.validation_labels, predictions)

    def get_train_get_decile_analysis(self, model):

        predictions = model.predict_proba(self.train_features)[::,1]

        return self.get_decile_analysis(self.train_labels, predictions)

    def get_decile_analysis(self, labe
    
        results = pd.DataFrame({'labels': labels, 'predictions': predictions})

        try:
            results['bin'] = pd.qcut(results.predictions, 10, duplicates='drop')
            grouped = results.groupby('bin', as_index = False)

            ks_table = pd.DataFrame()
            ks_table['min_prob'] = grouped.min()['predictions'].apply('{0:.6%}'.format)
            ks_table['max_prob'] = grouped.max()['predictions'].apply('{0:.6%}'.format)
            ks_table['mid_prob'] = grouped.median()['predictions'].apply('{0:.6%}'.format)
            ks_table['events']   = grouped.sum()['labels']
            ks_table['nonevents'] = grouped.sum()['labels']
            ks_table['event_rate'] = (ks_table.events / results['labels'].sum()).apply('{0:.4%}'.format)
            ks_table['nonevent_rate'] = (ks_table.nonevents / results['labels'].sum()).apply('{0:.4%}'.format)
            ks_table['cum_eventrate']=(ks_table.events / results['labels'].sum()).cumsum()
            ks_table['cum_noneventrate']=(ks_table.nonevents / results['labels'].sum()).cumsum()
            ks_table['KS'] = (np.round(ks_table['cum_eventrate']-ks_table['cum_noneventrate'], 3) * 100).abs()
            ks_table['tp'] = grouped.sum()['TP']
            ks_table['fp'] = grouped.sum()['FP']
            ks_table['fn'] = grouped.sum()['FN']
            ks_table['tn'] = grouped.sum()['TN']
            ks_table['leads'] = ks_table['tp'] + ks_table['fp']
            ks_table['actual'] = ks_table['tp']
            ks_table = ks_table.sort_values(by="max_prob", ascending=False).reset_index(drop = True)

            #Formating
            ks_table['cum_eventrate'] = ks_table['cum_eventrate'].apply('{0:.6%}'.format)
            ks_table['cum_noneventrate']= ks_table['cum_noneventrate'].apply('{0:.6%}'.format)
            ks_table.index = range(1,ks_table.index.max()+2)
            ks_table.index.rename('decile', inplace=True)
            pd.set_option('display.max_columns', 9)

            #Display KS
            print("KS is " + str(max(ks_table['KS']))+"%"+ " at decile " + str((ks_table.index[ks_table['KS']==max(ks_table['KS'])][0])))
            print(ks_table)

        except Exception as error:
            print(error, file = sys.stderr)
            
    def get_confusion_matrix(self, y, y_pred):
        cm = confusion_matrix(y, y_pred)
        ax = plt.gca()
        sns.heatmap(cm, annot=True, ax = ax, fmt='g', cmap='Greens') 
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels') 
        ax.set_title('Confusion Matrix') 
        ax.xaxis.set_ticklabels(self.label_values) 
        ax.yaxis.set_ticklabels(self.label_values, rotation=360)
        
            
    

