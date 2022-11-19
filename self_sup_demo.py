import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from tqdm import tqdm
import matplotlib.pyplot as plt

#https://arxiv.org/abs/2202.12040

pl_epochs = 20

thresh = [0.4, 0.6, 0.8, 1]
percs_null = [0.2, 0.5, 0.7]

df = datasets.load_wine()

model_rf = RandomForestClassifier(n_jobs=-1)

parameters = {
    'n_estimators': [10, 50],
    'class_weight': [None, 'balanced'],
    'max_depth': [None, 5, 10]
}

res = pd.DataFrame()

for thr in tqdm(thresh):
    for perc_null in percs_null:
        
        X_tr, X_ts, y_tr, y_ts = train_test_split(
                df.data, df.target, test_size = 0.3, shuffle = True)
        
        rng = np.random.RandomState()
        rand_ul_pnts = rng.rand(y_tr.shape[0]) < perc_null
        
        y_tr[rand_ul_pnts] = -1
        new_y_tr = y_tr.copy()
        
        for i in range(pl_epochs):
            #Select the labeled set
            X = X_tr[np.where(new_y_tr != -1)]
            y = new_y_tr[np.where(new_y_tr != -1)]
            
            #Select the unlabeled set
            X_ul = X_tr[np.where(new_y_tr == -1)]
            y_ul = new_y_tr[np.where(new_y_tr == -1)]
            
            if len(y_ul) == 0:
                break
            
            #Hyperparameter optimization
            model_rf_ = GridSearchCV(model_rf, parameters, cv=2).fit(X,y).best_estimator_
            
            #Probability Calibration
            calib_clf = CalibratedClassifierCV(base_estimator=model_rf_,
                                                    cv=2)
                                                    #ensemble=False)
            calib_clf.fit(X,y)
            preds = calib_clf.predict_proba(X_ul)
            
            #Adding high confidence labels
            labels = np.argmax(preds, axis=1)
            pr_labels = np.max(preds, axis=1)
            
            hconf_labels = labels[np.where(pr_labels >= thr)]
            y_ul[np.where(pr_labels >= thr)] = hconf_labels
            
            new_y_tr[np.where(new_y_tr == -1)] = y_ul
            
        #Validation
        X = X_tr[np.where(new_y_tr != -1)]
        y = new_y_tr[np.where(new_y_tr != -1)]
        calib_clf.fit(X,y)
        
        selflr_y_pred = calib_clf.predict(X_ts)
        
        X = X_tr[np.where(y_tr != -1)]
        y = y_tr[np.where(y_tr != -1)]
        
        calib_clf.fit(X,y)
        y_pred = calib_clf.predict(X_ts)
        
        res = pd.concat([res, pd.DataFrame([{
                'threshold':thresh, 'null_perc':perc_null,
                'normal_acc':accuracy_score(y_ts,y_pred),
                'pseudolabel_acc':accuracy_score(y_ts, selflr_y_pred)
                }])
                         ])

thres_results = res.groupby('threshold', as_index=False).agg({'normal_acc': ['mean', 'std'], 'pseudolabel_acc': ['mean', 'std']})
null_results = res.groupby('null_perc', as_index=False).agg({'normal_acc': ['mean', 'std'], 'pseudolabel_acc': ['mean', 'std']})


plt.figure(figsize=(12, 7))

plt.errorbar(null_results['null_perc'], y=null_results[['null_perc', 'normal_acc']]['normal_acc', 'mean'], yerr=null_results[['null_perc', 'normal_acc']]['normal_acc', 'std'])
plt.errorbar(null_results['null_perc'], y=null_results[['null_perc', 'pseudolabel_acc']]['pseudolabel_acc', 'mean'], yerr=null_results[['null_perc', 'pseudolabel_acc']]['pseudolabel_acc', 'std'])

plt.title('Null Percentage Study')
plt.legend(['Default Classifier', 'Self-Learning Classifier'])
plt.ylabel('Accuracy')
plt.xlabel('Null Percentage')

#Self-supervised learning using RF and MLP
mlp = MLPClassifier()
mlp_param = {
    'hidden_layer_sizes': [(50,), (50, 50), (5, 50, 50)],
    'alpha': [0.0001, 0.001, 0.01]
}

results = pd.DataFrame()

for threshold in tqdm(thresh):
    for null_perc in percs_null:
        
        # Creating a test set for us to validate our results (and compare to a non-self-learning classifier)
        X_train, X_test, y_train, y_test = train_test_split(
            df.data, df.target, test_size=0.3, shuffle=True)
        
        # Normalizing the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Randomly removing null_perc % of labels from training set
        rng = np.random.RandomState()
        random_unlabeled_points = rng.rand(y_train.shape[0]) < null_perc

        y_train[random_unlabeled_points] = -1
        new_y_train = y_train.copy()

        # Training loop
        for i in range(pl_epochs):
            # Choose the classifier to use
            if i % 2 == 0:
                clf = model_rf
                param = parameters
            else:
                clf = mlp
                param = mlp_param

            # Select the labeled set
            X = X_train[np.where(new_y_train != -1)]
            y = new_y_train[np.where(new_y_train != -1)]

            # Select the unlabeled set
            X_un = X_train[np.where(new_y_train == -1)]
            y_un = new_y_train[np.where(new_y_train == -1)]

            if len(y_un) == 0:
                break

            # Hyperparameter optimization
            clf_ = GridSearchCV(clf, param, cv=2).fit(X, y).best_estimator_

            # Probability Calibration    
            calibrated_clf = CalibratedClassifierCV(base_estimator=clf_,
                                                    cv=2)
                                                    #ensemble=False)
            calibrated_clf.fit(X, y)
            preds = calibrated_clf.predict_proba(X_un)

            # Adding the high confidence labels
            classes = np.argmax(preds, axis=1)
            classes_probabilities = np.max(preds, axis=1)

            high_confidence_classes = classes[np.where(classes_probabilities >= threshold)]

            y_un[np.where(classes_probabilities >= threshold)] = high_confidence_classes

            new_y_train[np.where(new_y_train == -1)] = y_un

        # Validation
        X = X_train[np.where(new_y_train != -1)]
        y = new_y_train[np.where(new_y_train != -1)]
        calibrated_clf.fit(X, y)

        y_pred_self_learning = calibrated_clf.predict(X_test)

        X = X_train[np.where(y_train != -1)]
        y = y_train[np.where(y_train != -1)]

        calibrated_clf.fit(X, y)
        y_pred = calibrated_clf.predict(X_test)
        
        results = pd.concat([results, pd.DataFrame([{
            'threshold': threshold, 'null_perc': null_perc,
            'normal_acc': accuracy_score(y_test, y_pred),
            'pseudo_acc': accuracy_score(y_test, y_pred_self_learning)
        }])])

thres_results = results.groupby('threshold', as_index=False).agg({'normal_acc': ['mean', 'std'], 'pseudo_acc': ['mean', 'std']})
null_results = results.groupby('null_perc', as_index=False).agg({'normal_acc': ['mean', 'std'], 'pseudo_acc': ['mean', 'std']})

plt.figure(figsize=(12, 7))

plt.errorbar(null_results['null_perc'], y=null_results[['null_perc', 'normal_acc']]['normal_acc', 'mean'], yerr=null_results[['null_perc', 'normal_acc']]['normal_acc', 'std'])
plt.errorbar(null_results['null_perc'], y=null_results[['null_perc', 'pseudo_acc']]['pseudo_acc', 'mean'], yerr=null_results[['null_perc', 'pseudo_acc']]['pseudo_acc', 'std'])

plt.title('Null Percentage Study for two classifiers')
plt.legend(['Default Classifier', 'Self-Learning Classifier'])
plt.ylabel('Accuracy')
plt.xlabel('Null Percentage')

plt.figure(figsize=(12, 7))

plt.errorbar(thres_results['threshold'], y=thres_results[['threshold', 'normal_acc']]['normal_acc', 'mean'], yerr=thres_results[['threshold', 'normal_acc']]['normal_acc', 'std'])
plt.errorbar(thres_results['threshold'], y=thres_results[['threshold', 'pseudo_acc']]['pseudo_acc', 'mean'], yerr=thres_results[['threshold', 'pseudo_acc']]['pseudo_acc', 'std'])

plt.title('Threshold Study for two classifiers')
plt.legend(['Default Classifier', 'Self-Learning Classifier'])
plt.ylabel('Accuracy')
plt.xlabel('Threshold')