# select and train the model
from collections import namedtuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import joblib
from util import Util
from enum import Enum

class Modeler:

    class ModelNames(Enum):
        rf_reg='rf_reg'     # random forest regressor
        lin_reg='lin_reg'   # linear regressor
        tree_reg='tree_reg' # tree regressor
        sgd_clf = 'sgd_clf' # stochastic gradient descent classifier
        rf_clf = 'rf_clf'   # random forest classifier
        svm_clf = 'svm_clf' # svm classifier

    rnd_seed = 1

    rf_param_grid = [{'n_estimators': [3, 10, 30],
                      'max_features': [2, 4, 6, 8]
                      },{
                    'bootstrap': [False],
                    'n_estimators': [3, 10],
                    'max_features': [2, 3, 4]
                   }]
    rf_param_dist = {"max_depth": [3, None],
                     "max_features": sp_randint(1, 11),
                     "min_samples_split": sp_randint(2, 11),
                     "bootstrap": [True, False],
                     "criterion": ["gini", "entropy"]
                     }

    Model = namedtuple('Model', 'model_name, model, scoring, cv, param_grid, param_dist')
    rf_reg_model = Model(model_name = ModelNames.rf_reg,
                         model=RandomForestRegressor(random_state=rnd_seed),
                         scoring='neg_mean_squared_error',
                         cv=10,
                         param_grid=rf_param_grid,
                         param_dist=rf_param_dist)

    lin_reg_model = Model( model_name=ModelNames.lin_reg,
                           model=LinearRegression(),
                           scoring='neg_mean_squared_error',
                           cv=10,
                           param_grid=None,
                           param_dist=None)

    tree_reg_model = Model(model_name=ModelNames.tree_reg,
                           model=DecisionTreeRegressor(),
                           scoring='neg_mean_squared_error',
                           cv=10,
                           param_grid=None,
                           param_dist=None)

    sgd_clf_model = Model(model_name=ModelNames.sgd_clf,
                          model=SGDClassifier(random_state=rnd_seed),
                          scoring='accuracy',
                          cv=3,
                          param_grid=None,
                          param_dist=None)

    rf_clf_model = Model(model_name=ModelNames.rf_clf,
                         model = RandomForestClassifier(random_state=rnd_seed, n_estimators=200),
                         scoring='accuracy',
                         cv=3,
                         param_grid=None,
                         param_dist=None)

    svm_clf_model = Model(model_name=ModelNames.svm_clf,
                          model=SVC(random_state=rnd_seed, probability=True),
                          scoring='accuracy',
                          cv=3,
                          param_grid=None,
                          param_dist=None)

    @staticmethod
    def lin_reg(x, y):
        lin_reg_model, cv_score = Modeler.build(Modeler.lin_reg_model, x, y)
        return lin_reg_model, cv_score

    @staticmethod
    def tree_reg(x, y):
        tree_reg_model, cv_score = Modeler.build(Modeler.tree_reg_model, x, y)
        return tree_reg_model, cv_score

    @staticmethod
    def rf_reg(x,y):
        rf_reg_model, cv_score = Modeler.build(Modeler.rf_reg_model, x, y)
        return rf_reg_model, cv_score

    @staticmethod
    def sgd_clf(x,y):
        sgd_clf_model, cv_score = Modeler.build(Modeler.sgd_clf_model, x, y)
        return sgd_clf_model, cv_score

    @staticmethod
    def rf_clf(x,y):
        rf_clf, cv_score = Modeler.build(Modeler.rf_clf_model, x, y)
        return rf_clf, cv_score

    @staticmethod
    def svm_clf(x,y):
        svm_clf, cv_score = Modeler.build(Modeler.svm_clf_model, x, y)
        return svm_clf, cv_score

    @staticmethod
    def build(model:Model, x, y):
        model.model.fit(x,y)
        scores = cross_val_score(model.model, x, y, cv=model.cv, scoring=model.scoring)
        if model.scoring == 'neg_mean_squared_error':
            scores = np.sqrt(-scores)
        #cv_pred = cross_val_predict(model.model, x, y, cv=model.cv, method='predict_proba')
        cv_pred=None,
        CVScore = namedtuple('CVScore', 'metric, mean, std, scores, pred')
        cv_result = CVScore(metric=model.scoring, scores=scores, mean=scores.mean(), std=scores.std(), pred=cv_pred)
        return model.model, cv_result

    # def build_clf(model, x, y, scoring = 'accuracy'):
    #     model.fit(x, y)
    #     cv = 3
    #     scores = cross_val_score(model, x, y, cv=cv, scoring=scoring)
    #     cv_pred = cross_val_predict(model, x, y, cv=cv)
    #     CVScore = namedtuple('CVScore', 'metric, mean, std, scores, pred')
    #     cv_result = CVScore(metric=scoring, scores=scores, mean=scores.mean(), std=scores.std(), pred=cv_pred)
    #     return model, cv_result

    @staticmethod
    def tune(model: Model, x, y, strategy='GridSearchCV'):
        if strategy == 'GridSearchCV':
            search = GridSearchCV(estimator=model.model,
                                       param_grid=model.param_grid,
                                       cv=5,
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True)
        elif strategy == 'RandomSearchCV':
            search = RandomizedSearchCV(estimator=model,
                                        param_distributions=model.param_dist,
                                        cv=5,
                                        scoring='neg_mean_squared_error',
                                        return_train_score=True)

        search.fit(x,y)
        return search.best_estimator_, search

    @staticmethod
    def validate_clf(model: Model, x, y): # validate on test set
        y_prob = model.decision_function(x)
        clf_metrics = Util.get_clf_metrics(y, y_prob)
        Util.plot_clf_metrics(clf_metrics)
        return clf_metrics

    @staticmethod
    def dump_models(models, pkl_name = 'model.pkl'):
        joblib.dump(models, pkl_name)

    @staticmethod
    def load_models(pkl_name):
        return joblib.load(pkl_name)

    # @staticmethod
    # def predict(y):
    #     y_test_lin_prediction = lin_reg_model.predict(x_test)
    #     rmse_lin = np.sqrt(mean_squared_error(y_test_lin_prediction, y_test))
    #
    #     y_test_tree_prediction = tree_reg_model.predict(x_test)
    #     rmse_tree = np.sqrt(mean_squared_error(y_test_tree_prediction, y_test))
    #
    #     y_test_rf_prediction = rf_reg_model.predict(x_test)
    #     rmse_rf = np.sqrt(mean_squared_error(y_test_rf_prediction, y_test))