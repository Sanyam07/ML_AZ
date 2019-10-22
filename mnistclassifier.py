# MNIST classifier

from getdata import MNISTData
from buildmodels import Modeler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from util import Util

mnist = MNISTData.get_mnist_data()

X, y = mnist['data'], mnist['target']
y = y.astype('uint8')

X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# clf_model, cv_result_5 = Modeler.sgd_clf(X_train, y_train_5)
# clf_model, cv_result_5 = Modeler.rf_clf(X_train, y_train_5)
clf_model, cv_result_5 = Modeler.svm_clf(X_train, y_train_5)

y_hat_prob_5 = clf_model.predict_proba(X_test)
y_hat_prob_5_pos = y_hat_prob_5[:,1]
y_hat = (y_hat_prob_5_pos>0.5)
clf_metrics = Util.get_clf_metrics_with_y_hat(y_test_5, y_hat_prob_5_pos, y_hat)

Util.plot_clf_metrics(clf_metrics)


