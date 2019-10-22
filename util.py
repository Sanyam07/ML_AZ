# Util
import numpy as np
from collections import namedtuple
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tensorflow import keras
import os
import time

class Util:
    ClassifierMetrics = namedtuple('ClassifierMetrics', 'confmat, precision, recall, f1, p_r_curve, roc_curve, auc')
    PrecisionRecallCurve = namedtuple('PrecisionRecallCurve', 'precisions, recalls, thresholds')
    RocCurve = namedtuple('RocCurve', 'fpr, tpr, thresholds')

    @staticmethod
    def largest_indices(ary, n=3):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    @staticmethod
    def get_clf_metrics_with_y_hat(y, y_prob, y_hat):
        confmat = confusion_matrix(y, y_hat)
        precision = precision_score(y_hat, y)
        recall = recall_score(y_hat, y)
        f1 = f1_score(y_hat, y)
        precisions, recalls, thresholds = precision_recall_curve(y_true=y, probas_pred=y_prob, pos_label=1)
        p_r_curve = Util.PrecisionRecallCurve(precisions=precisions,
                                              recalls=recalls,
                                              thresholds=thresholds)
        fpr, tpr, thresholds = roc_curve(y_true=y, y_score=y_prob, pos_label=1)
        roc = Util.RocCurve(fpr=fpr, tpr=tpr, thresholds=thresholds)
        auc = roc_auc_score(y_true=y, y_score=y_prob)
        metric_clf = Util.ClassifierMetrics(confmat=confmat,
                                            precision=precision,
                                            recall=recall,
                                            f1=f1,
                                            p_r_curve=p_r_curve,
                                            roc_curve=roc,
                                            auc=auc)

        return metric_clf


    @staticmethod
    def get_clf_metrics(model, x, y):
        y_prob = model.decision_function(x)
        y_hat = (y_prob > 0)
        return Util.get_clf_metrics_with_y_hat(y,y_hat)

    @staticmethod
    def get_clean_confusion_matrix(y, y_hat):
        confmat = confusion_matrix(y, y_hat)
        confmat_rel_err = confmat - np.eye(len(confmat))


    @staticmethod
    def plot_clf_metrics(clf_metrics):
        Util.plot_confmat(clf_metrics)
        Util.plot_roc_curve(clf_metrics)
        Util.plot_precision_recall_curve(clf_metrics)

    @staticmethod
    def plot_confmat(clf_metrics):
        fig10=plt.figure(10)
        plt.matshow(clf_metrics.confmat)
        plt.colorbar()

    @staticmethod
    def plot_roc_curve(clf_metrics):
        fig1=plt.figure(1)
        fpr = clf_metrics.roc_curve.fpr
        tpr = clf_metrics.roc_curve.tpr
        auc = clf_metrics.auc
        plt.plot(fpr, tpr)
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('RoC curve, AUC = {:4.2}'.format(auc))

    @staticmethod
    def plot_precision_recall_curve(clf_metrics):
        precisions = clf_metrics.p_r_curve.precisions
        recalls = clf_metrics.p_r_curve.recalls
        thresholds = clf_metrics.p_r_curve.thresholds
        fig2 = plt.figure(2)
        plt.plot(thresholds,precisions[:-1], label='precisions')
        plt.plot(thresholds,recalls[:-1], label='recalls')
        plt.legend()
        plt.xlabel('thresholds')
        plt.ylabel('precision & recall')
        plt.title('precision & recall vs. thresholds')
        fig3 = plt.figure(3)
        plt.plot(precisions, recalls)
        plt.xlabel('precisions')
        plt.ylabel('recalls')
        plt.title('precision-vs-recall curve')


    @staticmethod
    def animate_subplot(data, n_row=3, n_col=4):
        fig, axes = plt.subplots(n_row,n_col, sharex=True, sharey=True)
        ims= Util.fill_subplots(data, axes)
        n_ims = len(ims)
        def animate(i):
            i_start = i*n_ims
            i_end = (i+1)*n_ims
            data_local = data[i_start:i_end]
            Util.ims_set_data(data_local, ims)
            return Util.ims_set_data(data_local, ims)

        anim = animation.FuncAnimation(

            fig, animate, frames=int(data.shape[0]/n_ims), interval=10, blit=True, repeat=True)
        plt.show()

    @staticmethod
    def fill_subplots(data, axes):
        n_row, n_col = axes.shape
        ims = []
        for i_r in range(n_row):
            for i_c in range(n_col):
                i_flat = np.ravel_multi_index([i_r, i_c], (n_row,n_col))
                ims.append(axes[i_r,i_c].imshow(data[i_flat]))

        return ims

    @staticmethod
    def ims_set_data(data, ims):
        for i_m in range(len(ims)):
            ims[i_m].set_data(data[i_m])

        return ims

    @staticmethod
    def prep_keras_data(keras_data=keras.datasets.mnist):
        (X_train_full, y_train_full), (X_test, y_test) = keras_data.load_data()

        # normalize data scale:  rescale to [0,1]
        X_train_full = X_train_full / X_train_full.max()
        X_test = X_test / X_test.max()

        # split and data: further to Train, Valid, (Test)
        X_valid = X_train_full[:5000]
        X_train = X_train_full[5000:]

        y_valid = y_train_full[:5000]
        y_train = y_train_full[5000:]
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    @staticmethod
    def get_run_logdir():
        root_logdir = os.path.join(os.curdir, "my_logs")
        run_id = time.strftime("run_ %Y _ %m _ %d %H _ %M _ %S ")
        return os.path.join(root_logdir, run_id)

