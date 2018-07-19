"""
Multi-label classification models.

https://en.wikipedia.org/wiki/Multi-label_classification
"""
import sys
import warnings
import numpy as np

from sklearn import linear_model as lm

from keras.models import Sequential
from keras.models import Model

import ipdb

class MLC(Model):
    """
    Multi-label classifier.

    Extends keras.models.Model. Provides additional functionality and metrics
    specific to multi-label classification.
    """

    def __init__(self,inputs,outputs,input_names=None,output_names=None):
        super().__init__(inputs,outputs)
        self.input_names = input_names
        self.output_names = output_names

    def _construct_thresholds(self, probs, targets, top_k=None):

        assert probs.shape == targets.shape, \
            "The shape of predictions should match the shape of targets."
        nb_samples, nb_labels = targets.shape

        top_k = top_k or nb_labels

        # Sort predicted probabilities in descending order
        idx = np.argsort(probs, axis=1)[:,:-(top_k + 1):-1]
        p_sorted = np.vstack([probs[i, idx[i]] for i in range(len(idx))])
        t_sorted = np.vstack([targets[i, idx[i]] for i in range(len(idx))])

        # Compute F-1 measures for every possible threshold position
        F1 = []
        TP = np.zeros(nb_samples)
        FN = t_sorted.sum(axis=1)
        FP = np.zeros(nb_samples)
        for i in range(top_k):
            TP += t_sorted[:,i]
            FN -= t_sorted[:,i]
            FP += 1 - t_sorted[:,i]
            F1.append(2 * TP / (2 * TP + FN + FP))
        F1 = np.vstack(F1).T

        # Find the thresholds
        row = np.arange(nb_samples)
        col = F1.argmax(axis=1)
        p_sorted = np.hstack([p_sorted, np.zeros(nb_samples)[:, None]])
        T = 0.5 * (p_sorted[row, col] + p_sorted[row, col + 1])[:, None]

        return T

    def fit_thresholds(self, data, alpha, batch_size=128, verbose=0,
                       validation_data=None, cv=None, top_k=None):
        #ipdb.set_trace()
        inputs = np.hstack([data[k] for k in self.input_names])
        #inputs = data['X']
        predicts = self.predict(data, batch_size=batch_size)
        if isinstance(predicts,list):
            probs = {self.output_names[i]: k for i,k in enumerate(predicts)}
        else:
            probs = {'Y': predicts}

        #targets = {k: data[k.name] for k in self.outputs}
        targets = {k: data[k] for k in self.output_names}

        if isinstance(alpha, list):
            if validation_data is None and cv is None:
                warnings.warn("Neither validation data, nor the number of "
                              "cross-validation folds is provided. "
                              "The alpha parameter for threshold model will "
                              "be selected based on the default "
                              "cross-validation procedure in RidgeCV.")
            elif validation_data is not None:
                val_inputs = np.hstack([validation_data[k] for k in self.input_names])
                #val_inputs = validation_data['X']
                val_predicts = self.predict(validation_data)
                if isinstance(val_predicts,list):
                    val_probs = {self.output_names[i]: k for i,k in enumerate(val_predicts)}
                else:
                    val_probs = {'Y': self.predict(validation_data['X'])}
                #val_probs = {k: self.predict(validation_data['X'])}
                val_targets = {k: validation_data[k] for k in self.output_names}
                # val_targets = {'Y': validation_data['Y']}

        if verbose:
            sys.stdout.write("Constructing thresholds.")
            sys.stdout.flush()

        self.t_models = {}

        # output_names = [k.name for k in self.outputs]
        # output_names = ['Y']

        for k in self.output_names:
            if verbose:
                sys.stdout.write(".")
                sys.stdout.flush()

            T = self._construct_thresholds(probs[k], targets[k])

            if isinstance(alpha, list):
                if validation_data is not None:
                    val_T = self._construct_thresholds(val_probs[k],
                                                       val_targets[k],
                                                       top_k=top_k)
                    score_best, alpha_best = -np.Inf, None
                    for a in alpha:
                        model = lm.Ridge(alpha=a).fit(inputs, T)
                        score = model.score(val_inputs, val_T)
                        if score > score_best:
                            score_best, alpha_best = score, a
                    alpha = alpha_best
                else:
                    model = lm.RidgeCV(alphas=alpha, cv=cv).fit(inputs, T)
                    alpha = model.alpha_

            self.t_models[k] = lm.Ridge(alpha=alpha)
            self.t_models[k].fit(inputs, T)

        if verbose:
            sys.stdout.write("Done.\n")
            sys.stdout.flush()

    def threshold(self, data, verbose=0):
        inputs = np.hstack([data[k] for k in self.input_names])
        # inputs = data['X']

        if verbose:
            sys.stdout.write("Thresholding...")
            sys.stdout.flush()

        T = {k: self.t_models[k].predict(inputs) for k in self.output_names}

        if verbose:
            sys.stdout.write("Done.\n")
            sys.stdout.flush()

        return T

    def predict_threshold(self, data, batch_size=128, verbose=0):
        predicts = self.predict(data, batch_size=batch_size, verbose=verbose)
        if isinstance(predicts, list):
            probs = {self.output_names[i]: k for i,k in enumerate(predicts)}
        else:
            probs = {'Y': predicts}

        T = self.threshold(data, verbose=verbose)

        preds = {k: probs[k] >= T[k] for k in self.output_names}
        return probs, preds
