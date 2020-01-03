#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This file is part of RBF-SVM.

    RBF-SVM is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    RBF-SVM is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with RBF-SVM.  If not, see <https://www.gnu.org/licenses/>.
"""
from pathlib import Path
from random import randint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from scipy.stats import expon
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_validate
from sklearn.utils import Bunch


class Model:
    def __init__(self):
        self.is_load_example = False
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_probas = None
        self.images_train = None
        self.images_test = None
        self.image_dimension = None
        self.image_color_mode = None
        self.target_names = None
        self.clf = svm.SVC(kernel='rbf', probability=True)
        self.cv_score = None
        self.grid_search = None
        self.random_search = None
        self.param_C = None
        self.param_gamma = None
        self.cv = None
        self.rand_search_iter = None
        self.accuracy_score = None
        self.classification_report = None

    def load_example_dataset(self):
        self.reset()
        self.is_load_example = True
        digits = load_digits()
        self.image_color_mode = 'gray'
        self.image_dimension = (8, 8)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(digits.data, digits.target,
                                                                                test_size=0.33, random_state=42)
        self.images_train, self.images_test, _, _ = train_test_split(digits.images, digits.target,
                                                                     test_size=0.33, random_state=42)
        self.target_names = np.array([str(name) for name in digits.target_names])

    def get_random_train_image(self):
        rand_idx = randint(0, len(self.X_train) - 1)
        return [self.images_train[rand_idx], self.target_names[self.y_train[rand_idx]]]

    def load_train_data(self, container_path, dimension=(64, 64), mode='gray',
                        description="No description"):
        self.reset()
        self.image_dimension = dimension
        self.image_color_mode = mode
        training_data = self._load_image_files(container_path, self.image_dimension, self.image_color_mode, description)
        self.X_train = training_data.data
        self.y_train = training_data.target
        self.images_train = training_data.images
        self.target_names = training_data.target_names

    def set_hyperparams(self, C=1, gamma='scale'):
        if self.is_load_example:
            self.clear_parms()
        else:
            self.clear()
        self.param_C = C
        self.param_gamma = gamma

    def tuning_hyperparams_grid(self, cv=5, C_begin=0.1, C_end=10, gamma_begin=0.1, gamma_end=10, C_inter=None,
                                gamma_inter=None):
        if self.is_load_example:
            self.clear_parms()
        else:
            self.clear()
        self.cv = cv
        if C_inter is None and gamma_inter is None:
            param_grid = {'C': [10 ** i for i
                                in np.arange(np.log10(C_begin), np.log10(C_end * 10))],
                          'gamma': [10 ** i for i
                                    in np.arange(np.log10(gamma_begin), np.log10(gamma_end * 10))]}
        else:
            param_grid = {'C': np.linspace(C_begin, C_end, C_inter),
                          'gamma': np.linspace(gamma_begin, gamma_end, gamma_inter)}
        self.grid_search = GridSearchCV(self.clf, param_grid, n_jobs=-1, cv=self.cv, iid=False)

    def tuning_hyperparams_random(self, cv=5, n_iter=10, C_lambda=100, gamma_lambda=0.01):
        if self.is_load_example:
            self.clear_parms()
        else:
            self.clear()
        self.cv = cv
        self.rand_search_iter = n_iter
        param_dist = {'C': expon(scale=C_lambda),
                      'gamma': expon(scale=gamma_lambda)}
        self.random_search = RandomizedSearchCV(self.clf, param_dist, self.rand_search_iter, n_jobs=-1, cv=self.cv,
                                                iid=False)

    def cross_validate(self, cv=5):
        self.cv = cv
        self.clf = svm.SVC(C=self.param_C, kernel='rbf', gamma=self.param_gamma, probability=True)
        cv_score = cross_validate(self.clf, self.X_train, self.y_train, cv=self.cv, n_jobs=-1)
        self.cv_score = pd.DataFrame.from_dict(cv_score, orient='index',
                                               columns=['Fold %d' % cv for cv in range(1, self.cv + 1)]).transpose()
        return self.cv_score

    def train(self):
        if self.grid_search is not None:
            print("grid_search")
            self.grid_search.fit(self.X_train, self.y_train)
            self.param_C, self.param_gamma = self.grid_search.best_params_.values()
        elif self.random_search is not None:
            print("random_search")
            self.random_search.fit(self.X_train, self.y_train)
            self.param_C, self.param_gamma = self.random_search.best_params_.values()
        else:
            print("manual")
            self.clf = svm.SVC(C=self.param_C, kernel='rbf', gamma=self.param_gamma, probability=True)
            self.clf.fit(self.X_train, self.y_train)

    def get_tuning_cv_result(self):
        if self.grid_search is not None:
            return self.grid_search.cv_results_
        elif self.random_search is not None:
            return self.random_search.cv_results_

    def get_best_params(self):
        if self.grid_search is not None:
            return self.grid_search.best_params_
        elif self.random_search is not None:
            return self.random_search.best_params_

    def get_hyperparams_heatmap(self, values='mean_test_score', ax=None):
        if self.grid_search is not None or self.random_search is not None:
            table = pd.pivot_table(pd.DataFrame(self.get_tuning_cv_result()),
                                   values=values, index='param_C',
                                   columns='param_gamma')
            heatmap = sns.heatmap(table, ax=ax)
            heatmap.set(xlabel='gamma', ylabel='C')
            return heatmap

    def load_test_data(self, container_path, description="No description"):
        test_data = self._load_image_files(container_path, self.image_dimension, self.image_color_mode, description)
        self.X_test = test_data.data
        self.y_test = test_data.target
        self.images_test = test_data.images

    def predict(self):
        if self.grid_search is not None:
            self.y_pred = self.grid_search.predict(self.X_test)
            self.y_probas = self.grid_search.predict_proba(self.X_test)
        elif self.random_search is not None:
            self.y_pred = self.random_search.predict(self.X_test)
            self.y_probas = self.random_search.predict_proba(self.X_test)
        else:
            self.y_pred = self.clf.predict(self.X_test)
            self.y_probas = self.clf.predict_proba(self.X_test)
        return self.target_names[self.y_pred]

    def get_accuracy(self):
        self.accuracy_score = accuracy_score(self.y_test, self.y_pred)
        return self.accuracy_score

    def get_classification_report(self):
        clf_report = classification_report(self.y_test, self.y_pred, target_names=self.target_names, output_dict=True)
        self.classification_report = pd.DataFrame.from_dict(clf_report).transpose()
        return self.classification_report

    def get_confusion_matrix(self, ax=None, figsize=None):
        return plot_confusion_matrix(self.target_names[self.y_test], self.target_names[self.y_pred], normalize=True,
                                     ax=ax, figsize=figsize)

    def get_roc_curve(self, ax=None, figsize=None):
        return plot_roc(self.target_names[self.y_test], self.y_probas, ax=ax, figsize=figsize)

    def clear(self):
        self.X_test = None
        self.images_test = None
        self.y_test = None
        self.clear_parms()

    def clear_parms(self):
        self.y_pred = None
        self.y_probas = None
        self.clf = svm.SVC(kernel='rbf', probability=True)
        self.cv_score = None
        self.grid_search = None
        self.random_search = None
        self.param_C = None
        self.param_gamma = None
        self.cv = None
        self.rand_search_iter = None
        self.accuracy_score = None
        self.classification_report = None

    def reset(self):
        self.is_load_example = False
        self.X_train = None
        self.images_train = None
        self.y_train = None
        self.image_dimension = None
        self.image_color_mode = None
        self.target_names = None
        self.clear()

    @staticmethod
    def _load_image_files(container_path, dimension, mode, description):
        """
        Load image files with categories as subfolder names 
        which performs like scikit-learn sample dataset
        
        Parameters
        ----------
        container_path : string or unicode
            Path to the main folder holding one subfolder per category
        dimension : tuple
            size to which image are adjusted to
            
        Returns
        -------
        Bunch
        """
        image_dir = Path(container_path)
        folders = [directory for directory in sorted(image_dir.iterdir()) if directory.is_dir()]
        categories = [fo.name for fo in folders]

        flat_data = []
        target = []
        images = []

        for i, directory in enumerate(folders):
            print("Class", categories[i])
            for j, file in enumerate(directory.iterdir()):
                #            print("Sample", j)
                img = imread(str(file))
                if mode == 'gray':
                    img = rgb2gray(img)
                elif mode == 'rgb':
                    pass
                img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
                flat_data.append(img_resized.flatten())
                images.append(img_resized)
                target.append(i)
        flat_data = np.array(flat_data)
        target = np.array(target)
        return Bunch(data=flat_data,
                     target=target,
                     target_names=np.array(categories),
                     images=np.array(images),
                     DESCR=description)


if __name__ == '__main__':
    model = Model()
    # model.load_train_data('test_dataset/train/', dimension=(32, 32))
    model.load_example_dataset()
    fig = plt.figure(dpi=300)
    plt.gray()
    rand_img, target = model.get_random_train_image()
    plt.imshow(rand_img)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    print(target)
    C_begin = 0.001
    C_end = 1000
    gamma_begin = 0.001
    gamma_end = 1000
    model.tuning_hyperparams_grid(5, C_begin, C_end, gamma_begin, gamma_end)

    # model.tuning_hyperparams_random(5, n_iter=30, C_lambda=100, gamma_lambda=.01)
    # model.set_hyperparams()
    # cv_score = model.cross_validate()
    # print("cv_score:\n", cv_score)
    model.train()
    fig, ax = plt.subplots(dpi=300)
    htmap = model.get_hyperparams_heatmap(ax=ax)
    fig.savefig('hyperparams_heatmap.svg')
    # plt.show()
    # model.load_test_data('test_dataset/test/')
    y_pred = model.predict()
    acc = model.get_accuracy()
    print("acc:", acc)
    clf_repo = model.get_classification_report()
    print("clf_repo:\n", clf_repo)
    fig, ax = plt.subplots(dpi=300)
    model.get_confusion_matrix(ax=ax)
    # fig.add_subplot(ax)
    fig.savefig('confusion_matrix.svg')
    # plt.show()
    fig, ax = plt.subplots(dpi=300)
    model.get_roc_curve(ax=ax)
    plt.grid()
    fig.savefig('roc_curve.svg')
    # plt.show()
