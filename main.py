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
import ctypes
import os
import platform
import sys
from time import time

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QSize, QTranslator, QLocale
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QAction, QGridLayout, QLabel, \
    QPushButton, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit, QRadioButton, QMessageBox, QButtonGroup, QSlider, \
    QTabWidget, QTableWidget, QTableWidgetItem
from joblib import dump, load
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

from Model import Model


class ApplicationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.directory = None

        self.save_model_action = None
        self.load_example_action = None
        self.auto_tuning_action = None
        self.manual_setting_action = None

        # popup widget
        self.import_training_image_widget = None
        self.auto_tuning_widget = None
        self.manual_setting_widget = None

        self.auto_tuning_heatmap = None
        self.hm_toolbars = None
        self.cfs_mat_toolbar = None
        self.roc_curve_toolbar = None

        self.import_test_data_dir_txt = None
        self.predict_btn = None
        self.train_and_predict_btn = None

        self.result_tabs = None

        self.clip = QApplication.clipboard()
        self.table = None

        self.create_menu_bar()
        self.model = Model()
        self.setWindowTitle('RBF-SVM')
        self.resize(771, 600)
        # self.resize(900, 700)

        app_icon = QIcon()
        app_icon.addFile('icons/SVM_margin_16x16.ico', QSize(16, 16))
        app_icon.addFile('icons/SVM_margin_32x32.ico', QSize(32, 32))
        app_icon.addFile('icons/SVM_margin_64x64.ico', QSize(64, 64))
        app_icon.addFile('icons/SVM_margin_256x256.ico', QSize(256, 256))
        self.setWindowIcon(app_icon)

    def create_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        import_training_image_action = QAction('&' + self.tr('Import training images...'), self)
        import_training_image_action.triggered.connect(self.show_import_training_image_widget)
        load_model_action = QAction('&' + self.tr('Load model...'), self)
        load_model_action.triggered.connect(self.load_model)
        self.save_model_action = QAction('&' + self.tr('Save model...'), self)
        self.save_model_action.setDisabled(True)
        self.save_model_action.triggered.connect(self.save_model)
        self.load_example_action = QAction('&' + self.tr('Load example'), self)
        self.load_example_action.triggered.connect(self.load_example)
        exit_action = QAction('&' + self.tr('Exit'), self)
        exit_action.triggered.connect(self.exit)

        file = menu_bar.addMenu('&' + self.tr('File'))
        file.addAction(import_training_image_action)
        file.addAction(load_model_action)
        file.addAction(self.save_model_action)
        file.addSeparator()
        file.addAction(self.load_example_action)
        file.addSeparator()
        file.addAction(exit_action)

        # Hyper-parameter menu
        self.auto_tuning_action = QAction('&' + self.tr('Auto tuning...'), self)
        self.auto_tuning_action.setDisabled(True)
        self.auto_tuning_action.triggered.connect(self.show_auto_tuning_widget)
        self.manual_setting_action = QAction('&' + self.tr('Manual setting...'), self)
        self.manual_setting_action.setDisabled(True)
        self.manual_setting_action.triggered.connect(self.show_manual_setting_widget)

        hyper_params = menu_bar.addMenu('&' + self.tr('Hyper-parameter'))
        hyper_params.addAction(self.auto_tuning_action)
        hyper_params.addAction(self.manual_setting_action)

    @pyqtSlot(dict)
    def get_import_training_img_config_and_load(self, config):
        self.clear_toolbars()
        print('import config:', config)
        self.directory = config['dir']
        start = time()
        try:
            self.model.load_train_data(self.directory, config['dim'], config['mod'], config['dscr'])
            end = time()
            self.shuffle()
            self.save_model_action.setDisabled(True)
            self.auto_tuning_action.setDisabled(False)
            self.manual_setting_action.setDisabled(False)
            import_ok_msg = QMessageBox()
            import_ok_msg.setWindowTitle(self.tr('Import complete'))
            import_ok_msg.setIcon(QMessageBox.Icon.Information)
            import_ok_msg.setText(self.tr('Data has been imported successfully.'))
            import_ok_msg.setInformativeText(
                self.tr('This operation takes') + ' {0:.2f} '.format(end - start) + self.tr('s'))
            import_ok_msg.exec()
        except Exception:
            import_err_msg = QMessageBox()
            import_err_msg.setWindowTitle(self.tr('Import error'))
            import_err_msg.setIcon(QMessageBox.Icon.Warning)
            import_err_msg.setText(self.tr('Data might have wrong format'))
            import_err_msg.setInformativeText(
                self.tr(r'Please make sure the directory contains subfolders which are '
                        r'names of categories, and all images should be in subfolders.'))
            import_err_msg.exec()

    def show_import_training_image_widget(self):
        self.import_training_image_widget = ImportTrainingImage()
        self.import_training_image_widget.ipt_trn_img_sgl.connect(self.get_import_training_img_config_and_load)
        self.import_training_image_widget.show()

    def load_model(self):
        file_name = QFileDialog.getOpenFileName(self, self.tr('Load model'), self.directory,
                                                self.tr('Joblib dump file ') + '(*.joblib)')[0]
        if file_name != '':
            self.directory = file_name
            start = time()
            if isinstance(self.model, Model):
                try:
                    self.model = load(file_name)
                    end = time()
                    self.auto_tuning_action.setDisabled(False)
                    self.manual_setting_action.setDisabled(False)
                    if self.model.grid_search is not None or self.model.random_search is not None:
                        if self.model.y_pred is None:
                            self.auto_tuning_show()
                        else:
                            self.predict_show()
                    else:
                        if self.model.y_pred is None:
                            self.manual_setting_show()
                        else:
                            self.predict_show()
                    load_ok_msg = QMessageBox()
                    load_ok_msg.setWindowTitle(self.tr('Loading complete'))
                    load_ok_msg.setIcon(QMessageBox.Icon.Information)
                    load_ok_msg.setText(self.tr('Model has been loaded successfully.'))
                    load_ok_msg.setInformativeText(
                        self.tr('This operation takes') + ' {0:.2f} '.format(end - start) + self.tr('s'))
                    load_ok_msg.exec()
                    print('File: ' + file_name + ' opened!')
                except Exception:
                    type_err_msg = QMessageBox()
                    type_err_msg.setWindowTitle(self.tr('Loading error'))
                    type_err_msg.setIcon(QMessageBox.Icon.Warning)
                    type_err_msg.setText(self.tr('This might not be a joblib dump file.'))
                    type_err_msg.exec()
            else:
                load_err_msg = QMessageBox()
                load_err_msg.setWindowTitle(self.tr('Loading error'))
                load_err_msg.setIcon(QMessageBox.Icon.Warning)
                load_err_msg.setText(self.tr('This file is not a instance of RBF-SVM.'))
                load_err_msg.exec()
        else:
            print('Saving as has been cancelled.')

    def save_model(self):
        file_name = QFileDialog.getSaveFileName(self, self.tr('Save model'), self.directory,
                                                self.tr('Joblib dump file ') + '(*.joblib)')[0]
        if file_name != '':
            self.directory = file_name
            start = time()
            if not file_name.endswith('.joblib'):
                file_name += '.joblib'
            dump(self.model, file_name)
            end = time()
            load_ok_msg = QMessageBox()
            load_ok_msg.setWindowTitle(self.tr('Save complete'))
            load_ok_msg.setIcon(QMessageBox.Icon.Information)
            load_ok_msg.setText(self.tr('Model has been saved successfully.'))
            load_ok_msg.setInformativeText(
                self.tr('This operation takes') + ' {0:.2f} '.format(end - start) + self.tr('s'))
            load_ok_msg.exec()
            print('File saved as: {}'.format(file_name))
        else:
            print('Saving as has been cancelled.')

    def load_example(self):
        self.clear_toolbars()
        self.model.load_example_dataset()
        self.shuffle()
        self.save_model_action.setDisabled(True)
        self.auto_tuning_action.setDisabled(False)
        self.manual_setting_action.setDisabled(False)

    @staticmethod
    def exit():
        sys.exit()

    @pyqtSlot(dict)
    def get_auto_tuning_config_and_train(self, config):
        print('tuning config:', config)
        if config['mod'] == 'grid_log':
            self.model.tuning_hyperparams_grid(cv=config['cv'], C_begin=config['C_begin'], C_end=config['C_end'],
                                               gamma_begin=config['gamma_begin'], gamma_end=config['gamma_end'])
        elif config['mod'] == 'grid_lin':
            self.model.tuning_hyperparams_grid(cv=config['cv'], C_begin=config['C_begin'], C_end=config['C_end'],
                                               gamma_begin=config['gamma_begin'], gamma_end=config['gamma_end'],
                                               C_inter=config['C_inter'], gamma_inter=config['gamma_inter'])
        elif config['mod'] == 'rand':
            self.model.tuning_hyperparams_random(cv=config['cv'], n_iter=config['n_iter'],
                                                 C_lambda=config['C_lambda'], gamma_lambda=config['gamma_lambda'])
        start = time()
        self.model.train()
        end = time()
        self.auto_tuning_show()
        self.show_training_finished_message_box(end, start)

    def show_training_finished_message_box(self, end, start):
        training_ok_msg = QMessageBox()
        training_ok_msg.setWindowTitle(self.tr('Training complete'))
        training_ok_msg.setIcon(QMessageBox.Icon.Information)
        training_ok_msg.setText(self.tr('Data has been trained successfully.'))
        training_ok_msg.setInformativeText(
            self.tr('This operation takes') + ' {0:.2f} '.format(end - start) + self.tr('s'))
        training_ok_msg.exec()

    def auto_tuning_show(self):
        self.save_model_action.setDisabled(False)
        self.train_and_predict_btn = None
        self.clear_toolbars()
        layout = QVBoxLayout()
        self.auto_tuning_heatmap = QTabWidget()
        heatmaps = []
        self.hm_toolbars = []
        for i in range(self.model.cv + 5):
            heatmaps.append(QWidget())
        values = ['mean_test_score', 'std_test_score', 'rank_test_score', 'mean_fit_time', 'std_fit_time']
        values_names = [self.tr('Mean test score'), self.tr('Std test score'), self.tr('Rank test score'),
                        self.tr('Mean fit time'), self.tr('Std fit time')]

        for i in range(self.model.cv):
            values.append('split{}_test_score'.format(i))
            values_names.append(self.tr('Split ') + str(i + 1) + self.tr(' test score'))
        for i in range(self.model.cv + 5):
            htm_layout = QVBoxLayout(heatmaps[i])
            canvas = FigureCanvas(Figure())
            htm_layout.addWidget(canvas)
            self.hm_toolbars.append(NavigationToolbar(canvas, self))
            self.addToolBar(self.hm_toolbars[i])
            self.hm_toolbars[i].setVisible(False)
            ax = canvas.figure.subplots()
            self.model.get_hyperparams_heatmap(values=values[i], ax=ax)
            self.auto_tuning_heatmap.addTab(heatmaps[i], values_names[i])

        self.hm_toolbars[0].setVisible(True)
        self.auto_tuning_heatmap.currentChanged.connect(self.hm_tab_onchanged)
        layout.addWidget(self.auto_tuning_heatmap)

        tuning_show_layout = QHBoxLayout()
        best_params = self.model.get_best_params()
        tuning_show_layout.addWidget(
            QLabel(self.tr('Best parameters:') + ' C = {0:.4f}, γ = {1:.4f}'.format(best_params['C'],
                                                                                    best_params['gamma'])))
        self.add_load_and_predict_button_then_show(layout, tuning_show_layout)

    def add_load_and_predict_button_then_show(self, layout, bottom_layout):
        self.try_add_test_dir_btn(bottom_layout)
        self.predict_btn = QPushButton(self.tr('Load and Predict'))
        if not self.model.is_load_example:
            self.predict_btn.setDisabled(True)
        self.predict_btn.clicked.connect(self.predict)
        bottom_layout.addWidget(self.predict_btn, alignment=Qt.AlignCenter)
        layout.addLayout(bottom_layout)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def import_test_images(self):
        self.directory = QFileDialog.getExistingDirectory(self, self.tr('Choose image directory'), self.directory)
        self.import_test_data_dir_txt.setText(self.directory)
        if self.directory != '':
            print('image dir:', self.directory)

            if self.predict_btn is not None:
                self.predict_btn.setDisabled(False)
            elif self.train_and_predict_btn is not None:
                self.train_and_predict_btn.setDisabled(False)
        else:
            print('file chooser cancelled')
            if self.predict_btn is not None:
                self.predict_btn.setDisabled(True)
            elif self.train_and_predict_btn is not None:
                self.train_and_predict_btn.setDisabled(True)

    def predict(self):
        if not self.model.is_load_example:
            self.model.load_test_data(self.directory)
        self.model.predict()
        self.predict_show()

    def predict_show(self):
        self.clear_toolbars()
        layout = QVBoxLayout()

        self.result_tabs = QTabWidget()
        # clf tbl
        clf_repo = self.model.get_classification_report()
        self.table = self.dataframe_to_QTableWidget(clf_repo)

        self.result_tabs.addTab(self.table, self.tr('Classification report'))
        layout.addWidget(self.result_tabs)
        # cfs mat
        cfs_mat = QWidget()
        cfs_mat_layout = QVBoxLayout(cfs_mat)
        canvas = FigureCanvas(Figure())
        cfs_mat_layout.addWidget(canvas)
        self.cfs_mat_toolbar = NavigationToolbar(canvas, self)
        self.addToolBar(self.cfs_mat_toolbar)
        self.cfs_mat_toolbar.setVisible(False)
        ax = canvas.figure.subplots()
        self.model.get_confusion_matrix(ax=ax)
        self.result_tabs.addTab(cfs_mat, self.tr('Confusion matrix'))
        # roc curve
        roc_curve = QWidget()
        roc_curve_layout = QVBoxLayout(roc_curve)
        canvas = FigureCanvas(Figure())
        roc_curve_layout.addWidget(canvas)
        self.roc_curve_toolbar = NavigationToolbar(canvas, self)
        self.addToolBar(self.roc_curve_toolbar)
        self.roc_curve_toolbar.setVisible(False)
        ax = canvas.figure.subplots()
        self.model.get_roc_curve(ax=ax)
        ax.grid()
        self.result_tabs.addTab(roc_curve, self.tr('ROC curve'))
        self.result_tabs.currentChanged.connect(self.res_tab_onchanged)

        pred_show_layout = QHBoxLayout()
        self.add_load_and_predict_button_then_show(layout, pred_show_layout)

    def hm_tab_onchanged(self):
        for hm_toolbar in self.hm_toolbars:
            hm_toolbar.setVisible(False)
        curr_selected_ind = self.auto_tuning_heatmap.currentIndex()
        self.hm_toolbars[curr_selected_ind].setVisible(True)

    def res_tab_onchanged(self):
        curr_selected_ind = self.result_tabs.currentIndex()
        if curr_selected_ind == 0:
            self.cfs_mat_toolbar.setVisible(False)
            self.roc_curve_toolbar.setVisible(False)
        elif curr_selected_ind == 1:
            self.cfs_mat_toolbar.setVisible(True)
            self.roc_curve_toolbar.setVisible(False)
        elif curr_selected_ind == 2:
            self.cfs_mat_toolbar.setVisible(False)
            self.roc_curve_toolbar.setVisible(True)

    def show_auto_tuning_widget(self):
        self.auto_tuning_widget = AutoTuning()
        self.auto_tuning_widget.at_tn_sgl.connect(self.get_auto_tuning_config_and_train)
        self.auto_tuning_widget.show()

    @pyqtSlot(dict)
    def get_manual_setting_and_cv(self, config):
        print('manual setting:', config)
        self.model.set_hyperparams(config['C'], config['gamma'])
        start = time()
        self.model.cross_validate(config['cv'])
        end = time()
        self.manual_setting_show()
        cv_ok_msg = QMessageBox()
        cv_ok_msg.setWindowTitle(self.tr('Cross validate complete'))
        cv_ok_msg.setIcon(QMessageBox.Icon.Information)
        cv_ok_msg.setText(self.tr('Cross validation has been done successfully.'))
        cv_ok_msg.setInformativeText(self.tr('This operation takes') + ' {0:.2f} '.format(end - start) + self.tr('s'))
        cv_ok_msg.exec()

    def manual_setting_show(self):
        self.save_model_action.setDisabled(False)
        self.predict_btn = None
        self.clear_toolbars()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(self.tr('Cross validation result')))
        self.table = self.dataframe_to_QTableWidget(self.model.cv_score)
        layout.addWidget(self.table)

        cv_show_layout = QHBoxLayout()
        curr_param_C, curr_param_gamma = self.model.param_C, self.model.param_gamma
        print(curr_param_C, curr_param_gamma)
        if type(curr_param_gamma) is not str:
            params_lbl = QLabel(
                self.tr('Current parameters:') + ' C = {0:.4f}, γ = {1:.4f}'.format(curr_param_C, curr_param_gamma))
        else:
            params_lbl = QLabel(
                self.tr('Current parameters:') + ' C = {0:.4f}, γ = {1}'.format(curr_param_C, curr_param_gamma))
        cv_show_layout.addWidget(params_lbl)
        self.add_load_and_predict_button_then_show(layout, cv_show_layout)

    def train_and_predict(self):
        start = time()
        self.model.train()
        end = time()
        self.predict()
        self.show_training_finished_message_box(end, start)

    def try_add_test_dir_btn(self, hbox_layout):
        if not self.model.is_load_example:
            self.import_test_data_dir_txt = QLineEdit()
            self.import_test_data_dir_txt.setReadOnly(True)
            hbox_layout.addWidget(self.import_test_data_dir_txt)
            import_test_data_btn = QPushButton(self.tr('Import test images...'))
            import_test_data_btn.clicked.connect(self.import_test_images)
            hbox_layout.addWidget(import_test_data_btn, alignment=Qt.AlignCenter)

    @staticmethod
    def dataframe_to_QTableWidget(dataframe):
        dataframe_tbl = QTableWidget()
        dataframe_tbl.setColumnCount(dataframe.columns.size)
        dataframe_tbl.setRowCount(dataframe.index.size)
        for i in range(dataframe.columns.size):
            dataframe_tbl.setHorizontalHeaderItem(i, QTableWidgetItem(dataframe.columns[i]))
        for i in range(dataframe.index.size):
            dataframe_tbl.setVerticalHeaderItem(i, QTableWidgetItem(dataframe.index[i]))
        for row in range(dataframe.index.size):
            for col in range(dataframe.columns.size):
                dataframe_tbl.setItem(row, col, QTableWidgetItem(str(np.round(dataframe.values[row][col], decimals=2))))
        return dataframe_tbl

    def keyPressEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            selected = self.table.selectedRanges()

            if e.key() == Qt.Key_C:  # copy
                try:
                    s = '\t' + "\t".join([str(self.table.horizontalHeaderItem(i).text()) for i in
                                          range(selected[0].leftColumn(), selected[0].rightColumn() + 1)])
                    s = s + '\n'

                    for r in range(selected[0].topRow(), selected[0].bottomRow() + 1):
                        s += self.table.verticalHeaderItem(r).text() + '\t'
                        for c in range(selected[0].leftColumn(), selected[0].rightColumn() + 1):
                            try:
                                s += str(self.table.item(r, c).text()) + "\t"
                            except AttributeError:
                                s += "\t"
                        s = s[:-1] + "\n"  # eliminate last '\t'
                    self.clip.setText(s)
                except:
                    pass

    def show_manual_setting_widget(self):
        self.manual_setting_widget = ManualSetting()
        self.manual_setting_widget.mn_st_sgl.connect(self.get_manual_setting_and_cv)
        self.manual_setting_widget.show()

    def shuffle(self):
        layout = QGridLayout()
        for i in range(6):
            canvas = FigureCanvas(Figure())
            layout.addWidget(canvas, (i // 3) * 2, i % 3)
            ax = canvas.figure.subplots()
            rand_img, target = self.model.get_random_train_image()
            if self.model.image_color_mode == 'gray':
                ax.imshow(rand_img, cmap='gray')
            else:
                ax.imshow(rand_img)
            ax.set_xticks([])
            ax.set_yticks([])
            img_lbl = QLabel(target)
            layout.addWidget(img_lbl, (i // 3) * 2 + 1, i % 3, alignment=Qt.AlignCenter)
        shuffle_btn = QPushButton(self.tr("Shuffle"))
        shuffle_btn.clicked.connect(self.shuffle)
        layout.addWidget(shuffle_btn, 4, 2, alignment=Qt.AlignCenter)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def clear_toolbars(self):
        if self.hm_toolbars is not None:
            for hm_toolbar in self.hm_toolbars:
                hm_toolbar.setVisible(False)
            self.hm_toolbars = None
        if self.cfs_mat_toolbar is not None:
            self.cfs_mat_toolbar.setVisible(False)
            self.cfs_mat_toolbar = None
        if self.roc_curve_toolbar is not None:
            self.roc_curve_toolbar.setVisible(False)
            self.roc_curve_toolbar = None


class ImportTrainingImage(QWidget):
    ipt_trn_img_sgl = pyqtSignal(dict)

    def __init__(self):
        QWidget.__init__(self)
        self.directory = ''
        self.setWindowTitle(self.tr('Import training image'))
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        # Directory
        dir_layout = QHBoxLayout()
        self.dir_txt = QLineEdit()
        self.dir_txt.setReadOnly(True)
        dir_layout.addWidget(self.dir_txt)
        dir_choose_btn = QPushButton(self.tr('Choose...'))
        dir_choose_btn.clicked.connect(self.choose_file)
        dir_layout.addWidget(dir_choose_btn)
        form_layout.addRow(self.tr('Directory'), dir_layout)
        # Image dimension
        dimension_layout = QHBoxLayout()
        self.dimension_txt_1 = QLineEdit()
        self.dimension_txt_1.setText('32')
        self.dimension_txt_1.setValidator(QIntValidator())
        self.dimension_txt_1.setMaxLength(2)
        dimension_layout.addWidget(self.dimension_txt_1)
        dimension_layout.addWidget(QLabel('×'))
        self.dimension_txt_2 = QLineEdit('32')
        self.dimension_txt_2.setValidator(QIntValidator())
        self.dimension_txt_2.setMaxLength(2)
        dimension_layout.addWidget(self.dimension_txt_2)
        form_layout.addRow(self.tr('Image dimension'), dimension_layout)
        # Color mode
        color_layout = QHBoxLayout()
        self.gray_radio_btn = QRadioButton(self.tr('Grayscale'))
        self.gray_radio_btn.setChecked(True)
        color_layout.addWidget(self.gray_radio_btn)
        self.rgb_radio_btn = QRadioButton(self.tr('RGB'))
        color_layout.addWidget(self.rgb_radio_btn)
        form_layout.addRow(self.tr('Color mode'), color_layout)
        # Description
        self.dscr_txt = QLineEdit(self.tr('No description.'))
        form_layout.addRow(self.tr('Description'), self.dscr_txt)
        layout.addLayout(form_layout)
        import_btn = QPushButton(self.tr('Import'))
        import_btn.clicked.connect(self.send_config)
        layout.addWidget(import_btn, alignment=Qt.AlignCenter)
        self.setLayout(layout)

    def choose_file(self):
        self.directory = QFileDialog.getExistingDirectory(self, self.tr('Choose image directory'))
        self.dir_txt.setText(self.directory)
        if self.directory != '':
            print('image dir:', self.directory)
        else:
            print('file chooser cancelled')

    @pyqtSlot()
    def send_config(self):
        if self.directory == '':
            empty_dir_msg = QMessageBox()
            empty_dir_msg.setWindowTitle(self.tr('Warning'))
            empty_dir_msg.setIcon(QMessageBox.Icon.Warning)
            empty_dir_msg.setText(self.tr('Please choose directory'))
            empty_dir_msg.setStandardButtons(QMessageBox.Ok)
            empty_dir_msg.exec()
        else:
            config = {'dir': self.directory,
                      'dim': (int(self.dimension_txt_1.text()), int(self.dimension_txt_2.text())),
                      'dscr': self.dscr_txt.text()}
            if self.gray_radio_btn.isChecked():
                config['mod'] = 'gray'
            else:
                config['mod'] = 'rgb'
            self.ipt_trn_img_sgl.emit(config)
            self.close()


class AutoTuning(QWidget):
    at_tn_sgl = pyqtSignal(dict)

    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle(self.tr('Choose auto tuning method and parameters'))
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        # CV
        cv_layout = QHBoxLayout()
        self.cv_sld = QSlider(Qt.Horizontal)
        self.cv_sld.setMaximum(10)
        self.cv_sld.setMinimum(2)
        self.cv_sld.setValue(5)
        self.cv_sld.setTickInterval(1)
        cv_layout.addWidget(self.cv_sld)
        self.cv_sld.valueChanged.connect(self.cv_sld_changed)
        self.cv_lbl = QLabel(self.tr('5 Folds'))
        cv_layout.addWidget(self.cv_lbl)
        form_layout.addRow(self.tr('Cross validation'), cv_layout)
        # mode
        mode_layout = QGridLayout()
        search_mode_group = QButtonGroup(self)
        self.grid_radio_btn = QRadioButton(self.tr('Gird search'))
        self.grid_radio_btn.setChecked(True)
        self.grid_radio_btn.toggled.connect(self.grid_toggled)
        search_mode_group.addButton(self.grid_radio_btn)
        self.rand_radio_btn = QRadioButton(self.tr('Random search'))
        self.rand_radio_btn.toggled.connect(self.rand_toggled)
        search_mode_group.addButton(self.rand_radio_btn)
        mode_layout.addWidget(self.grid_radio_btn, 0, 0)
        mode_layout.addWidget(self.rand_radio_btn, 0, 1)
        search_scale_group = QButtonGroup(self)
        self.log_radio_btn = QRadioButton(self.tr('Logarithmic scale'))
        self.log_radio_btn.setChecked(True)
        self.log_radio_btn.toggled.connect(self.log_toggled)
        search_scale_group.addButton(self.log_radio_btn)
        self.lin_radio_btn = QRadioButton(self.tr('Linear scale'))
        self.lin_radio_btn.toggled.connect(self.lin_toggled)
        search_scale_group.addButton(self.lin_radio_btn)
        mode_layout.addWidget(self.log_radio_btn, 1, 0)
        mode_layout.addWidget(self.lin_radio_btn, 1, 1)
        form_layout.addRow(self.tr('Tuning method'), mode_layout)
        # Hyper-parameters
        hyper_params_layout = QVBoxLayout()
        # n_iter
        self.rand_iter_layout = QHBoxLayout()
        rand_iter_lbl = QLabel(self.tr('Iterations'))
        self.rand_iter_layout.addWidget(rand_iter_lbl)
        self.rand_iter_txt = QLineEdit('20')
        self.rand_iter_txt.setValidator(QIntValidator())
        self.rand_iter_txt.setMaxLength(3)
        self.rand_iter_layout.addWidget(self.rand_iter_txt, alignment=Qt.AlignCenter)
        for i in range(2):
            self.rand_iter_layout.itemAt(i).widget().setVisible(False)
        hyper_params_layout.addLayout(self.rand_iter_layout)
        # C
        self.param_C_layout = QHBoxLayout()
        self.param_C_layout.addWidget(QLabel('C:['))
        self.param_C_left_txt = QLineEdit('0.01')
        self.param_C_left_txt.setValidator(QDoubleValidator())
        self.param_C_left_txt.setMaxLength(6)
        self.param_C_layout.addWidget(self.param_C_left_txt, alignment=Qt.AlignCenter)
        self.param_C_layout.addWidget(QLabel(','))
        self.param_C_right_txt = QLineEdit('100')
        self.param_C_right_txt.setValidator(QDoubleValidator())
        self.param_C_right_txt.setMaxLength(6)
        self.param_C_layout.addWidget(self.param_C_right_txt, alignment=Qt.AlignCenter)
        self.param_C_layout.addWidget(QLabel(']'))
        self.param_C_layout.addWidget(QLabel(self.tr('Number')))
        self.param_C_num_txt = QLineEdit('10')
        self.param_C_num_txt.setValidator(QIntValidator())
        self.param_C_num_txt.setMaxLength(3)
        self.param_C_layout.addWidget(self.param_C_num_txt, alignment=Qt.AlignCenter)
        self.param_C_layout.addWidget(QLabel('λ_C'))
        self.param_C_lambda_txt = QLineEdit('100')
        self.param_C_lambda_txt.setValidator(QDoubleValidator())
        self.param_C_lambda_txt.setMaxLength(6)
        self.param_C_layout.addWidget(self.param_C_lambda_txt, alignment=Qt.AlignCenter)
        for i in range(5, 9):
            self.param_C_layout.itemAt(i).widget().setVisible(False)
        hyper_params_layout.addLayout(self.param_C_layout)
        # gamma
        self.param_gamma_layout = QHBoxLayout()
        self.param_gamma_layout.addWidget(QLabel('γ:['))
        self.param_gamma_left_txt = QLineEdit('0.01')
        self.param_gamma_left_txt.setValidator(QDoubleValidator())
        self.param_gamma_left_txt.setMaxLength(6)
        self.param_gamma_layout.addWidget(self.param_gamma_left_txt, alignment=Qt.AlignCenter)
        self.param_gamma_layout.addWidget(QLabel(','))
        self.param_gamma_right_txt = QLineEdit('100')
        self.param_gamma_right_txt.setValidator(QDoubleValidator())
        self.param_gamma_right_txt.setMaxLength(6)
        self.param_gamma_layout.addWidget(self.param_gamma_right_txt, alignment=Qt.AlignCenter)
        self.param_gamma_layout.addWidget(QLabel(']'))
        self.param_gamma_layout.addWidget(QLabel(self.tr('Number')))
        self.param_gamma_num_txt = QLineEdit('10')
        self.param_gamma_num_txt.setValidator(QIntValidator())
        self.param_gamma_num_txt.setMaxLength(3)
        self.param_gamma_layout.addWidget(self.param_gamma_num_txt, alignment=Qt.AlignCenter)
        self.param_gamma_layout.addWidget(QLabel('λ_γ'))
        self.param_gamma_lambda_txt = QLineEdit('0.01')
        self.param_gamma_lambda_txt.setValidator(QDoubleValidator())
        self.param_gamma_lambda_txt.setMaxLength(6)
        self.param_gamma_layout.addWidget(self.param_gamma_lambda_txt, alignment=Qt.AlignCenter)
        for i in range(5, 9):
            self.param_gamma_layout.itemAt(i).widget().setVisible(False)
        hyper_params_layout.addLayout(self.param_gamma_layout)
        form_layout.addRow(self.tr('Hyper-parameters'), hyper_params_layout)
        layout.addLayout(form_layout)
        tune_and_train_btn = QPushButton(self.tr('Tune and Train'))
        tune_and_train_btn.clicked.connect(self.tune_and_train)
        layout.addWidget(tune_and_train_btn, alignment=Qt.AlignCenter)
        self.setLayout(layout)

    def grid_toggled(self):
        self.log_radio_btn.setVisible(True)
        self.lin_radio_btn.setVisible(True)
        self.log_radio_btn.setChecked(True)
        for i in range(2):
            self.rand_iter_layout.itemAt(i).widget().setVisible(False)
        self.log_toggled()

    def rand_toggled(self):
        self.log_radio_btn.setVisible(False)
        self.lin_radio_btn.setVisible(False)
        for i in range(2):
            self.rand_iter_layout.itemAt(i).widget().setVisible(True)
        for i in range(7):
            self.param_C_layout.itemAt(i).widget().setVisible(False)
            self.param_gamma_layout.itemAt(i).widget().setVisible(False)
        for i in range(7, 9):
            self.param_C_layout.itemAt(i).widget().setVisible(True)
            self.param_gamma_layout.itemAt(i).widget().setVisible(True)

    def log_toggled(self):
        for i in range(5):
            self.param_C_layout.itemAt(i).widget().setVisible(True)
            self.param_gamma_layout.itemAt(i).widget().setVisible(True)
        for i in range(5, 9):
            self.param_C_layout.itemAt(i).widget().setVisible(False)
            self.param_gamma_layout.itemAt(i).widget().setVisible(False)

    def lin_toggled(self):
        for i in range(7):
            self.param_C_layout.itemAt(i).widget().setVisible(True)
            self.param_gamma_layout.itemAt(i).widget().setVisible(True)
        for i in range(7, 9):
            self.param_C_layout.itemAt(i).widget().setVisible(False)
            self.param_gamma_layout.itemAt(i).widget().setVisible(False)

    def cv_sld_changed(self):
        cv_value = self.cv_sld.value()
        self.cv_lbl.setText(str(cv_value) + self.tr(' Folds'))

    def tune_and_train(self):
        config = {'cv': self.cv_sld.value()}
        if self.grid_radio_btn.isChecked():
            if float(self.param_C_left_txt.text()) > float(self.param_C_right_txt.text()) or \
                    float(self.param_gamma_left_txt.text()) > float(self.param_gamma_right_txt.text()):
                interval_error_msg = QMessageBox()
                interval_error_msg.setWindowTitle(self.tr('Warning'))
                interval_error_msg.setIcon(QMessageBox.Icon.Warning)
                interval_error_msg.setText(self.tr('Right bound must greater than left bound.'))
                interval_error_msg.setStandardButtons(QMessageBox.Ok)
                interval_error_msg.exec()
            else:
                config['C_begin'] = float(self.param_C_left_txt.text())
                config['C_end'] = float(self.param_C_right_txt.text())
                config['gamma_begin'] = float(self.param_gamma_left_txt.text())
                config['gamma_end'] = float(self.param_gamma_right_txt.text())
                if self.log_radio_btn.isChecked():
                    config['mod'] = 'grid_log'
                elif self.lin_radio_btn.isChecked():
                    config['C_inter'] = int(self.param_C_num_txt.text())
                    config['gamma_inter'] = int(self.param_gamma_num_txt.text())
                    config['mod'] = 'grid_lin'
                self.at_tn_sgl.emit(config)
                self.close()
        elif self.rand_radio_btn.isChecked():
            config['n_iter'] = int(self.rand_iter_txt.text())
            config['C_lambda'] = float(self.param_C_lambda_txt.text())
            config['gamma_lambda'] = float(self.param_C_lambda_txt.text())
            config['mod'] = 'rand'
            self.at_tn_sgl.emit(config)
            self.close()


class ManualSetting(QWidget):
    mn_st_sgl = pyqtSignal(dict)

    def __init__(self):
        QWidget.__init__(self)
        self.setWindowTitle(self.tr('Manual setting'))
        layout = QVBoxLayout()
        params_layout = QFormLayout()
        # CV
        cv_layout = QHBoxLayout()
        self.cv_sld = QSlider(Qt.Horizontal)
        self.cv_sld.setMaximum(10)
        self.cv_sld.setMinimum(2)
        self.cv_sld.setValue(5)
        self.cv_sld.setTickInterval(1)
        cv_layout.addWidget(self.cv_sld)
        self.cv_sld.valueChanged.connect(self.cv_sld_changed)
        self.cv_lbl = QLabel(self.tr('5 Folds'))
        cv_layout.addWidget(self.cv_lbl)
        params_layout.addRow(self.tr('Cross validation'), cv_layout)

        hyper_params_layout = QHBoxLayout()
        param_C_lbl = QLabel('C')
        hyper_params_layout.addWidget(param_C_lbl)
        self.param_C_txt = QLineEdit('1')
        self.param_C_txt.setValidator(QDoubleValidator())
        self.param_C_txt.setMaxLength(5)
        hyper_params_layout.addWidget(self.param_C_txt, alignment=Qt.AlignCenter)
        param_gamma_lbl = QLabel('γ')
        hyper_params_layout.addWidget(param_gamma_lbl)
        self.param_gamma_txt = QLineEdit('scale')
        self.param_gamma_txt.setMaxLength(5)
        hyper_params_layout.addWidget(self.param_gamma_txt, alignment=Qt.AlignCenter)
        params_layout.addRow(self.tr('Hyper-parameters'), hyper_params_layout)

        layout.addLayout(params_layout)
        cv_btn = QPushButton(self.tr('Cross Validate'))
        cv_btn.clicked.connect(self.cv)
        layout.addWidget(cv_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def cv_sld_changed(self):
        cv_value = self.cv_sld.value()
        self.cv_lbl.setText(str(cv_value) + self.tr(' Folds'))

    def cv(self):
        param_gamma = self.param_gamma_txt.text()
        if param_gamma.isnumeric():
            param_gamma = float(param_gamma)
            config = {'cv': self.cv_sld.value(), 'C': float(self.param_C_txt.text()), 'gamma': param_gamma}
            self.mn_st_sgl.emit(config)
            self.close()
        elif param_gamma == 'scale' or param_gamma == 'auto':
            config = {'cv': self.cv_sld.value(), 'C': float(self.param_C_txt.text()), 'gamma': param_gamma}
            self.mn_st_sgl.emit(config)
            self.close()
        else:
            param_gamma_error_msg = QMessageBox()
            param_gamma_error_msg.setWindowTitle(self.tr('Warning'))
            param_gamma_error_msg.setIcon(QMessageBox.Icon.Warning)
            param_gamma_error_msg.setText('γ ' + self.tr('should be auto, scale or numeric.'))
            param_gamma_error_msg.setStandardButtons(QMessageBox.Ok)
            param_gamma_error_msg.exec()


if __name__ == "__main__":
    system_type = platform.system()
    if system_type == 'Windows':
        appid = 'jordangong.rbf_svm'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)

    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    qapp = QApplication(sys.argv)

    locale = QLocale.system().name()
    translator = QTranslator()
    translator.load('translate/' + locale + '.qm')
    qapp.installTranslator(translator)

    app = ApplicationWindow()
    app.show()
    qapp.exec_()
