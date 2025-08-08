#! /usr/bin/env python3

import napari
import imageio
import numpy as np
import os as os
import pandas as pd

from skimage import data
from skimage import io
from os import listdir
from os.path import isfile, join
from ipyfilechooser import FileChooser

import time
from qtpy.QtWidgets import QMainWindow, QAction, QFileDialog

from pandas import DataFrame
from qtpy.QtCore import QTimer, Qt, QSortFilterProxyModel
from qtpy.QtWidgets import (
    QLabel,
    QLineEdit,
    QTableWidget,
    QHBoxLayout,
    QTableWidgetItem,
    QWidget,
    QGridLayout,
    QPushButton,
    QFileDialog,
    QComboBox,
)

from typing import Union
import matplotlib.pyplot as plt

path_dir = os.getcwd()
fc_dir_rawData = None
fc_rawData = None


scale_voxel = [0.1344, 0.1025, 0.1025]
unit_voxel = "um"

im_as_tiff = False

viewer = napari.Viewer(ndisplay=3)


def load_image():
    path, _ = QFileDialog.getOpenFileName(None, "Select a 4D image", ".", "*.tif")
    if path:
        image = io.imread(path)
        viewer.add_image(image, name="raw image", scale=scale_voxel)

        # projection moyenne
        AIP = np.average(image, axis=0)
        viewer.add_image(AIP, name="AIP", scale=scale_voxel)


def load_labels():
    path, _ = QFileDialog.getOpenFileName(None, "Select labels", ".", "*.tif")
    if path:
        labels = io.imread(path).astype(np.uint32)
        label_layer = viewer.add_labels(labels, name="labels", scale=scale_voxel)
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = unit_voxel


def load_csv():
    global path_csv
    path_csv, _ = QFileDialog.getOpenFileName(None, "Select CSv file", ".", "*.csv")
    if path_csv:
        reg_props = pd.read_csv(path_csv, sep=";")

        # Vérifier que la couche 'labels' est présente
        if "labels" in viewer.layers:
            label_layer = viewer.layers["labels"]
            label_layer.properties = reg_props
            add_table(label_layer, viewer)

            # Optionnel : colorer les labels selon une propriété
            if "Class" in reg_props.columns:
                from napari.utils.colormaps import label_colormap

                unique_classes = reg_props["Class"].unique()
                class_to_color = {c: i + 1 for i, c in enumerate(unique_classes)}

                label_colors = {
                    int(row["Label"]): class_to_color[row["Class"]]
                    for _, row in reg_props.iterrows()
                }
                label_layer.color = label_colors
        else:
            print("No 'labels' layer found in the viewer. Please load labels first.")


# Créer les actions et le menu
menu_bar = viewer.window._qt_window.menuBar()
menu_fichier = menu_bar.addMenu("Segmentation_files")

action_image = QAction("Load raw data", viewer.window._qt_window)
action_image.triggered.connect(load_image)
menu_fichier.addAction(action_image)

action_labels = QAction("Load labels", viewer.window._qt_window)
action_labels.triggered.connect(load_labels)
menu_fichier.addAction(action_labels)

action_csv = QAction("Load CSV", viewer.window._qt_window)
action_csv.triggered.connect(load_csv)
menu_fichier.addAction(action_csv)


# ----------------------------------------------------------------------
# Loading of the features table


class TableWidget(QWidget):
    """
    The table widget represents a table inside napari.
    Tables are just views on `properties` of `layers`.
    """

    def __init__(self, layer: napari.layers.Layer, viewer: napari.Viewer = None):
        super().__init__()

        self._layer = layer
        self._viewer = viewer
        self._hidden_rows = []

        self._view = QTableWidget()
        self._view.setSortingEnabled(True)  # To be able to sort by column
        self._view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self._view.clicked.connect(self._clicked_table)
        layer.mouse_drag_callbacks.append(self._clicked_labels)

        # Initialiser tous les widgets d'interface
        self._label_text_box = QLineEdit()
        self._highlight_box = QLineEdit()
        self._highlight_box.setPlaceholderText("ex: 12 or 12,15,18")
        self._class_combo = QComboBox()
        self._class_combo.addItem("Select a class")

        # Définir le layout principal
        main_layout = QGridLayout()
        self.setLayout(main_layout)

        # Boutons d'action existants
        filter_button = QPushButton("Find simultaneous events")
        filter_button.clicked.connect(self._filter)

        reset_button = QPushButton("Reset table")
        reset_button.clicked.connect(self._reset_view_from_viewer)

        # Layout pour les boutons d'action existants
        action_widget = QWidget()
        action_layout = QHBoxLayout()
        action_layout.addWidget(filter_button)
        action_layout.addWidget(self._label_text_box)
        action_layout.addWidget(reset_button)
        action_layout.setSpacing(3)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_widget.setLayout(action_layout)

        # Zone de saisie pour labels à afficher
        highlight_button = QPushButton("Events masks generator")
        highlight_button.clicked.connect(self._highlight_labels)

        highlight_widget = QWidget()
        highlight_layout = QHBoxLayout()
        highlight_layout.addWidget(self._highlight_box)
        highlight_layout.addWidget(highlight_button)
        highlight_layout.setSpacing(3)
        highlight_layout.setContentsMargins(0, 0, 0, 0)
        highlight_widget.setLayout(highlight_layout)

        # Zone pour sélection par classe
        class_filter_button = QPushButton("Class mask generator")
        class_filter_button.clicked.connect(self._generate_class_mask)

        class_widget = QWidget()
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Class:"))
        class_layout.addWidget(self._class_combo)
        class_layout.addWidget(class_filter_button)
        class_layout.setSpacing(3)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_widget.setLayout(class_layout)

        # Ajout de tous les widgets au layout principal
        main_layout.addWidget(action_widget)
        main_layout.addWidget(highlight_widget)
        main_layout.addWidget(class_widget)
        main_layout.addWidget(self._view)

        self.setWindowTitle("Properties of " + layer.name)

        # Charger le contenu après avoir initialisé tous les widgets
        if hasattr(layer, "properties"):
            self.set_content(layer.properties)
        else:
            self.set_content({})

    def _highlight_labels(self):
        text = self._highlight_box.text()
        if not text:
            print("No labels provided to highlight.")
            return

        try:
            labels_to_show = [int(x.strip()) for x in text.split(",")]
        except ValueError:
            print(
                "Invalid label format. Please enter integers separated by commas: 12, 15, 18"
            )
            return

        labels_data = self._layer.data

        # Filtrer les labels
        filtered_labels = np.where(np.isin(labels_data, labels_to_show), labels_data, 0)

        if "label_selected" in self._viewer.layers:
            self._viewer.layers["label_selected"].data = filtered_labels
        else:
            self._viewer.add_labels(
                filtered_labels, name="label_selected", scale=scale_voxel, opacity=1
            )

    def _generate_class_mask(self):
        """Génère un masque contenant tous les événements d'une classe donnée"""
        selected_class = self._class_combo.currentText()

        if not selected_class or selected_class == "Select a class":
            print("No class selected for mask generation.")
            return

        # Vérifier que les données nécessaires sont disponibles
        if (
            not hasattr(self, "_table")
            or "Class" not in self._table
            or "Label" not in self._table
        ):
            print("Unable to generate mask: missing class or label data.")
            return

        # Trouver tous les labels correspondant à la classe sélectionnée
        labels_of_class = []
        for i, class_name in enumerate(self._table["Class"]):
            if class_name == selected_class:
                labels_of_class.append(self._table["Label"][i])

        if not labels_of_class:
            print(f"No labels found for class '{selected_class}'.")
            return

        print(
            f"Mask generation for class '{selected_class}' with {len(labels_of_class)} labels."
        )
        print(f"Labels of class '{selected_class}': {labels_of_class}")

        # Créer le masque
        labels_data = self._layer.data
        class_mask = np.where(np.isin(labels_data, labels_of_class), labels_data, 0)

        # Nom de la couche basé sur la classe
        layer_name = f"mask_{selected_class}"

        # Ajouter ou mettre à jour la couche dans le viewer
        if layer_name in self._viewer.layers:
            self._viewer.layers[layer_name].data = class_mask
        else:
            self._viewer.add_labels(
                class_mask, name=layer_name, scale=scale_voxel, opacity=0.7
            )

        print(f"Successfully generated mask for class '{selected_class}'.")

    def _update_class_combo(self):
        """Met à jour la liste déroulante des classes disponibles"""
        self._class_combo.clear()
        self._class_combo.addItem("Sélectionner une classe")

        if hasattr(self, "_table") and "Class" in self._table:
            unique_classes = sorted(set(self._table["Class"]))
            for class_name in unique_classes:
                self._class_combo.addItem(str(class_name))

    def _clicked_table(self):
        if "Label" in self._table.keys():
            row = self._view.currentRow()
            label = self._table["Label"][row]
            print("Table clicked, set label", label)
            self._layer.selected_label = label

            frame_column = _determine_frame_column(self._table)
            if frame_column is not None and self._viewer is not None:
                frame = self._table[frame_column][row]
                current_step = list(self._viewer.dims.current_step)
                if len(current_step) >= 4:
                    current_step[-4] = frame
                    self._viewer.dims.current_step = current_step

    def _after_labels_clicked(self):
        if "Label" in self._table.keys() and hasattr(self._layer, "selected_label"):
            row = self._view.currentRow()
            label = self._table["Label"][row]

            frame_column = _determine_frame_column(self._table)
            if frame_column is not None and self._viewer is not None:
                current_step = list(self._viewer.dims.current_step)
                if len(current_step) >= 4:
                    frame = current_step[-4]

            if label != self._layer.selected_label:
                if frame_column is not None and self._viewer is not None:
                    for r, (l, f) in enumerate(
                        zip(self._table["Label"], self._table[frame_column])
                    ):
                        if l == self._layer.selected_label and f == frame:
                            self._view.setCurrentCell(r, self._view.currentColumn())
                            break
                else:
                    for r, l in enumerate(self._table["Label"]):
                        if l == self._layer.selected_label:
                            self._view.setCurrentCell(r, self._view.currentColumn())
                            break

    # We need to run this later as the labels_layer.selected_label isn't changed yet.
    def _clicked_labels(self, event, event1):
        QTimer.singleShot(200, self._after_labels_clicked)

    def _save_clicked(self, event=None, filename=None):
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save as csv...", ".", "*.csv"
            )
        DataFrame(self._table).to_csv(filename)

    def _copy_clicked(self):
        DataFrame(self._table).to_clipboard()

    def _filter(self):
        self._reset_view()

        label_value = self._label_text_box.text()
        nbFrame = self._layer.data.shape[0]

        if label_value == "":
            string = "No label provided!"
            print(string)
            self._label_text_box.setText(string)
        elif float(label_value) > len(self._table["index"]):
            string = (
                "Provided label should be inferior to "
                + str(len(self._table["index"]) + 1)
                + "!"
            )
            print(string)
            self._label_text_box.setText(string)
        elif float(label_value) == 0:
            string = "Provided label should be superior to " + str(0) + "!"
            print(string)
            self._label_text_box.setText(string)
        else:
            # Compute silmutaneous event: events starting at the same frame
            start_frame = self._table["T0 [frame]"][int(label_value) - 1]

            for index in self._table["index"]:
                index = int(index)
                if self._table["T0 [frame]"][index - 1] != start_frame:
                    self._hidden_rows.append(int(index) - 1)

            for row in self._hidden_rows:
                self._view.setRowHidden(row, True)

    def _reset_view_from_viewer(self):
        for row in self._hidden_rows:
            self._view.setRowHidden(row, False)
        self._hidden_rows = []
        self._label_text_box.setText("")

    def _reset_view(self):
        for row in self._hidden_rows:
            self._view.setRowHidden(row, False)
        self._hidden_rows = []

    def set_content(self, table: dict):
        """
        Overwrites the content of the table with the content of a given dictionary.
        """
        if table is None:
            table = {}

        # Workaround to fix wrong row display in napari status bar
        # https://github.com/napari/napari/issues/4250
        # https://github.com/napari/napari/issues/2596
        if "Label" in table.keys() and "index" not in table.keys():
            table["index"] = table["Label"]

            # workaround until this issue is fixed:
            # https://github.com/napari/napari/issues/4342
            if len(np.unique(table["index"])) != len(table["index"]):
                # indices exist multiple times, presumably because it's timelapse data
                def get_status(
                    position,
                    *,
                    view_direction=None,
                    dims_displayed=None,
                    world: bool = False,
                ) -> str:
                    value = self._layer.get_value(
                        position,
                        view_direction=view_direction,
                        dims_displayed=dims_displayed,
                        world=world,
                    )

                    from napari.utils.status_messages import generate_layer_status

                    msg = generate_layer_status(self._layer.name, position, value)
                    return msg

                self._layer.get_status = get_status

                import warnings

                warnings.warn(
                    "Status bar display of label properties disabled because labels/indices exist multiple times (napari-skimage-regionprops)"
                )

        self._table = table

        self._layer.properties = table

        self._view.clear()
        try:
            self._view.setRowCount(len(next(iter(table.values()))))
            self._view.setColumnCount(len(table))
        except StopIteration:
            pass

        for i, column in enumerate(table.keys()):
            self._view.setHorizontalHeaderItem(i, QTableWidgetItem(column))
            for j, value in enumerate(table.get(column)):
                # self._view.setItem(j, i, QTableWidgetItem(str(value)))

                ### Ajout de ma part
                # ATTENTION: If we load everything as string avec la ligne ci dessus, the sort function will give 1,10,2 instead of 1,2,10

                if column == "Class":  # Load as string
                    self._view.setItem(j, i, QTableWidgetItem(str(value)))
                else:  # Load as float.
                    item = QTableWidgetItem()
                    item.setData(Qt.DisplayRole, float(value))
                    self._view.setItem(j, i, item)
                ### Fin ajout de ma part

        self._update_class_combo()

    def get_content(self) -> dict:
        """
        Returns the current content of the table
        """
        return self._table

    def update_content(self):
        """
        Read the content of the table from the associated labels_layer and overwrites the current content.
        """
        self.set_content(self._layer.properties)

    def append_content(self, table: Union[dict, DataFrame], how: str = "outer"):
        """
        Append data to table.

        Parameters
        ----------
        table : Union[dict, DataFrame]
            New data to be appended.
        how : str, OPTIONAL
            Method how to join the data. See also https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.merge.html
        Returns
        -------
        None.
        """
        # Check input type
        if not isinstance(table, DataFrame):
            table = DataFrame(table)

        _table = DataFrame(self._table)

        # Check whether there are common columns and switch merge type accordingly
        common_columns = np.intersect1d(table.columns, _table.columns)
        if len(common_columns) == 0:
            table = pd.concat([table, _table])
        else:
            table = pd.merge(table, _table, how=how, copy=False)

        self.set_content(table.to_dict("list"))


def generate_label_colors(labels):
    cmap = plt.get_cmap("tab10")  # ou "tab20", "gist_ncar", etc.
    unique_labels = sorted(set(labels))
    color_map = {}
    for i, label in enumerate(unique_labels):
        color = (np.array(cmap(i % cmap.N))[:3] * 255).astype(np.uint8)
        color_map[label] = color  # RGB uint8
    return color_map


def add_table(labels_layer: napari.layers.Layer, viewer: napari.Viewer) -> TableWidget:
    """
    Add a table to a viewer and return the table widget. The table will show the `properties` of the given layer.
    """
    dock_widget = get_table(labels_layer, viewer)
    if dock_widget is None:
        dock_widget = TableWidget(labels_layer, viewer)
        # add widget to napari
        viewer.window.add_dock_widget(
            dock_widget, area="right", name="Properties of " + labels_layer.name
        )
    else:
        dock_widget.set_content(labels_layer.properties)
        if not dock_widget.parent().isVisible():
            dock_widget.parent().setVisible(True)

    return dock_widget


def get_table(labels_layer: napari.layers.Layer, viewer: napari.Viewer) -> TableWidget:
    """
    Searches inside a viewer for a given table and returns it. If it cannot find it,
    it will return None.
    """
    import warnings

    # see: https://github.com/napari/napari/issues/3944
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for widget in list(viewer.window._dock_widgets.values()):
            potential_table_widget = widget.widget()
            if isinstance(potential_table_widget, TableWidget):
                if potential_table_widget._layer is labels_layer:
                    return potential_table_widget

    return None


def _determine_frame_column(table):
    candidates = ["T0 [frame]"]
    for c in candidates:
        if c in table.keys():
            return c
    return None


napari.run()
