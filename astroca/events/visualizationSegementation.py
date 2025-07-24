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
from qtpy.QtWidgets import QLabel, QLineEdit, QTableWidget, QHBoxLayout, QTableWidgetItem, QWidget, QGridLayout, \
    QPushButton, QFileDialog

from typing import Union


path_dir = os.getcwd()
fc_dir_rawData = None
fc_rawData = None

# select raw data
fc_rawData = FileChooser("/home/matteo/Bureau/INRIA/codePython/outputdir/checkDir20/")
# display(fc_rawData)

# select labels
fc_labels = FileChooser("/home/matteo/Bureau/INRIA/codePython/outputdir/checkDir20/")
# display(fc_labels)

# select csv file with features
fc_csv = FileChooser("/home/matteo/Bureau/INRIA/codePython/outputdir/checkDir20/")
# display(fc_csv)

scale_voxel = [0.1344, 0.1025, 0.1025]
unit_voxel = "um"



im_as_tiff = False

# if fc_rawData != None:
#     path_image = fc_rawData.selected
#     im_as_tiff = True;  # True: If the sequence is contained in one tiff image; False: If the sequence is in a directory with several tiff images
#
# if fc_dir_rawData != None:
#     path_image_dir = fc_dir_rawData.selected

# path_image = "/home/matteo/Bureau/INRIA/codeJava/outputdir20/data_boundaries.tif"
# im_as_tiff = True

# path_labels = "/home/matteo/Bureau/INRIA/codeJava/outputdir20/ID_calciumEvents.tif"
# path_csv = "/home/matteo/Bureau/INRIA/codeJava/outputdir20/Features.csv"

# ----------------------------------------------------------------------
# Visualization of the data and the segmentations

#### Open Napari viewer

viewer = napari.Viewer(ndisplay=3)



def load_image():
    path, _ = QFileDialog.getOpenFileName(None, "Sélectionner une image 4D", ".", "*.tif")
    if path:
        image = io.imread(path)
        viewer.add_image(image, name="raw image", scale=scale_voxel)

        # projection moyenne
        AIP = np.average(image, axis=0)
        viewer.add_image(AIP, name="AIP", scale=scale_voxel)

def load_labels():
    path, _ = QFileDialog.getOpenFileName(None, "Sélectionner les labels", ".", "*.tif")
    if path:
        labels = io.imread(path).astype(np.uint32)
        label_layer = viewer.add_labels(labels, name="labels", scale=scale_voxel)
        viewer.scale_bar.visible = True
        viewer.scale_bar.unit = unit_voxel

       

def load_csv():
    global path_csv
    path_csv, _ = QFileDialog.getOpenFileName(None, "Sélectionner le fichier CSV", ".", "*.csv")
    if path_csv:
        reg_props = pd.read_csv(path_csv, sep=';')

        # Vérifier que la couche 'labels' est présente
        if 'labels' in viewer.layers:
            label_layer = viewer.layers['labels']
            label_layer.properties = reg_props
            add_table(label_layer, viewer)

            # Optionnel : colorer les labels selon une propriété
            if 'Class' in reg_props.columns:
                from napari.utils.colormaps import label_colormap
                unique_classes = reg_props['Class'].unique()
                class_to_color = {c: i+1 for i, c in enumerate(unique_classes)}

                label_colors = {int(row["Label"]): class_to_color[row["Class"]] for _, row in reg_props.iterrows()}
                label_layer.color = label_colors
        else:
            print("Aucun label n'a été chargé dans le viewer.")


# Créer les actions et le menu
menu_bar = viewer.window._qt_window.menuBar()
menu_fichier = menu_bar.addMenu("Fichier")

action_image = QAction("Charger image", viewer.window._qt_window)
action_image.triggered.connect(load_image)
menu_fichier.addAction(action_image)

action_labels = QAction("Charger labels", viewer.window._qt_window)
action_labels.triggered.connect(load_labels)
menu_fichier.addAction(action_labels)

action_csv = QAction("Charger CSV", viewer.window._qt_window)
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

        self._view = QTableWidget()

        ### Ajout de ma part
        self._view.setSortingEnabled(True)  # To be able to sort by column
        ### Fin ajout de ma part

        self._view.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        if hasattr(layer, "properties"):
            self.set_content(layer.properties)
        else:
            self.set_content({})

        self._view.clicked.connect(self._clicked_table)
        layer.mouse_drag_callbacks.append(self._clicked_labels)

        ### Ajout de ma part
        filter_button = QPushButton("Find simultaneous events")
        filter_button.clicked.connect(self._filter)

        self._label_text_box = QLineEdit()

        reset_button = QPushButton("Reset table")
        reset_button.clicked.connect(self._reset_view_from_viewer)

        self._hidden_rows = []
        ### Fin ajout de ma part

        self.setWindowTitle("Properties of " + layer.name)
        self.setLayout(QGridLayout())

        data_widget = QWidget()
        data_widget.setLayout(QHBoxLayout())

        action_widget = QWidget()
        action_widget.setLayout(QHBoxLayout())
        action_widget.layout().addWidget(filter_button)
        action_widget.layout().addWidget(self._label_text_box)
        action_widget.layout().addWidget(reset_button)

        self.layout().addWidget(data_widget)
        self.layout().addWidget(action_widget)
        self.layout().addWidget(self._view)
        action_widget.layout().setSpacing(3)
        action_widget.layout().setContentsMargins(0, 0, 0, 0)

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
                    for r, (l, f) in enumerate(zip(self._table["Label"], self._table[frame_column])):
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
        if filename is None: filename, _ = QFileDialog.getSaveFileName(self, "Save as csv...", ".", "*.csv")
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
        elif float(label_value) > len(self._table['index']):
            string = "Provided label should be inferior to " + str(len(self._table['index']) + 1) + "!"
            print(string)
            self._label_text_box.setText(string)
        elif float(label_value) == 0:
            string = "Provided label should be superior to " + str(0) + "!"
            print(string)
            self._label_text_box.setText(string)
        else:
            # Compute silmutaneous event: events starting at the same frame
            start_frame = self._table['T0 [frame]'][int(label_value) - 1]

            for index in self._table['index']:
                index = int(index)
                if (self._table['T0 [frame]'][index - 1] != start_frame):
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
            if len(np.unique(table['index'])) != len(table['index']):
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
                    'Status bar display of label properties disabled because labels/indices exist multiple times (napari-skimage-regionprops)')

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

    def append_content(self, table: Union[dict, DataFrame], how: str = 'outer'):
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

        self.set_content(table.to_dict('list'))


def add_table(labels_layer: napari.layers.Layer, viewer: napari.Viewer) -> TableWidget:
    """
    Add a table to a viewer and return the table widget. The table will show the `properties` of the given layer.
    """
    dock_widget = get_table(labels_layer, viewer)
    if dock_widget is None:
        dock_widget = TableWidget(labels_layer, viewer)
        # add widget to napari
        viewer.window.add_dock_widget(dock_widget, area='right', name="Properties of " + labels_layer.name)
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