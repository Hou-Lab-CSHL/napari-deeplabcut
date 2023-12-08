import numpy as np
from dataclasses import dataclass
from collections import deque
from enum import auto
from typing import Sequence, Optional
from qtpy.QtCore import QObject, QTimer, Qt
from qtpy.QtWidgets import (QWidget,
                            QHBoxLayout,
                            QVBoxLayout,
                            QGridLayout,
                            QGroupBox,
                            QButtonGroup,
                            QRadioButton,
                            QCheckBox,
                            QComboBox,
                            QSlider,
                            QLabel,
                            QPushButton)
from napari.layers import Image, Points
from napari.layers.utils import color_manager

from napari_deeplabcut.misc import CycleEnum

# napari auto-sets the color cycling behavior to continuous
# when > 16 labels are present
# here we force it back to categorical
# see https://github.com/napari/napari/issues/4746
def guess_continuous(property):
    if issubclass(property.dtype.type, np.floating):
        return True
    else:
        return False
color_manager.guess_continuous = guess_continuous

class LabelMode(CycleEnum):
    """
    Labeling modes.
    SEQUENTIAL: points are placed in sequence, then frame after frame;
        clicking to add an already annotated point has no effect.
    QUICK: similar to SEQUENTIAL, but trying to add an already
        annotated point actually moves it to the cursor location.
    LOOP: the currently selected point is placed frame after frame,
        before wrapping at the end to frame 1, etc.
    """

    SEQUENTIAL = auto()
    QUICK = auto()
    LOOP = auto()

    @classmethod
    def default(cls):
        return cls.SEQUENTIAL

# Description tooltips for the labeling modes radio buttons.
LABEL_MODE_TOOLTIPS = {
    "SEQUENTIAL": "Points are placed in sequence, then frame after frame;\n"
    "clicking to add an already annotated point has no effect.",
    "QUICK": "Similar to SEQUENTIAL, but trying to add an already\n"
    "annotated point actually moves it to the cursor location.",
    "LOOP": "The currently selected point is placed frame after frame,\n"
    "before wrapping at the end to frame 1, etc.",
}

@dataclass
class QtBlocker:
    obj: QObject

    def __enter__(self):
        self.obj.blockSignals(True)

    def __exit__(self, *exception_args):
        self.obj.blockSignals(False)
        return False # pass exception through

class DropdownMenu(QComboBox):
    def __init__(self, labels: Sequence[str], placeholder: Optional[str] = None):
        super().__init__()
        self.set_items(labels, placeholder)

    def current(self):
        if self.currentIndex() >= 0:
            return self.currentText()
        else:
            return None

    def update_placeholder(self, text: str):
        if text is not None:
            self.setPlaceholderText(text)

    def update_to(self, text: str):
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)

    def unselect(self):
        self.setCurrentIndex(-1)

    def reset(self):
        self.setCurrentIndex(0)

    def get_items(self, start_at_current = False):
        idx = deque(list(range(self.count())))
        current = self.currentIndex()
        if start_at_current and current >= 0:
            idx.rotate(-current)

        return [self.itemText(i) for i in idx]

    def set_items(self, items, placeholder = None):
        self.clear()
        self.addItems(items)
        self.update_placeholder(placeholder)

    def blocked(self):
        return QtBlocker(self)

class LikelihoodSlider(QSlider):
    def __init__(self, value = 0):
        super().__init__(Qt.Orientation.Horizontal)
        self.setMinimum(0)
        self.setMaximum(100)
        self.setSingleStep(1)
        self.set_value(value)

    def value(self):
        return super().value() / 100

    def set_value(self, value):
        assert value >= 0 and value <= 1, "LikelihoodSlider value must be in [0, 1]"
        self.setValue(int(value * 100))

# class ControllerState(EventedModel):
#     current_point: Optional[Keypoint] = None
#     label_mode: LabelMode = LabelMode.default()

class Controller(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        # self._state = ControllerState()
        # build layout
        self._layout = QVBoxLayout(self)

        # add misc. controls
        check_groupbox = QGroupBox("Misc. controls")
        layout = QVBoxLayout()
        self._show_gen_checkbox = QCheckBox("Show generated points")
        self._show_gen_checkbox.setChecked(True)
        layout.addWidget(self._show_gen_checkbox)
        self._merge_machine_button = QPushButton("Merge machine labels")
        self._merge_machine_button.setAutoDefault(False)
        self._merge_machine_button.setEnabled(False)
        layout.addWidget(self._merge_machine_button)
        check_groupbox.setLayout(layout)
        self._layout.addWidget(check_groupbox)
        # connect event handlers
        self._show_gen_checkbox.stateChanged.connect(self.on_visibility_change)
        self._merge_machine_button.clicked.connect(self.merge_machine_labels)

        # add labeling mode radio buttons
        label_groupbox = QGroupBox("Labeling mode")
        layout = QHBoxLayout()
        self._label_mode = QButtonGroup(self)
        for i, mode in enumerate(LabelMode.__members__, start=1):
            btn = QRadioButton(mode.lower())
            btn.setToolTip(LABEL_MODE_TOOLTIPS[mode])
            self._label_mode.addButton(btn, i)
            layout.addWidget(btn)
        self._label_mode.button(1).setChecked(True)
        label_groupbox.setLayout(layout)
        self._layout.addWidget(label_groupbox)
        # connect event handlers
        # self._label_mode.buttonClicked.connect(self._update_label_mode)

        # add sliders for confidence and visibility threshold
        slider_groupbox = QGroupBox("Likelihood thresholds")
        layout = QGridLayout()
        label = QLabel("Highlight")
        self._confidence_slider = LikelihoodSlider()
        layout.addWidget(label, 0, 0, 1, 1)
        layout.addWidget(self._confidence_slider, 0, 1, 1, 2)
        label = QLabel("Visibility")
        self._visibility_slider = LikelihoodSlider()
        layout.addWidget(label, 1, 0, 1, 1)
        layout.addWidget(self._visibility_slider, 1, 1, 1, 2)
        slider_groupbox.setLayout(layout)
        self._layout.addWidget(slider_groupbox)
        # connect event handlers
        self._confidence_slider.valueChanged.connect(self.on_confidence_change)
        self._visibility_slider.valueChanged.connect(self.on_visibility_change)

        # add id and keypoint dropdown menu
        keypoint_groupbox = QGroupBox("Keypoint selection")
        layout = QVBoxLayout()
        self._id_menu = DropdownMenu([], placeholder="no points loaded")
        self._keypoint_menu = DropdownMenu([], placeholder="no points loaded")
        layout.addWidget(self._id_menu)
        layout.addWidget(self._keypoint_menu)
        keypoint_groupbox.setLayout(layout)
        self._layout.addWidget(keypoint_groupbox)
        # connect event handlers
        self._id_menu.currentIndexChanged.connect(self.on_idmenu_select)
        self._keypoint_menu.currentIndexChanged.connect(self.on_pointmenu_select)

        # add callbacks for when layers are added/removed/selected
        self.viewer.layers.events.inserted.connect(self.on_layer_insert)
        self.viewer.layers.selection.events.active.connect(self.on_layer_select)

        # add callbacks for when the frame changes
        self.viewer.dims.events.current_step.connect(self.on_frame_change)

    def _update_id_menu(self):
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Points):
            ids = layer.metadata["header"].individuals
        else:
            ids = None
        with self._id_menu.blocked():
            if ids is None:
                self._id_menu.set_items([], "no points layer selected")
            elif len(ids) == 1 and ids[0] == "":
                self._id_menu.set_items([], "single individual")
            else:
                self._id_menu.set_items(ids, "no keypoint selected")
            self._id_menu.unselect()

    def _update_keypoint_menu(self):
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Points):
            bodyparts = layer.metadata["header"].bodyparts
        else:
            bodyparts = []
        with self._keypoint_menu.blocked():
            self._keypoint_menu.set_items(bodyparts, "no keypoint selected")
            self._keypoint_menu.unselect()

    def current_frame(self):
        return self.viewer.dims.current_step[0]

    def set_frame(self, frame):
        frame = max(min(frame, self.viewer.dims.nsteps[0]), 0)
        self.viewer.dims.set_current_step(0, frame)

    def current_id(self):
        current_id = self._id_menu.current()

        return "" if current_id is None else current_id

    def current_keypoint(self):
        return self._keypoint_menu.current()

    def label_mode(self):
        return LabelMode(self._label_mode.checkedButton().text().lower())

    def has_human_labels(self, ignore = None):
        return any(not layer.metadata["header"].is_machine_labeled()
                   for layer in self.viewer.layers if (layer != ignore and
                                                       isinstance(layer, Points)))

    def existing_points(self, layer, ignore = None):
        if isinstance(layer, Points):
            idx = np.arange(layer.data.shape[0])
            if ignore is not None:
                ignore = np.where(ignore == -1, layer.data.shape[0] - 1, ignore)
                idx = np.setdiff1d(idx, ignore, assume_unique=True)

            return ((layer.data[idx, 0] == self.current_frame()) &
                    (layer.properties["id"][idx] == self.current_id())).nonzero()[0]
        else:
            raise RuntimeError("Cannot get existing points for layer of type"
                               f"{type(layer)} (only for Points layers).")

    def merge_machine_labels(self, _):
        machine_layer = self.viewer.layers.selection.active
        delete_layer = False
        for layer in self.viewer.layers:
            if isinstance(layer, Points) and not layer.metadata["header"].is_machine_labeled():
                with layer.events.data.blocker():
                    npts = layer.data.shape[0]
                    layer.add(machine_layer.data)
                    properties = layer.properties
                    for k, v in properties.items():
                        v[npts:] = machine_layer.properties[k]
                        properties[k] = v
                    layer.properties = properties
                delete_layer = True
                layer.refresh()
        if delete_layer:
            self.viewer.layers.remove(machine_layer)

    def on_layer_insert(self, event):
        # get the newest layer
        layer = event.source[-1]
        if isinstance(layer, Image):
            # paths = layer.metadata.get("paths")
            # # Store the metadata and pass them on to the other layers
            # self._images_meta.update(
            #     {
            #         "paths": paths,
            #         "shape": layer.level_shapes[0],
            #         "root": layer.metadata["root"],
            #         "name": layer.name,
            #     }
            # )
            # Move the image layer to the bottom of the layer stack
            QTimer.singleShot(10,
                              lambda: self.viewer.layers.move_selected(event.index, 0))
        elif isinstance(layer, Points):
            # we only allow a single human labels layer
            if not layer.metadata["header"].is_machine_labeled():
                if self.has_human_labels(ignore=layer):
                    raise RuntimeError("Labeling tool only supports a single"
                                       " human labeled layer!")

            # disable labels over keypoints
            layer.text.visible = False
            # set slider positions
            self._confidence_slider.set_value(layer.metadata["confidence_thresh"])
            self._visibility_slider.set_value(layer.metadata["visibility_thresh"])
            # add handler for when points are selected
            layer.events.highlight.connect(self.on_point_select)
            layer.events.mode.connect(self.on_point_mode)
            layer.events.data.connect(self.on_point_add)

    def on_layer_select(self, _):
        layer = self.viewer.layers.selection.active
        # update dropdown menus based on selected layer
        self._update_id_menu()
        self._update_keypoint_menu()
        # if this an a machine layer, activate the merge controls
        if (isinstance(layer, Points) and
            layer.metadata["header"].is_machine_labeled() and
            self.has_human_labels(ignore=layer)):
            self._merge_machine_button.setEnabled(True)
        else:
            self._merge_machine_button.setEnabled(False)

    def on_frame_change(self, _):
        layer = self.viewer.layers.selection.active
        if (isinstance(layer, Points) and
            layer.mode == "add" and
            self.label_mode() != LabelMode.LOOP):
            self.select_next_unlabeled_point(from_start=True)

    def on_confidence_change(self, _):
        for layer in self.viewer.layers:
            if isinstance(layer, Points):
                thresh = self._confidence_slider.value()
                props = {k: v for k, v in layer.properties.items()}
                metadata = {k: v for k, v in layer.metadata.items()}
                metadata["confidence_thresh"] = thresh
                valid = props["likelihood"] > thresh
                props["valid"] = np.where(valid, "valid", "invalid")
                layer.edge_width = np.where(valid, 0, 1)
                layer.properties = props
                layer.metadata = metadata

    def on_visibility_change(self, _):
        for layer in self.viewer.layers:
            if isinstance(layer, Points):
                thresh = self._visibility_slider.value()
                show_gen = self._show_gen_checkbox.isChecked()
                likelihood = layer.properties["likelihood"]
                isgenerated = layer.properties["generated"]
                metadata = {k: v for k, v in layer.metadata.items()}
                metadata["visibility_thresh"] = thresh
                layer.shown = (likelihood > thresh) & (show_gen | ~isgenerated)
                layer.metadata = metadata

    def on_point_mode(self, _):
        layer = self.viewer.layers.selection.active
        if layer.mode == "add":
            self.select_next_unlabeled_point()
        else:
            self._keypoint_menu.unselect()

    def on_point_add(self, event):
        layer = self.viewer.layers.selection.active
        if event.action == "add":
            idx = np.asarray(event.data_indices)
            idx = np.where(idx == -1, layer.data.shape[0] - 1, idx)
            properties = layer.current_properties
            current_points = self.existing_points(layer, ignore=idx)
            current_labels = layer.properties["label"][current_points]
            current_ids = layer.properties["id"][current_points]
            matched_points = ((properties["label"][0] == current_labels) &
                              (properties["id"][0] == current_ids)).nonzero()[0]
            if np.any(matched_points):
                if self.label_mode() == LabelMode.QUICK:
                    layer.data[current_points[matched_points]] = layer.data[idx]
                with layer.events.blocker_all():
                    layer.selected_data = list(idx)
                    layer.remove_selected()
            full = (len(self.existing_points(layer)) ==
                    len(self._keypoint_menu.get_items()))
            if full or (self.label_mode() == LabelMode.LOOP):
                self.set_frame(self.current_frame() + 1)
            else:
                self.select_next_unlabeled_point()

    def on_point_select(self, _):
        # get selection properties
        layer = self.viewer.layers.selection.active
        if layer is None or not isinstance(layer, Points) or layer.mode != "select":
            return
        properties = layer.current_properties
        selection = layer.selected_data.active
        # update the dropdowns
        with self._id_menu.blocked(), self._keypoint_menu.blocked():
            if selection is None:
                self._id_menu.unselect()
                self._keypoint_menu.unselect()
            else:
                if properties["id"][0] != "":
                    self._id_menu.update_to(properties["id"][0])
                self._keypoint_menu.update_to(properties["label"][0])

    def on_idmenu_select(self, _):
        self._keypoint_menu.unselect()

    def on_pointmenu_select(self, _):
        current_id = self.current_id()
        current_keypoint = self.current_keypoint()
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Points):
            idx = ((layer.data[:, 0] == self.current_frame()) &
                   (layer.properties["id"] == current_id) &
                   (layer.properties["label"] == current_keypoint)).nonzero()[0]
            if len(idx) == 1:
                layer.selected_data = list(idx)
            else:
                layer.selected_data = []

            if layer.mode == "add":
                if len(idx) == 0:
                    layer.current_properties = {
                        "label": np.asarray([current_keypoint]),
                        "id": np.asarray([current_id]),
                        "likelihood": 1.0,
                        "valid": True,
                        "generated": False
                    }

            with layer.events.highlight.blocker():
                # this is faster than refreshing the whole layer
                layer.refresh()
        else:
            raise RuntimeError("Unknown controller state."
                               " Keypoint menu selection triggered w/o"
                               " a keypoint layer selected.")

    def select_next_unlabeled_point(self, from_start = False):
        layer = self.viewer.layers.selection.active
        if isinstance(layer, Points):
            if from_start:
                self._keypoint_menu.reset()
            current_points = self.existing_points(layer)
            labels = layer.properties["label"][current_points]
            bodyparts = self._keypoint_menu.get_items(start_at_current=True)
            next_point = None
            for bodypart in bodyparts:
                if bodypart not in labels:
                    next_point = bodypart
                    break
            if next_point is None:
                self._keypoint_menu.reset()
            else:
                self._keypoint_menu.update_to(next_point)
