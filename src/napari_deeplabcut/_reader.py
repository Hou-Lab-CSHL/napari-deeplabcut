import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from dask_image.imread import imread
from napari.types import LayerData
from natsort import natsorted
from napari_video.napari_video import VideoReaderNP

from napari_deeplabcut import misc

SUPPORTED_IMAGES = ".jpg", ".jpeg", ".png"
SUPPORTED_VIDEOS = ".mp4", ".mov", ".avi"


def get_hdf_reader(path):
    if isinstance(path, list):
        path = path[0]

    if not path.endswith(".h5"):
        return None

    return read_hdf


def get_image_reader(path):
    if isinstance(path, list):
        path = path[0]

    if not any(path.lower().endswith(ext) for ext in SUPPORTED_IMAGES):
        return None

    return read_images


def get_video_reader(path):
    if isinstance(path, str) and any(
        path.lower().endswith(ext) for ext in SUPPORTED_VIDEOS
    ):
        return read_video
    return None


def get_config_reader(path):
    if isinstance(path, list):
        path = path[0]

    if not path.endswith(".yaml"):
        return None

    return read_config


def get_folder_parser(path):
    if not os.path.isdir(path):
        return None

    layers = []
    files = os.listdir(path)
    images = ""
    for file in files:
        if any(file.lower().endswith(ext) for ext in SUPPORTED_IMAGES):
            images = os.path.join(path, f"*{os.path.splitext(file)[1]}")
            break
    if not images:
        raise OSError("No supported images were found.")

    layers.extend(read_images(images))
    datafile = ""
    for file in os.listdir(path):
        if file.endswith(".h5"):
            datafile = os.path.join(path, "*.h5")
            break
    if datafile:
        layers.extend(read_hdf(datafile))

    return lambda _: layers


def read_images(path):
    if isinstance(path, list):
        root, ext = os.path.splitext(path[0])
        path = os.path.join(os.path.dirname(root), f"*{ext}")
    # Retrieve filepaths exactly as parsed by pims
    filepaths = []
    for filepath in glob.iglob(path):
        relpath = Path(filepath).parts[-3:]
        filepaths.append(os.path.join(*relpath))
    params = {
        "name": "images",
        "metadata": {
            "paths": natsorted(filepaths),
            "root": os.path.split(path)[0],
        },
    }

    # https://github.com/soft-matter/pims/issues/452
    if len(filepaths) == 1:
        path = glob.glob(path)[0]

    return [(imread(path), params, "image")]


def _populate_metadata(
    header: misc.DLCHeader,
    *,
    labels: Optional[Sequence[str]] = None,
    ids: Optional[Sequence[str]] = None,
    likelihood: Optional[Sequence[float]] = None,
    paths: Optional[List[str]] = None,
    size: Optional[int] = 8,
    pcutoff: Optional[float] = 0.9,
    colormap: Optional[str] = "viridis",
    generated: bool = False
) -> Dict:
    if labels is None:
        labels = np.array([""])
    if ids is None:
        ids = np.array([""])
    if likelihood is None:
        likelihood = np.ones(len(labels))
    elif isinstance(likelihood, pd.Series):
        likelihood = np.asarray(likelihood.values)
    if paths is None:
        paths = np.array([""])
    face_color_cycle_maps = misc.build_color_cycles(header, colormap)
    face_color_prop = "id" if ids[0] else "label"
    return {
        "name": "keypoints",
        "text": "{id}–{label}" if ids[0] else "label",
        "properties": {
            "label": labels,
            "id": ids,
            "likelihood": likelihood,
            "valid": np.where(likelihood > pcutoff, "valid", "invalid"),
            "generated": np.repeat(generated, len(labels)),
            "path": paths
        },
        "shown": likelihood > pcutoff,
        "face_color_cycle": face_color_cycle_maps[face_color_prop],
        "face_color": face_color_prop,
        "face_colormap": colormap,
        "edge_color": "valid",
        "edge_color_cycle": {"valid": "black", "invalid": "red"},
        "edge_width": np.zeros(len(labels)) if ids[0] else 0.0,
        "edge_width_is_relative": False,
        "size": size,
        "metadata": {
            "header": header,
            "face_color_cycles": face_color_cycle_maps,
            "colormap_name": colormap,
            "confidence_thresh": pcutoff,
            "visibility_thresh": pcutoff
        },
    }


def _load_config(config_path: str):
    with open(config_path) as file:
        return yaml.safe_load(file)


def read_config(configname: str) -> List[LayerData]:
    config = _load_config(configname)
    header = misc.DLCHeader.from_config(config)
    metadata = _populate_metadata(
        header,
        size=config["dotsize"]
    )
    metadata["name"] = f"CollectedData_{config['scorer']}"
    metadata["metadata"]["root"] = None
    metadata["metadata"]["name"] = metadata["name"]
    metadata["ndim"] = 3
    metadata["property_choices"] = metadata.pop("properties")
    metadata["metadata"]["project"] = os.path.dirname(configname)
    return [(np.empty((0, 3)), metadata, "points")]


def read_hdf(filename: str) -> List[LayerData]:
    layers = []
    for filename in glob.iglob(filename):
        temp = pd.read_hdf(filename)
        temp = misc.merge_multiple_scorers(temp)
        header = misc.DLCHeader(temp.columns)
        temp = temp.droplevel("scorer", axis=1)
        if "individuals" not in temp.columns.names:
            # Append a fake level to the MultiIndex
            # to make it look like a multi-animal DataFrame
            old_idx = temp.columns.to_frame()
            old_idx.insert(0, "individuals", "")
            temp.columns = pd.MultiIndex.from_frame(old_idx)
            colormap = "viridis"
        else:
            colormap = "Set3"
        if isinstance(temp.index, pd.MultiIndex):
            temp.index = [os.path.join(*row) for row in temp.index]
        df = (
            temp.stack(["individuals", "bodyparts"])
            .reindex(header.individuals, level="individuals")
            .reindex(header.bodyparts, level="bodyparts")
            .reset_index()
        )
        nrows = df.shape[0]
        data = np.empty((nrows, 3))
        image_paths = df["level_0"]
        if np.issubdtype(image_paths.dtype, np.number):
            image_inds = image_paths.values
            image_paths = np.asarray([None for _ in range(len(image_inds))])
        else:
            image_inds = misc.encode_categories(image_paths)
            image_paths = image_paths.values
        data[:, 0] = image_inds
        data[:, 1:] = df[["y", "x"]].to_numpy()
        metadata = _populate_metadata(
            header,
            labels=df["bodyparts"],
            ids=df["individuals"],
            likelihood=df.get("likelihood"),
            paths=image_paths,
            colormap=colormap,
            generated=header.is_machine_labeled()
        )
        metadata["name"] = os.path.split(filename)[1].split(".")[0]
        metadata["metadata"]["root"] = os.path.split(filename)[0]
        # Store file name in case the layer's name is edited by the user
        metadata["metadata"]["name"] = metadata["name"]
        layers.append((data, metadata, "points"))
    return layers

def read_video(path):
    frames = VideoReaderNP(path, remove_leading_singleton=True)
    root, file = os.path.split(path)
    params = {
        "name": file.split(".")[0],
        "metadata": {
            "root": root
        }
    }

    return [(frames, params, 'image')]
