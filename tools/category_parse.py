import fiftyone.zoo as foz
import numpy as np
from pathlib import Path
import os


# data parameters
dataset_name = "voc-2007"

# initial import from fiftyone.zoo
dataset_train = foz.load_zoo_dataset(
    dataset_name,
    splits=["train"]
)
dataset_val = foz.load_zoo_dataset(
    dataset_name,
    splits=["validation"]
)

cat_ids_train = dataset_train.distinct("ground_truth.detections.label")
cat_ids_val = dataset_val.distinct("ground_truth.detections.label")
cat_ids = {label: (cat_id + 1) for cat_id, label in enumerate(np.unique(cat_ids_train + cat_ids_val))}
id_cats = {cat_ids[key]: key for key in cat_ids.keys()}

base_dir = os.path.join("config", dataset_name)
Path(base_dir).mkdir(parents=True, exist_ok=True)
filename = os.path.join(base_dir, "id2str_mapping.txt")

lines = []
for int_id, str_id in id_cats.items():
    lines.append("{}={}".format(int_id, str_id))
with open(filename, 'w') as file:
    file.write(str.join("\n", lines))
