import random
import json
import os

root_dir = "./data_RAD"
full_annotations_fpath = os.path.join(root_dir, "train_w_options_new_full.json")
train_annotations_fpath = os.path.join(root_dir, "train_w_options_new.json")
val_annotations_fpath = os.path.join(root_dir, "val_w_options_new.json")
with open(full_annotations_fpath, "r") as f:
    data = json.load(f)
full_size = len(data)
print(full_size)
random.shuffle(data)
val = data[:451]
train = data[451:]
with open(train_annotations_fpath, "w") as f:
    json.dump(train, f)
with open(val_annotations_fpath, "w") as f:
    json.dump(val, f)
