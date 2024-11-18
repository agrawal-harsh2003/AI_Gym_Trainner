
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="/Users/harshagrawal/Downloads/Dataset")

dataset.push_to_hub("harsh-7070/COCO-Wholebody-annotated")
