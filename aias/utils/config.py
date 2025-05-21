import yaml
import os

def load_config(path="aias/config.yaml"):
	if not os.path.exists(path):
		raise FileNotFoundError("Missing config.yaml file")

	with open(path, "r") as f:
		return yaml.safe_load(f)
