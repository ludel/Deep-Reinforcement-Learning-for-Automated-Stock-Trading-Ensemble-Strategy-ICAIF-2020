import datetime
import os

# data
TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"

now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "test.csv"
