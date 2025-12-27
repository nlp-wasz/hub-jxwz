CATEGORY_NAME = ['差评', '好评']

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BERT_MODEL_PKL_PATH = os.path.join(BASE_DIR, "assets/weights/bert.pt")
BERT_MODEL_PERTRAINED_PATH = os.path.join(BASE_DIR, "assets/models/google-bert/bert-base-chinese/")