import pandas as pd
import numpy as np

import os

from datasets import load_dataset
from transformers import AutoFeatureExtractor, AutoModel

import torchvision.transforms as T
import torch 

import mlflow

from configs import configs
from custom_functions import extract_embeddings

my_local_path = os.getcwd()
model_ckpt = configs.model_ckpt
seed = configs.seed
num_samples = configs.num_samples
proj_name = configs.proj_name
embedding_db_path = configs.embedding_db_path
extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

image_path = my_local_path + '/dataset'
dataset = load_dataset("imagefolder", data_dir=image_path, drop_labels=True)

# Data transformation chain.
transformation_chain = T.Compose(
    [
        # We first resize the input image to 256x256 and then we take center crop.
        T.Resize(int((224 / 224) * extractor.size["height"])),
        T.CenterCrop(extractor.size["height"]),
        T.ToTensor(),
        T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
    ]
)

### producing the embeddigs using a sample of the train data
candidate_subset = dataset["train"].shuffle(seed=seed).select(range(num_samples))
device = "cuda" if torch.cuda.is_available() else "cpu"
extract_fn = extract_embeddings(model.to(device), transformation_chain)
candidate_subset_emb = candidate_subset.map(extract_fn, batched=True, batch_size=24)

### converting the embeddings to torch matrix
all_candidate_embeddings = np.array(candidate_subset_emb["embeddings"])
all_candidate_embeddings = torch.from_numpy(all_candidate_embeddings)

#### saving results

extractor_dict = {
    "size": extractor.size,
    "image_mean":extractor.image_mean,
    "image_std":extractor.image_std,
}

### logging the extractor as json in mlflow
registered_model_name = f'{proj_name}_models' 
with mlflow.start_run():

    mlflow.log_dict(extractor_dict, "extractor_dict.json")

    ### logging the model in mlflow for faster execution
    mlflow.pytorch.log_model(
        model, 
        artifact_path=f"{proj_name}-run",
        registered_model_name = registered_model_name,
    )

##saving embeddings and ids in a torch pt object
if not os.path.exists(embedding_db_path):
        os.makedirs(embedding_db_path)

emb_true_id = torch.from_numpy(np.array(candidate_subset_emb['id']))
embeding_db = torch.cat((all_candidate_embeddings, emb_true_id.unsqueeze(1)), 1)
torch.save(embeding_db, f'{embedding_db_path}embedding_db.pt')