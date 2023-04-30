import numpy as np
from datasets import load_dataset
import torchvision.transforms as T
import torch 
import mlflow
from tqdm.auto import tqdm
from jobs.configs import configs

run_id = configs.run_id
proj_name = configs.proj_name
embedding_path = configs.embedding_object
tmp_path = configs.image_path

## loading ml objects
def prediction():

    extractor_dict = mlflow.artifacts.load_dict(
        'runs:/'+run_id+'/extractor_dict.json'
    )

    model_local_path = mlflow.artifacts.download_artifacts(
        run_id= run_id,
        artifact_path=f"{proj_name}-run"
    )   

    model = mlflow.pytorch.load_model(model_local_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeding_db = torch.load(embedding_path)

    transformation_chain = T.Compose(
        [
            # We first resize the input image to 256x256 and then we take center crop.
            T.Resize(int((224 / 224) * extractor_dict["size"]["height"])),
            T.CenterCrop(extractor_dict["size"]["height"]),
            T.ToTensor(),
            T.Normalize(mean=extractor_dict["image_mean"], std=extractor_dict["image_std"]),
        ]
    )

    embeddings_loaded = embeding_db[:,0:-1]
    ids_loaded = embeding_db[:,-1]

    candidate_ids = []

    for id in tqdm(range(len(ids_loaded))):
        id_true = int(np.array(ids_loaded[id]))

        # Create a unique indentifier.
        entry = str(id) + "_" + str(id_true)

        candidate_ids.append(entry)

    def compute_scores(emb_one, emb_two):
        """Computes cosine similarity between two vectors."""
        scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)
        return scores.numpy().tolist()


    def fetch_similar(model, all_candidate_embeddings, image, top_k=5):
        """Fetches the `top_k` similar images with `image` as the query."""
        # Prepare the input query image for embedding computation.
        image_transformed = transformation_chain(image).unsqueeze(0)
        new_batch = {"pixel_values": image_transformed.to(device)}

        # Comute the embedding.
        with torch.no_grad():
            query_embeddings = model(**new_batch).last_hidden_state[:, 0].cpu()

        # Compute similarity scores with all the candidate images at one go.
        # We also create a mapping between the candidate image identifiers
        # and their similarity scores with the query image.
        sim_scores = compute_scores(all_candidate_embeddings, query_embeddings)
        similarity_mapping = dict(zip(candidate_ids, sim_scores))
    
        # Sort the mapping dictionary and return `top_k` candidates.
        similarity_mapping_sorted = dict(
            sorted(similarity_mapping.items(), key=lambda x: x[1], reverse=True)
        )
        id_entries = list(similarity_mapping_sorted.keys())[:top_k]

        ids = list(map(lambda x: int(x.split("_")[0]), id_entries))
        ids_true = list(map(lambda x: int(x.split("_")[-1]), id_entries))
        scores = list(similarity_mapping_sorted.values())[:top_k]
        
        return ids, ids_true, scores, similarity_mapping_sorted

    tmp_dataset = load_dataset("imagefolder", data_dir=tmp_path, drop_labels=True)

    tmp_image = tmp_dataset['train'][0]['image']

    sim_ids, sim_ids_true, sim_score, sim_map = fetch_similar(model, embeddings_loaded,tmp_image)

    final_result = dict(zip(sim_ids_true,sim_score))
    print(f"top 5 ids are: {sim_ids}")
    print(f"top 5 actual ids are: {sim_ids_true}")
    print(f"top 5 scores are: {sim_score}")

    return final_result