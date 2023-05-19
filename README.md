# ImageFinder

Hello!
here I share a detailed description of the ImageFinder.

goal: create an API that given an image, posts the top x,(3 in my case) most similar images

## Summary

The following API takes an image as an input and then gives the three images that resembles it the most in a given database. The obstacles: without labels, a variable database size and different image sizes, which similarity measure to use? The current use case enters in the domain of image processing, more specifically in the image matching field, and it was addressed using transfer learning and embeddings.

To begin with, the image dataset was explored to measure the problem complexity. Later on, some research was done to find potential solutions, resulting in the finding of pre-trained models (with encoders) and embeddigns that were adapted to handle the use case. Then, the solution was tested in a research environment until getting good results. Moreover, some simulations using the expected pipeline were done in order to ensure that the solution can be implemented in production. Finally, the model/solution was deployed in production by using MLflow and FastApi.

## Requirements
* opencv
* datasets
* pytorch
* torchvision
* transformers
* mlflow
* fastapi

## The Solution

here I share a diagram of the data/ml pipeline I applied:

![alt text](https://github.com/miguelmayhem92/ImageFinder/blob/main/diagrams/ImageFinder.jpg)

Main sections:
* research environment
* jobs or production code
* API

<b> 1. Research environment

In the research envirment the goal was to explore data, measure the problem complexity and test solution end-2-end (if possible)
the first obstacles that were encountered:
  * about 290 images -> how to query as many images as possible with less effort possible? (avoid for loop)
  * big image sizes and diverse sizes -> how to preserve features when processing? how to uniform images with the less effort possible?
  * no labels -> no possibility to apply any supervised training
  * very diverse images -> even if try to label images, result would require too many labels, data and time, not an option
  * preserve the ids -> get a method that can screen the images and ,at the same time, preserve the order so that I can get the Ids of the best similar image

The solution uses an encoder model in order to infer the image embeddings. So in some words, an underlaying pre-trained model is used to get a numerical representation of the image candidates (DB) and an input image (Y). 

The embedings in short is a numerical matrix representation of the vocabulary (or image classes DB) at the last layer of the encoder block and it is calculated during the supervised training task of the model (for this use case we do not need the predictions of the model but the encoder output).

the candidates DB embeddings serve as reduced data base that will be the objects to be compared to. The embeddings of Y is the reduced version of the input image. Once both embeddings are gotten, cosine similarity is applied to get the scores of the most similar images in candidates DB, and therefore a ranking procedure can be applied to get the top x most similar embeddings and as a result the most similar images from the DB.

the following image can explain the solution:

![alt text](https://github.com/miguelmayhem92/ImageFinder/blob/main/diagrams/embedding.jpg)

the chosen underlaying model is google/vit-base-patch16-224-in21k and it is a generic model that was trained using 14 million images for more than 21.8K classes.

the chosen framework is Pytorch and tests were done to ensure that:
  * model the data pipeline -> using a dataset object for images
  * prepare train/test datasets -> train data of about 200 images and target to get 190 images as candidate embeddings
  * reproduce the model and the embeddings
  * test the model using train and test datasets
 
advantages of the model:
  * possibility to treat any kind of image size -> resizing  is part of the model pipeline
  * embeddigns allow to store a significant number of images in a matrix 
  * the dataset object allows to retrieve the Id of a given row in the embedding matrix
 
 up to here, every test, exploration and experimentation was done using Jupyter notebooks in local

<b> 2. Jobs or production code

the production code replicates the data pipeline done in the research environment so that by the end of the day the outputs are: the model, the embeddings matrix and some configs (important for the API later)

steps to execute are:

 1. image-downloader.py (provided code) it dowloads the raw images in file images/
 2. clean_csv.py it cleans the data-interview.csv (drop duplicates and some treatement) and creates data-interview-clean.csv in athe file extract_data/
 3. create_dataset.py it splits the image data in images/ into train and test data in a file dataset/ (200 images trainning data). In addition, it creates a metadata file important to retrieve Ids in the image dataset
 4. train.py
     * produces the dataset object using the train/ data
     * download the model from google/vit-base-patch16-224-in21k
     * takes a sample 190 images from the train dataset
     * calculates the embeddings using the 190 images
     * transform the data extractor to json (needed for production API)
     * store in mlflow: model and the json extractor, folder mlruns/
     * store the embeddings in embedding_db/ folder

after running the previous steps, the api has everything to run a model and to get embeddings for new images or test images

<b> 3. API

to build the API, I used FastAPI, a restful python framework, to develope the POST endpoint
the process to get the similar image is the following:

  * upload an image stored in local
  * save the image in a temporary file (so that it can be transformed in a dataset object so that torch can read it easily)
  * create the dataset object
  * call the model and the extractor json from MLflow and get the embeddings of the input image
  * call the embeddings from the train data (that is stored in embedding_db/)
  * compute cosine similarity between the input image and the embeddings db
  * sort and rank results
  * display the top 3 most similar images in json (where the key is the Id of the image and the value is the cosine similarity score)
  
the code structure is:

           .
           ├── app
           |   ├── main.py 
           |   ├── embedding_db/                            # folder where the embedding_db is stored
           |   ├── mlruns/ 
           ├── main.py                                      # location of the api executable
           ├── research_env/                                # folder that contains the notebooks where the solution was explored and tested
           ├── extract_data/                                # folder containing the csv files data-interview
           ├── images/                                      # folder containing the raw images output of the image-downloader.py
           ├── dataset/
           |   ├── train/                                   # folder containing train images
           |   ├── test/                                    # folder containing test images                            
           ├── jobs/
           |   ├── execute_ImageFinder.py                   # some functions and code that executes the image matching worflow
           |   ├── configs.py                               # configs for the api and the prod code
           |   ├── image-downloader.py                      # code 
           |   ├── clean_csv.py 
           |   ├── create_dataset.py  
           |   ├── train.py 
           ├── Dockerfile
           └── README.md

           

a demo gif:

![](https://github.com/miguelmayhem92/ImageFinder/blob/main/diagrams/demo.gif)

## Conclusions

The ImageFinder use case enters in the domain of image matching. A solution was found using a pretrained model that helps to convert an image dataset into a numerical matrix (embedding) and the same for an input image. Then cosine similarity can be computed and the embeddings image dataset can be ranked and a top x can be displayed. 
Some prerequisites needed are the model, the extractor and the embeddigns of the candidate images. Those prerequisites are necessary for the API (using FastAPI) that it can compare new images against the image candidate db (converted into a matrix or embedding)

## Some improvements for later

* in case of larger number of images to be screened (5K, 10K, 100K or even 1M) a hashmap method can be applied
* To improve precision, model finetuning using a labelized dataset with a more specific purposed will be required
* maybe implementation of a database image oriented to display the images and that can serve as scalable image database

### References

* pretrained model: https://huggingface.co/google/vit-base-patch16-224-in21k
