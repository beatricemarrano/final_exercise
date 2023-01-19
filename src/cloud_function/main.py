import io
import pickle

#import nltk
import torch
from google.cloud import storage

#from transformers import AutoTokenizer
#from tweet_cleaner import clean_tweet

BUCKET_NAME = "final_exercise"
MODEL_FILE = "trained_model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = blob.download_as_string()
my_model = io.BytesIO(my_model)

model = torch.jit.load(my_model)


def commands(request):
    """if input= 0 print model, if input=1 print parameters"""
    request_json = request.get_json()
    if request_json['input']==0:
        print("The model is:")
        print(model)
        print('=====')
    elif request_json['input']==1:
        print("The parameters are:")
        for name, param in model.named_parameters():
            print('name: ', name)
            print(type(param))
            print('param.shape: ', param.shape)
            print('param.requires_grad: ', param.requires_grad)
            print('=====')
    else:
        print("No choice of input request received")
        print('=====')

print("init done")
print('=====')