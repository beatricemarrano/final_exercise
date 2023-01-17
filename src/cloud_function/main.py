#import io
import pickle
#import nltk
import torch
from google.cloud import storage
#from transformers import AutoTokenizer
#from tweet_cleaner import clean_tweet

BUCKET_NAME = "final_exercise"
MODEL_FILE = "deployable_model.pt"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = blob.download_as_string()
my_model = pickle.load(my_model)#



def predict(request):
   """ will to stuff to your request """
   request_json = request.get_json()
   if request_json and 'input_data' in request_json:
         data = request_json['input_data']
         input_data = list(map(int, data.split(',')))
         prediction = my_model.predict([input_data])
         return f'Belongs to class: {prediction}'
   else:
         return 'No input data received'


print("init done")