# Data Processing
import pandas as pd

# Embedding generation
from openai import OpenAI
import openai
import cohere
import tiktoken

# progress
from tqdm import tqdm

data = pd.read_csv("triage_original.csv", sep=',')
print(data.head())
                   
# remove rows with null values and duplicate rows
data = data.dropna()
data = data[data.duplicated()==False]

# remove outliers
data = data.drop(data.loc[data['heartrate'] > 200].index)
data = data.drop(data.loc[data['temperature'] > 106].index)
data = data.drop(data.loc[data['temperature'] < 90].index) # farenheits only
data = data.drop(data.loc[data['o2sat'] > 100].index) # assume percentage
data = data.drop(data.loc[data['sbp'] > 400].index) # impossible
data = data.drop(data.loc[data['dbp'] > 400].index) # impossible

# rebalance classes
data1 = data[data.acuity==1].sample(1029)
data2 = data[data.acuity==2].sample(1029)
data3 = data[data.acuity==3].sample(1029)
data4 = data[data.acuity==4].sample(1029)
data5 = data[data.acuity==5].sample(1029)
data_rebalanced = pd.concat([data1, data2, data3, data4, data5], ignore_index=True)

# check num tokens in any string if needed
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# use encoding cl100k_base for 3rd gen embedding models
print(num_tokens_from_string('when i pee', "cl100k_base"))

api_key = "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"
client = OpenAI(api_key=api_key)

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ") # just in case there are newlines
   return client.embeddings.create(input = [text], model=model).data[0].embedding

embeddings = []
for i in tqdm(range(len(data_rebalanced))):
    embeddings.append(get_embedding(data_rebalanced.pain[i], model='text-embedding-3-small'))
data_rebalanced.insert(9, "pain_embedding", embeddings)
print(data_rebalanced.head())

data_rebalanced.to_csv('./triage_rebalanced_pain_embeddings.csv', sep=',', index=False)