import os
import numpy as np
import pandas as pd
import ktrain
from sklearn.datasets import fetch_20newsgroups

#data preprocessing or data vectorization

# Define types of text to remove
remove={'headers','footers','quotes'}

# Fetching training data
train=fetch_20newsgroups(subset='train',remove=remove)
test=fetch_20newsgroups(subset='test',remove=remove)
texts=train.data+test.data

# Initializing topic model
tm=ktrain.text.get_topic_model(texts,n_features=10000)

# Building topic model
tm.build(texts,threshold=0.25)

#data training
tm.train_recommender()
rawtext="""NASA is set for a groundbreaking leap in space exploration with the upcoming deployment of its Advanced Composite Solar Sail System (ACS3). Scheduled for launch in April aboard Rocket Lab’s Electron rocket from New Zealand, this innovative mission aims to tap into the power of sunlight for propulsion."""
 # Generating recommendations based on input text
tm.recommend(text=rawtext,n=5)

# Printing recommendations
for i, doc in enumerate(tm.recommend(text=rawtext,n=5)):
        print('Result a%s'%(i+1))
        print('Text \n')
        print(" ".join(doc['text'].split()[:500]))
        print()
