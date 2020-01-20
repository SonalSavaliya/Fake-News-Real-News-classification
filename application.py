# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from sklearn.externals import joblib

import pandas as pd
import numpy as np

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('sentiwordnet')

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from nltk.corpus import wordnet as wn

# Do this first, that'll do something eval() 
# to "materialize" the LazyCorpusLoader
next(wn.words())

#import tensorflow as tf
#global model,graph
#graph = tf.get_default_graph()

model = joblib.load('news_classification_lstm.pkl') # Loding LSTM pickle model


# Pre-processign the entered unseen data
wn = WordNetLemmatizer()
def prepare_text(x):
    
	review = re.sub('[^a-zA-Z]', ' ', str(x)) # removing sepcial characters and numbers
	review = review.lower() # lowering the text
	review = review.split() 
	# removing stopwords and lemmatization
	
	review = [wn.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
	review = ' '.join(review)
    
	MAX_NB_WORDS = 60000
	# Max number of words in each news.
	MAX_SEQUENCE_LENGTH = 500
	EMBEDDING_DIM = 100

	tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
	tokenizer.fit_on_texts([review])
	word_index = tokenizer.word_index
	#print(f'Found {len(word_index)} unique tokens.')

	X = tokenizer.texts_to_sequences([review])
	X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
		
	return X


## GUI for Textarea and Submit Button
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
	dcc.Textarea(id = 'input-1-state', value = '', style={'width': '40%', 'rows': '5'}),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='output-state')
])



@app.callback(Output('output-state', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-1-state', 'value')])
def update_output(n_clicks, input1):
	
	if input1 is not None and input1 is not '':
		try:
			clean_text = prepare_text([input1])
			#with graph.as_default():
			preds = model.predict(clean_text)
			labels = ['Fake','Real']
			return 'This news is {} and predication value: {}.'.format(labels[np.argmax(pred)], preds)
		except ValueError as e:
			print(e)
			return "Unable to classify! {}".format(e)
			

if __name__ == '__main__':
	
	app.run_server(debug=True)