# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
#from sklearn.externals import joblib

import pandas as pd
import numpy as np

import nltk
nltk.download('wordnet')
nltk.download('stopwords')

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from nltk.corpus import wordnet as wn
from keras.models import load_model

# Do this first, that'll do something eval() 
# to "materialize" the LazyCorpusLoader
next(wn.words())

from tensorflow import keras
import tensorflow as tf

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)


config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)
keras.backend.set_session(session)

#model = joblib.load('news_classification_lstm.pkl')
model = load_model('news_classification_lstm.h5') # Loding LSTM pickle model

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
	html.Div([
	html.H3("Fake News Detection App"	),
	html.H5("Enter an article in the below textarea!"),
	dcc.Textarea(id = 'input-1-state', value = '', style={'width': '100%', 'height': '200px'}),
    html.Button(id='submit-button', n_clicks=0, children='Submit', 
				style={"margin-left":"10px", "margin-bottom":"4px", 
						"position": "relative", "vertical-align": "text-bottom"}),
						
    html.Div(id='output-state'),
	], style={"width": "40%", "text-align": "center", "margin": "auto"}),
	
	dcc.Markdown('''
__Sonal Savaliya__

[LinkedIn](https://www.linkedin.com/in/sonal-savaliya/) | [GitHub](https://github.com/SonalSavaliya)
''', style={"text-align":"right"})
	
], style={"text-align":"center"})





@app.callback(Output('output-state', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-1-state', 'value')])
def update_output(n_clicks, input1):
	input1 = input1.lstrip() # Removing spaces and enter from the begining and end, incase someone only enters spaces or enters
	if input1 is not None and input1 is not '':
		try:
			with session.as_default():
				with session.graph.as_default():
					clean_text = prepare_text([input1])
					#with graph.as_default():
					preds = model.predict(clean_text)
					labels = ['Fake','Real']
					return 'This news is {}'.format(labels[np.argmax(preds[0])])
		except ValueError as e:
			print(e)
			return "Unable to classify! {}".format(e)
	
			

if __name__ == '__main__':
	
	app.run_server(debug=True)