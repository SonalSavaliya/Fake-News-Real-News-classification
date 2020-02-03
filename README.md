# Fake News Real News Analysis

I performed Fake news analysis, Real news analysis and built classification model using MultinomialNB, Logistic Regression, Random Forest and LSTM with the TF-IDF vectorizer for fake and real news. I have used two datasets for this work, [Fake News](https://www.kaggle.com/mrisdal/fake-news) and [Real News](https://www.kaggle.com/anthonyc1/gathering-real-news-for-oct-dec-2016). This data are from October to December 2016. Here, I analyzed both news types and built the classification model by combining both.

**Built a web app by deploying the LSTM model to AWS using dash and flask**: http://dash-env-dev.us-east-1.elasticbeanstalk.com/

### Fake News Analysis

- There are different fake news types(bias, conspiracy, hate, satire, etc.) in which bias, conspiracy, and hate are the most frequent
  ![Image of fake news types](https://github.com/SonalSavaliya/Fake-News-Real-News-Analysis/blob/master/Images/fake_types.PNG)
- Most of the news articles belong to US country
- In content, most frequently used words in the single word list are Trump, Clinton, People, and Election
  <p align="center"><img src="https://github.com/SonalSavaliya/Fake-News-Real-News-Analysis/blob/master/Images/fake_wordcloud.PNG" height="260"/></p>
- In the title, most frequently used words in single word list are Trump, Hillary, Clinton, Election, Russia, and Obama
- Check news types(bias, conspiracy, hate, satire, etc.) for words, Trump, Clinton, Obama, Election, Putin, and Russia 
  - Found that Trump and Clinton words appear most of the time in bias, conspiracy, and hate news types, however, Trump word appears a lot in bias news type around 138 articles and Clinton word appears a lot in conspiracy news type, around 66 articles
  - Obama, Election, Putin, Russia words appear in the bias, conspiracy, and hate news types, but in a few articles
- Most frequent bigram words: hillary clinton, donald trump, onion america, america finest, clinton campaign, etc
- Most frequent trigram words: onion america finest, dakota access pipeline, syrian report november, etc
- Top publications for fake news are presstv, greanvillepost, scott, and many others

### Real News Analysis

- In content, most frequently used words are trump, people, clinton, and president
  <p align="center"><img src="https://github.com/SonalSavaliya/Fake-News-Real-News-Analysis/blob/master/Images/real_wordcloud.PNG" height="260"/></p>
- In the title, most frequently seen words are trump, clinton, election, obama, and russia
- Most frequent bigram words: donald trump, hillary clinton, hurricane matthew, white house, and north carolina
- Most frequent trigram words: atlanic politics policy, politics policy daily, and dakota access pipeline
- Top publications: Reuters, NPR, Washington Post, Guardian, CNN, and New York Times

### Feature Engineering

- Added a few columns such as title and content length, number of capital letters of title and content, number of punctuation in title and content

### Text Pre-processing

- Removed special characters and numbers from text
- Lowering the text
- Removed stopwords
- Tokenization: convert sentences to words
- Lemmatization: An approach to remove inflection by determining the part of speech and utilizing detailed database of the language

### Classification

Compared models for classification: MultinominialNB, Logistic Regression, Random Forests and LSTM.
LSTM performs well, it has 99% accuracy rate for training set and 93% accuracy rate for testing set.
