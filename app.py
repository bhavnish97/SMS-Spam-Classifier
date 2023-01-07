import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')

ps = PorterStemmer()

def text_preprocess(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  new_text = []
  for i in text:
    if i.isalnum():
      new_text.append(i)
    
  text = new_text.copy()
  new_text.clear()
  
  for i in text:
    if (i not in stopwords.words('english')) and (i not in string.punctuation):
      new_text.append(i)
  
  text = new_text.copy()
  new_text.clear()

  for i in text:
    new_text.append(ps.stem(i))

  return " ".join(new_text)



tfidf = pickle.load(open('vectorizer.pkl','rb'))
#text_preprocess = pickle.load(open('preprocessing.pkl', 'rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("SMS Spam Classifier")



input_sms = st.text_area("Enter the message to check")

# Preprocessing
transform_sms = text_preprocess(input_sms)

# Vectorize
vector_input = tfidf.transform([transform_sms])

# Model prediction

prediction = model.predict(vector_input)[0]

if prediction == 1:
    st.header("Spam")
else:
    st.header("Not Spam")