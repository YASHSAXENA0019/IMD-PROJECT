import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

from joblib import parallel, delayed
import joblib

import streamlit as st

def clean_text(text):
    nopunc = [w for w in text if w not in string.punctuation]
    nopunc = ''.join(nopunc)
    return  ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])

def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

stemmer = PorterStemmer()
def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def text_process(review):
    nopunc = [char for char in review if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


model = joblib.load('fake.pkl')

def predict(text):
    txt = text
    txt = clean_text(txt)
    txt = preprocess(txt)
    txt = stem_words(txt)
    txt = lemmatize_words(txt)
    txt = [txt]
    if model.predict(txt)=='OR':
        return 'Original'
    else:
        return 'Fake'


def main():
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">  Fake Product Review Detection </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    review = st.text_input("Text","Type Here")
    safe_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:white;text-align:center;"> Please check the product carefully it might not be good.</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:black ;text-align:center;"> Genuine Review. </h2>
       </div>
    """

    if st.button("Predict"):
        output=predict(review)
        st.success('The review is {}'.format(output))

        if output == 'Original':
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()