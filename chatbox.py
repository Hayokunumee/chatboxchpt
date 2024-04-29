import nltk
import random 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
#import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

#Download required NLTK data 
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

data = pd.read_csv('iPhone Dialog.txt', sep = ':', header = None)


cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

cust  = cust[1].reset_index(drop = True)
sales = sales[1].reset_index(drop = True)

new_data = pd.DataFrame()
new_data ['Question'] = cust
new_data ['Answer'] = sales


# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)
new_data['tokenized Questions'] = new_data['Question'].apply(preprocess_text)


xtrain = new_data['tokenized Questions'].to_list()


# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)

#----------------Streamlit Implementation-----------------------

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: geneva'>TOYIN'S HOUSE OF GADGET</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By Oluwatoyin Olusoga</h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)


user_hist = []
reply_hist = []

robot_image, space1, space2, chats = st.columns(4)
with robot_image: 
    robot_image.image('pngwing.com phones.png', width = 400)

with chats:
    st.image('pngwing.com bot2.png')
    user_message = chats.text_input('Hello there you can ask your questions: ')
    def responder(user_input):
        user_input_processed = preprocess_text(user_input)
        vectorized_user_input = tfidf_vectorizer.transform([user_input_processed])
        similarity_score = cosine_similarity(vectorized_user_input, corpus)
        argument_maximum = similarity_score.argmax()
        return new_data['Answer'].iloc[ argument_maximum]

bot_greetings = ['Hello user, I am Ayo..... Pls ask your question',
             'Howdy, what can I do for you?',
             'Whazzap, what you need?',
             'Oremi, ki lo need?',
             'Welcome user, what can I do you for?']

bot_farewell = ['Thanks for your usage..... Bye.',
            'Tainku, come again soon.',
            'Gracias, hope to see you again.',
            'Oshey oremi...... Odabo.',
            'Bye user, thanks for your patronage.']

human_greetings = ['hi', 'hello', 'hey', 'whatsup', 'good day', 'hello there', 'howdy', 'waddup']

human_exits = ['bye', 'good bye', 'ciao', 'close', 'exit', 'thanks', 'thank you','close']

import random
random_greeting = random.choice(bot_greetings)
random_farewell = random.choice(bot_farewell)

if user_message.lower() in human_exits:
    chats.write(f"\nChatbot: {random_farewell}!")
    user_hist.append(user_message)
    reply_hist.append(random_farewell)

elif user_message.lower() in human_greetings:
    chats.write(f"\nChatbot: {random_greeting}!")
    user_hist.append(user_message)
    reply_hist.append(random_greeting)

elif user_message == '':
    chats.write('')

else:
    response = responder(user_message)
    chats.write(f"\nChatbot: {response}!")
    user_hist.append(user_message)
    reply_hist.append(response)

    #save the history of user texts
import csv
with open('history.txt', 'a') as file:
    for item in user_hist:
        file.write(str(item) + '\n')

#save history of bot reply
with open('reply.txt', 'a') as file:
    for item in reply_hist:
        file.write(str(item) + '\n')

#import the file to display it in the frontend 
with open('history.txt') as f:
    reader = csv.reader(f)
    data1 = list(reader)

with open('reply.txt') as f:
    reader = csv.reader(f)
    data2 = list(reader)

data1 = pd.Series(data1)
data2 = pd.Series(data2)

history = pd.DataFrame({'User Input': data1, 'Bot Reply': data2})

#history = pdSeries(data)
st.subheader('Chat History', divider = True)
st.dataframe(history, use_container_width = True)

#st.sidebar.write(data2)
st.sidebar.image('pngwing.com gadgets.png')

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.divider()
st.sidebar.header('Phones & Gadgets')

df = pd.read_csv('Mobile-Phones.csv')

st.sidebar.write(df)
st.dataframe(data, use_container_width = True)