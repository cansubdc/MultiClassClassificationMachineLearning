import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from matplotlib import pyplot as plt

url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

data = pd.read_csv('file.csv', sep=',')


def cleaning_data(df):

    # REMOVE URLS
    df['text'] = df['text'].str.replace(url, '')
    # REMOVE PUNC
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    # REMOVE NUMBERS
    df['text'] = df['text'].str.replace('\d+', '')
    # LOWER CASE
    df['text'] = df['text'].str.lower()
    # REMOVE STOP WORDS
    stop = set(stopwords.words('turkish'))
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    # REMOVE SAME LINES
    df = df.drop_duplicates(subset='text')

    return df


def most_frequent_words_in_tweets(df):

    most_common = Counter(" ".join(df["text"]).lower().split()).most_common(10)

    x_label = []
    y_label = []

    for words, counts in most_common:
        print('Word:', words, '   ', 'Frequent:', counts)
        x_label.append(words)  # x axis
        y_label.append(counts)   # y axis

    plt.figure(figsize=(9,5))
    plt.plot(x_label, y_label, 'ro')
    plt.show()



