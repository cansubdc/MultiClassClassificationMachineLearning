from PreProecesssing import data
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from matplotlib import pyplot as plt

texts = data.text.str.cat(sep=' ')
tokens = word_tokenize(texts)

vocabulary = set(tokens)
print("Total Word Number: ", len(vocabulary))

frequency_dist = nltk.FreqDist(tokens)
sorted(frequency_dist, key=frequency_dist.__getitem__, reverse=True)[0:50]

word_cloud = WordCloud().generate_from_frequencies(frequency_dist)

plt.imshow(word_cloud)
plt.axis("off")
plt.show()


# NUMBER OF CATEGORY FROM ALL DATA
plt.figure(figsize=(8, 4))
data.category.value_counts().plot(kind='bar')
plt.show()







