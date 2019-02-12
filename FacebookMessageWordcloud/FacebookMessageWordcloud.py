import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

df = pd.read_json('facebookmessage.json')
dict1 = pd.DataFrame.from_dict(data)
dict1 = pd.DataFrame.from_dict(data['messages'])
dict1 = pd.DataFrame.from_dict(data['messages'])

messages = dict1[dict1['type']=='generic']
messages = dict1[dict1['type']=='Generic']


mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
mpl.rcParams['font.size']=12                #10 
mpl.rcParams['savefig.dpi']=100             #72 
mpl.rcParams['figure.subplot.bottom']=.1 


stopwords = set(STOPWORDS)


wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(messages['content']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
text = " ".join(review for review in messages.content)
print ("There are {} words in the combination of all review.".format(len(text)))
text = " ".join(str(review) for review in messages.content)
print ("There are {} words in the combination of all review.".format(len(text)))
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
messages = messages[messages[['files', 'gifs', 'missed', 'photos', 'reactions','share','stickers','videos']== 'nan']]
messages = messages[messages['photos']== 'nan']
messages = dict1[dict1['type']=='Generic']
messages.dropna(thresh=2)
messages.dropnan(thresh=2)
test = messages[['photos'] == 'nan']
test = messages['photos'] == 'nan'
messages.info()
test = messages.dropna(subset = ['photos','gifs','videos'])
stopwords = set(STOPWORDS)
stopwords.update(["mother", "sara", "head", "flavor", "flavors"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
