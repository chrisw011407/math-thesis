import pandas as pd
import os
import random
from PIL import Image
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')
# from nltk.tokenize import word_tokenize

def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """
    Generates a color function where:
    - Largest words are deep, vibrant red (low lightness, high saturation).
    - Smaller words are slightly more magenta (higher hue).
    - Smaller words are lighter, and larger words are darker.
    """
    # Hue adjustment: Smaller words are more magenta (~330°), larger words are red (~0°)
    hue = max(0, min(330, 330 - font_size * 0.05))  # Shifts smaller words towards magenta

    # Saturation: Larger font size → Higher saturation (max 100%)
    saturation = min(100, max(30, font_size * 2))  # Ensures a smooth scale

    # Lightness: Smaller words are lighter, larger words are darker
    lightness = min(85, max(30, 100 - font_size // 2))  # Keeps readability
    
    return "hsl(%d, %d%%, %d%%)" % (hue, saturation, lightness)

def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # Hue adjustment: Smaller words are more magenta (~330°), larger words are red (~0°)
    hue = max(0, min(120, 120 - font_size * 0.05))  # Shifts smaller words towards magenta

    # Saturation: Larger font size → Higher saturation (max 100%)
    saturation = min(100, max(30, font_size * 2))  # Ensures a smooth scale

    # Lightness: Smaller words are lighter, larger words are darker
    lightness = min(85, max(30, 100 - font_size // 2))  # Keeps readability
    
    return "hsl(%d, %d%%, %d%%)" % (hue, saturation, lightness)

def blue_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    # Hue adjustment: Smaller words are more magenta (~330°), larger words are red (~0°)
    hue = max(0, min(250, 250 - font_size * 0.05))  # Shifts smaller words towards magenta

    # Saturation: Larger font size → Higher saturation (max 100%)
    saturation = min(100, max(30, font_size * 2))  # Ensures a smooth scale

    # Lightness: Smaller words are lighter, larger words are darker
    lightness = min(85, max(30, 100 - font_size // 2))  # Keeps readability
    
    return "hsl(%d, %d%%, %d%%)" % (hue, saturation, lightness)


# Load data
data = pd.read_csv('P70 responses.csv', encoding='latin1')
hw1 = data[['Graduating Year (1)', 'Response (1)']].sort_values('Graduating Year (1)', ascending=False).dropna()
hw2 = data[['Graduating Year (2)', 'Response (2)']].sort_values('Graduating Year (2)', ascending=False).dropna()
hw3 = data[['Graduating Year (3)', 'Response (3)']].sort_values('Graduating Year (3)', ascending=False).dropna()

hw1 = hw1.drop(hw1[hw1['Graduating Year (1)'] <= 2026].index).reset_index(drop=True)
hw2 = hw2.drop(hw2[hw2['Graduating Year (2)'] <= 2026].index).reset_index(drop=True)
hw3 = hw3.drop(hw3[hw3['Graduating Year (3)'] <= 2026].index).reset_index(drop=True)

print(hw3)
stopwords = stopwords.words('english')
custom_word_list = ['learned', 'important', 'concept', 'really', 'thing', 'seen', 'liked', 'also', 'helps', 'made', 'helped', 'x', 'understand', 'helpful']
stopwords.extend(custom_word_list)
stop_words = set(stopwords) # Stop words

cloud_words = ''

for i in range(len(hw3)):
    response = hw3['Response (3)'][i]

    # Tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(response)

    filtered_sentence = [w for w in tokens if not w.lower() in stop_words]
    cloud_words += ' '.join(filtered_sentence) + ' '

# print(cloud_words)

wc = WordCloud(width = 4000, height = 3200,
                background_color ='white',
                stopwords = stop_words,
                min_font_size = 10,
                color_func=blue_color_func).generate(cloud_words)


plt.figure(figsize = (10, 8), facecolor = None)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.savefig('wordcloud_hw3.png',dpi=500)
plt.show()
