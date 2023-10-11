#!/usr/bin/env python
# coding: utf-8

# In[329]:


import nltk

nltk.download([
     "names",
     "stopwords",
     "state_union",
     "twitter_samples",
     "movie_reviews",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt",
 ])


# In[330]:


nltk.download('brown')
nltk.download('stopwords')
nltk.download('punkt')


# In[331]:


from nltk.corpus import brown
brown.categories()


# In[332]:


import pandas as pd

# Loading FIFA 18 WC CSV file
fifa = pd.read_csv('FIFA 18 WC.csv')


# In[333]:


fifa.head
fifa


# In[334]:


fifa_1 = fifa.copy()


# **DATA CLEANING**

# In[335]:


#checking for ant missing values
fifa_1.isna().any()


# In[336]:


#removing NAs in the Tweet column to enable the text data to be concatenating into a string
fifa_1 = fifa_1.dropna(subset=['Tweet'])


# In[337]:


#checking duplicates from the dataset
number_of_duplicates = fifa_1.duplicated().sum()
print("Number of duplicate rows: ", number_of_duplicates)


# In[338]:


#removing duplicates from the dataset
fifa_1 = fifa_1.drop_duplicates()


# In[339]:


#Removing any punctuations from Tweet column
fifa_1['Tweet'] = fifa_1['Tweet'].str.replace("[^a-zA-Z#]", " ", regex=True)


# In[340]:


#checking for url in tweets
import numpy as np

number_of_urls = (fifa_1['Tweet'].str.contains('http')).sum()
print("Rows in the observations with URLs: ", number_of_urls)


#creating rows with url in the dataset
fifa_1['Url'] = np.where(fifa_1['Tweet'].str.contains('http'), 1, 0)


# In[341]:


#checking for url in tweets
import numpy as np

number_of_urls = (fifa_1['Tweet'].str.contains('http')).sum()
print("Rows in the observations with URLs: ", number_of_urls)


#creating rows with url in the dataset
fifa_1['Url'] = np.where(fifa_1['Tweet'].str.contains('http'), 1, 0)


# In[342]:


#converting words in the Tweet to lowercase
fifa_1['Tweet'] = fifa_1['Tweet'].str.lower()
print(fifa_1['Tweet'])


# # **STATISTICAL ANALYSIS**

# In[343]:


#converting the dates in the Date column to be in the same format
fifa_1['Date'] = pd.to_datetime(fifa_1['Date'])


# grouping the dataset DataFrame by date and calculate the mean of Likes and RTs for each date
average_likes_AND_rts = fifa_1.groupby('Date')[['Likes', 'RTs']].mean()

print(average_likes_AND_rts)


# In[344]:


#checking the frquency of tweets by date
frequency_of_tweet = fifa_1['Date'].value_counts().sort_index()
print(frequency_of_tweet)


# In[345]:


import matplotlib.pyplot as plt

# Counting daily tweets and sorting by date 
frequency_of_tweet = fifa_1['Date'].value_counts().sort_index()

# Ploting the frequency of tweets on a histogram to see its distribution
plt.hist(frequency_of_tweet.values, bins=10)

# labelling x and y axes labels and that of the title
plt.xlabel('Number of Tweets')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Tweets')

#Displaying the plot
plt.show()


# In[346]:


# Converting the date column to a pandas DatetimeIndex to extract the timeframe from the date
hour = pd.DatetimeIndex(fifa_1['Date']).hour

# Categorizing the hours into sunrise, sunset, and night
bins = [0, 12, 15, 23]
labels = ['sunrise', 'sunset', 'night']
day_time = pd.cut(hour, bins=bins, labels=labels)
fifa_1 ['Time_of_day'] = day_time
fifa_1 ['Hour_of_day'] = hour


# In[347]:


# Creating a DataFrame that computes the sum total of Likes and RTs happening daily
Total_Likes_AND_RTs = fifa_1.groupby(['Date'])[['Likes', 'RTs']].sum().reset_index()

#scatter plot of Likes vs RTs
plt.scatter(Total_Likes_AND_RTs['Likes'], Total_Likes_AND_RTs['RTs'])
plt.title('Likes vs. RTs')
plt.xlabel('Likes')
plt.ylabel('RTs')
plt.show()


# In[348]:


# Count the number of tweets per day and sort by date
frequency_of_tweet = fifa_1['Date'].value_counts().sort_index()

# Plot the frequency of tweets as a line graph
plt.plot(frequency_of_tweet.index, frequency_of_tweet.values)

# Add x-axis and y-axis labels and a title
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.title('Frequency of Tweets')

# Customize the x-axis tick labels
plt.xticks(rotation=45)

# Display the plot
plt.show()


# In[349]:


sum_likes_hourly = fifa_1.groupby('Hour_of_day')['Tweet'].count().plot.bar()

# Add x-axis and y-axis labels and a title
plt.xlabel('Hour of Day')
plt.ylabel('Number of Tweets')
plt.title('Frequency of Tweets')

# labelling the x-axis
plt.xticks(rotation=360)

# Displaying the plot
plt.show()


# In[350]:


sum_tweets_time = fifa_1.groupby('Time_of_day')['Tweet'].count().plot.bar()

# Add x-axis and y-axis labels and a title
plt.xlabel('Time of Day')
plt.ylabel('Number of Tweets')
plt.title('Frequency of Tweets')

# Customize the x-axis tick labels
plt.xticks(rotation=360)

# Display the plot
plt.show()


# # **DATA MININIG EXPLORATION**
# 
# ### **1. Tokenizing and removing Stop words**

# In[351]:


import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
fifa_1['Tweet'] = fifa_1['Tweet'].apply(lambda x: word_tokenize(x))


# In[352]:


def tokenize_tweet(tweet):
    return nltk.word_tokenize(tweet)

fifa_1['Tweet'] = fifa_1['Tweet'].astype(str)
fifa_1['Tweet'] = fifa_1['Tweet'].apply(lambda x: nltk.word_tokenize(x))
fifa_1['Tweet'] = fifa_1['Tweet'].astype(str)


# In[353]:


## removing stopwords
import nltk
import pandas as pd
nltk.download('stopwords') # Downloading the NLTK stopwords


# In[354]:


from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
mus_stop = stopwords.words("english")
mus_stopwords = nltk.corpus.stopwords.words('english') #Defining lists of stopwords


# In[355]:


# Defining function to remove stopwords in tokens' list
def remove_stopwords(tokens):
       list_of_stopwords = stopwords.words('english')
       return[token for token in tokens if token.lower() not in list_of_stopwords] 


# In[356]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
mus_stop = stopwords.words('english')

#increasing the stopwords list to incoporate other stopwords
mus_stopwords = ['https', 'I', 'That','This','There','amp', 'It']
for i in mus_stopwords:
    mus_stop.append(i)
print(mus_stop)

#applying it into the Tweet column of fifa_1
fifa_1["Tweet"] = fifa_1["Tweet"].apply(lambda x: ' '.join([word for word in x.split() if word not in (mus_stop)]))


# In[357]:


# removing texts less than 7 characters
# Defining function for removing shortwords from a text
def delete_shortwords(text):
    words = text.split()
    filtered_words = [word for word in words if len(word) > 7]
    filtered_text = " ".join(filtered_words)
    return filtered_text

fifa_1['Tweet'] = fifa_1['Tweet'].apply(delete_shortwords)


# In[358]:


fifa_1['length_of_tweet'] = fifa_1['Tweet'].apply(len)


# ### **2. Wordcloud**

# In[359]:


get_ipython().system('pip install wordcloud')

from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
wordcloud = WordCloud(width=600, height=400, random_state=2, max_font_size=100).generate(' '.join(fifa_1['Tweet']))


# In[360]:


import plotly.express as px
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# ### **3. Sentiment Analysis**

# In[361]:


get_ipython().system('pip install textblob')

nltk.download('wordnet')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob





# In[362]:


# calculating and assigning polarity to the clean_Tweet in the fifa_1 data 
fifa_1['polarity'] = fifa_1['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)

#creating a sentiment column and assigning polarity values to positive > 0, negative < 0 and neutral=0 tweets
fifa_1['sentiment'] = fifa_1['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))


# In[363]:


# Counting and creating the dataframe for the negative, neutral and positive sentiments 
number_of_sentiment_counts = fifa_1.groupby(['sentiment'])['Tweet'].count().reset_index()
print(number_of_sentiment_counts)


# In[364]:


# Bar chart showing sentiments at the FIFA 18 World Cup
sns.set_style('whitegrid')
plt.bar(number_of_sentiment_counts['sentiment'], number_of_sentiment_counts['Tweet'], color=['magenta', 'black', 'blue'])
plt.title('Sentiment Analysis of FIFA 18 World Cup Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# ### **Top Five Tweets**

# In[365]:


# Sort the DataFrame by number of likes and select the top 5 tweets
top_five_tweets = fifa_1.sort_values("Likes", ascending=False).head(5)

for i, tweet in top_five_tweets.iterrows():
    print(f"{tweet['Tweet']}\nLikes: {tweet['Likes']}\n")


# In[366]:


top_five_tweets['polarity'] = top_five_tweets['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
top_five_tweets['sentiment'] = top_five_tweets['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))


# In[367]:


# Create a DataFrame with the counts of positive, negative, and neutral tweets
senti_Anal_count = top_five_tweets.groupby(['sentiment'])['Tweet'].count().reset_index()


# In[368]:


# plotting a bar chart for the outputted sentiment analysis
sns.set_style('whitegrid')
plt.bar(senti_Anal_count['sentiment'], senti_Anal_count['Tweet'], color=['magenta', 'black', 'blue'])
plt.title('Top Five Tweets With Most Likes')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# In[369]:


from textblob import TextBlob

# creating the function that will calculate sentiment polarity of a text.
def mus_sentiment_score(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# inputting the derived function to the fifa_1 dataframe
fifa_1['Sentiment_score'] = fifa_1['Tweet'].apply(mus_sentiment_score)


# In[370]:


display(fifa_1)


# # **MACHINE LEARNING MODELS**
# 
# ### 1. Linear Regression

# In[371]:


# setting the seed so that it can be reproducable
import random
random.seed(1845)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


input_variable = fifa_1[['Url','RTs','polarity','Hour_of_day','length_of_tweet','Sentiment_score']]
target_variable = fifa_1[['Likes']]


# Split the data into training and testing sets
input_variable_train, input_variable_test, target_variable_train, target_variable_test = train_test_split(input_variable, target_variable, test_size=0.20)


# Train the model
model = LinearRegression()
model.fit(input_variable, target_variable)


# Make predictions on the testing set
y_pred = model.predict(input_variable_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(target_variable_test, y_pred)
print("Mean Squared Error:", mse)

# Mean Absolute Error
mae = mean_absolute_error(target_variable_test, y_pred)
print("Mean Absolute Error:", mae)

# Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)


# ### 2. Decision Tree

# In[372]:


# # Decision Tree Model

import pandas as pd
from sklearn.tree import DecisionTreeRegressor

a = input_variable
b = target_variable
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.20, random_state=40)



# Train the decision tree model
decT_model = DecisionTreeRegressor(random_state=40)
decT_model.fit(a_train, b_train)


# Make predictions on the testing set
y_pred = decT_model.predict(a_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(b_test, y_pred)
print("Mean Squared Error:", mse)

# Mean Absolute Error
mae = mean_absolute_error(target_variable_test, y_pred)
print("Mean Absolute Error:", mae)

# Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)


# ### 3. Random Forest

# In[373]:


# # Random Forest Model

from sklearn.ensemble import RandomForestRegressor

# Split the data into training and testing sets
a = input_variable
b = target_variable
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.20, random_state=40)

# Train the Random Forest model
ranF_model = RandomForestRegressor(random_state=40)
ranF_model.fit(a_train, b_train)



# Make predictions on the testing set
y_pred = ranF_model.predict(a_test)


# Evaluate the model using Mean Squared Error
mse = mean_squared_error(b_test, y_pred)
print("Mean Squared Error:", mse)

# Mean Absolute Error
mae = mean_absolute_error(target_variable_test, y_pred)
print("Mean Absolute Error:", mae)

# Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)


# In[ ]:




