#Import libraries 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import transformers 
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import bertopic 
import gc
from torch import cuda 
from textblob import TextBlob

#load data into pandas dataframe 
df_bjp = pd.read_csv(r'BJP4India.csv')
df_congress = pf.read_csv(r'INCIndia.csv')
df_aap = pd.read_csv(r'AamAadmiParty.csv')

# Add a 'party' column to each dataframe
df_bjp['party'] = 'BJP'
df_congress['party'] = 'Congress'
df_aap['party'] = 'AAP'

# Concatenate the dataframes
df = pd.concat([df_bjp, df_congress, df_aap])

# Convert 'Datetime' to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['Datetime'])

#view the data 
df.head()

def model_topics(df, num_topics=5, batch_size=5000):
    model = BERTopic(language="multilingual", calculate_probabilities=True)
    topics_total = []
    topic_words = {}

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]['Text']
        topics, _ = model.fit_transform(batch)
        topics_total.extend(topics)
        # For each unique topic, get the words that characterize it
        for topic in set(topics):
            if topic not in topic_words:
                topic_words[topic] = model.get_topic(topic)
        if cuda.is_available():
            cuda.empty_cache()

     # Assign the topic IDs to the DataFrame
    df['Topic'] = topics_total
    # Map the topic IDs to the corresponding words
    df['Topic_words'] = df['Topic'].map(topic_words)
    return df

model_topics(df)
df.head()

def create_word_cloud(df):
    # Join all text from the 'Text' column
    all_words = ' '.join([text for text in df['Text']])

    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

    #plot the wordcloud 
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

def run_eda(df):
    print(f"Performing EDA")

    # Print basic information about the dataset
    print(df.info())
    print(df.head())

    # Check for missing values
    print(df.isnull().sum())

    #visualize the distribution of tweets over time 
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['date'] = df['Datetime'].dt.date
    tweet_counts = df['date'].value_counts().sort_index()

    plt.figure(figsize=(12,6))
    sns.lineplot(data=tweet_counts)
    plt.title('Number of tweets over time')
    plt.xlabel('Date')
    plt.ylabel('Number of tweets')
    plt.show()

    # Generate a word cloud
    create_word_cloud(df)

# Define a function to calculate sentiment
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Calculate sentiment for each tweet
df['sentiment'] = df['Text'].apply(get_sentiment)

df.head()
run_eda(df)

df.to_csv('processed_tweets.csv', index=False)

def plot_engagement_by_topic(df, party_name):
    # Filter the dataframe for the given party
    df = df[df['Username'] == party_name]

    # If the filtered dataframe is empty, return
    if df.empty:
        print(f"No data available for {party_name}")
        return
    
    # Calculate the average likes for each topic
    topic_engagement = df.groupby('Topic')['likeCount'].mean().sort_values()

    # Plot the engagement for each topic
    plt.figure(figsize=(10,6))
    sns.barplot(x=topic_engagement.index, y=topic_engagement.values)
    plt.title(f'Average Engagement (likes) of Topics for {party_name}')
    plt.xlabel('Topic')
    plt.ylabel('Average likes')
    plt.show()

plot_engagement_by_topic(df, 'BJP4India')
plot_engagement_by_topic(df, 'INCIndia')
plot_engagement_by_topic(df, 'AamAadmiParty')

df.head()

# Ensure 'likeCount' is numeric
df['likeCount'] = pd.to_numeric(df['likeCount'], errors='coerce')

# Convert the timestamp to datetime and extract the month
df['Timestamp'] = pd.to_datetime(df['Datetime'], errors='coerce')
df['Month'] = df['Timestamp'].dt.to_period('M')

# Drop any rows with missing 'Username', 'likeCount' or 'Month'
df = df.dropna(subset=['Username', 'likeCount', 'Month'])

# Calculate the average likes for each party by month
party_monthly_engagement = df.groupby(['Username', 'Month'])['likeCount'].mean().reset_index()

# Plot the engagement for each party by month
plt.figure(figsize=(10,6))
sns.lineplot(x='Month', y='likeCount', hue='Username', data=party_monthly_engagement)
plt.title('Monthly Average Engagement (likes) by Party')
plt.xlabel('Month')
plt.ylabel('Average likes')
plt.show()

def plot_engagement_by_sentiment(df, party_name):
    # Filter the dataframe for the given party
    df = df[df['Username'] == party_name]

    # Check if the filtered df is empty
    if df.empty:
        print(f"No data for party {party_name}.")
        return

    # Define a function to categorize sentiment
    def categorize_sentiment(sentiment_score):
        if sentiment_score < -0.05:
            return 'negative'
        elif sentiment_score > 0.05:
            return 'positive'
        else:
            return 'neutral'
        
     # Apply the function to the 'sentiment' column
    df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

    # Calculate the average likes for each sentiment category
    sentiment_engagement = df.groupby('sentiment_category')['likeCount'].mean().sort_values()

    # Check if sentiment_engagement is empty
    if sentiment_engagement.empty:
        print(f"No sentiment engagement data for party {party_name}.")
        return 
    
    # Plot the engagement for each sentiment category
    plt.figure(figsize=(10,6))
    sns.barplot(x=sentiment_engagement.index, y=sentiment_engagement.values)
    plt.title(f'Average Engagement (likes) by Sentiment Category for {party_name}')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Average likes')
    plt.show()

    # Calculate sentiment for each tweet
df['sentiment'] = df['Text'].apply(get_sentiment)

# Then you can run the sentiment analysis function
plot_engagement_by_sentiment(df, 'BJP')
plot_engagement_by_sentiment(df, 'Congress')
plot_engagement_by_sentiment(df, 'AAP')

def plot_engagement_by_hour(df, party_name):
    # Filter the dataframe for the given party
    df = df[df['Username'] == party_name]

    # Extract the hour from the timestamp
    df['Hour'] = df['Timestamp'].dt.hour

    # Calculate the average likes for each hour
    hourly_engagement = df.groupby('Hour')['likeCount'].mean()

    # Plot the engagement for each hour
    plt.figure(figsize=(10,6))
    sns.lineplot(x=hourly_engagement.index, y=hourly_engagement.values)
    plt.title(f'Average Engagement (likes) by Hour for {party_name}')
    plt.xlabel('Hour')
    plt.ylabel('Average likes')
    plt.show()

plot_engagement_by_hour(df, 'BJP')
plot_engagement_by_hour(df, 'Congress')
plot_engagement_by_hour(df, 'AAP')

def plot_engagement_by_day(df, party_name):
    # Filter the dataframe for the given party
    df = df[df['Username'] == party_name]

    # Extract the day of week from the timestamp (Monday=0, Sunday=6)
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    # Calculate the average likes for each day of week
    daily_engagement = df.groupby('DayOfWeek')['likeCount'].mean()

    # Plot the engagement for each day of week
    plt.figure(figsize=(10,6))
    sns.lineplot(x=daily_engagement.index, y=daily_engagement.values)
    plt.title(f'Average Engagement (likes) by Day of Week for {party_name}')
    plt.xlabel('Day of Week')
    plt.ylabel('Average likes')
    plt.show()

plot_engagement_by_day(df, 'BJP')
plot_engagement_by_day(df, 'Congress')
plot_engagement_by_day(df, 'AAP')

def plot_engagement_by_post_length(df, party_name):
    # Filter the dataframe for the given party
    df = df[df['Username'] == party_name]

    # Calculate the length of each post
    df['PostLength'] = df['Text'].str.len()

    # Calculate the average likes for each post length
    post_length_engagement = df.groupby('PostLength')['likeCount'].mean()

    # Plot the engagement for each post length
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=post_length_engagement.index, y=post_length_engagement.values)
    plt.title(f'Average Engagement (likes) by Post Length for {party_name}')
    plt.xlabel('Post Length')
    plt.ylabel('Average likes')
    plt.show()

plot_engagement_by_post_length(df, 'BJP')
plot_engagement_by_post_length(df, 'Congress')
plot_engagement_by_post_length(df, 'AAP')