import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob

# Load dataset
dff = pd.read_csv('netflix_titles.csv')
print(dff.shape)
print(dff.columns)

# Distribution of Content Ratings
z = dff.groupby(['rating']).size().reset_index(name='counts')
pieChart = px.pie(z, values='counts', names='rating',
                  title='Distribution of Content Ratings on Netflix',
                  color_discrete_sequence=px.colors.qualitative.Set3)
pieChart.show()

# Top 5 Directors
dff['director'] = dff['director'].fillna('No Director Specified')
filtered_directors = dff['director'].str.split(',', expand=True).stack().to_frame()
filtered_directors.columns = ['Director']
directors = filtered_directors.groupby(['Director']).size().reset_index(name='Total Content')
directors = directors[directors.Director != 'No Director Specified']
directorsTop5 = directors.sort_values(by=['Total Content'], ascending=False).head()
directorsTop5 = directorsTop5.sort_values(by=['Total Content'])
fig1 = px.bar(directorsTop5, x='Total Content', y='Director', title='Top 5 Directors on Netflix')
fig1.show()

# Top 5 Actors
dff['cast'] = dff['cast'].fillna('No Cast Specified')
filtered_cast = dff['cast'].str.split(',', expand=True).stack().to_frame()
filtered_cast.columns = ['Actor']
actors = filtered_cast.groupby(['Actor']).size().reset_index(name='Total Content')
actors = actors[actors.Actor != 'No Cast Specified']
actorsTop5 = actors.sort_values(by=['Total Content'], ascending=False).head()
actorsTop5 = actorsTop5.sort_values(by=['Total Content'])
fig2 = px.bar(actorsTop5, x='Total Content', y='Actor', title='Top 5 Actors on Netflix')
fig2.show()

# Content Production Trend
df1 = dff[['type', 'release_year']].rename(columns={"release_year": "Release Year"})
df2 = df1.groupby(['Release Year', 'type']).size().reset_index(name='Total Content')
df2 = df2[df2['Release Year'] >= 2010]
fig3 = px.line(df2, x="Release Year", y="Total Content", color='type', title='Trend of Content Produced Over the Years')
fig3.show()

# Sentiment Analysis
dfx = dff[['release_year', 'description']].rename(columns={'release_year': 'Release Year'})
sentiments = []
for description in dfx['description']:
    if pd.isnull(description):
        sentiments.append('Neutral')
        continue
    testimonial = TextBlob(description)
    polarity = testimonial.sentiment.polarity
    if polarity > 0:
        sentiments.append('Positive')
    elif polarity < 0:
        sentiments.append('Negative')
    else:
        sentiments.append('Neutral')
dfx['Sentiment'] = sentiments
dfx = dfx.groupby(['Release Year', 'Sentiment']).size().reset_index(name='Total Content')
dfx = dfx[dfx['Release Year'] >= 2010]
fig4 = px.bar(dfx, x="Release Year", y="Total Content", color="Sentiment", title="Sentiment of Content on Netflix")
fig4.show()
