import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\intes\Downloads\COMPROG\INTES_COMPROG_LAB7\spotify_top_1000_tracks.csv")
print(df.head())

# 1 Histogram of song duration
plt.figure()
plt.hist(df['duration_min'], bins=20)
plt.title("Distribution of Song Duration")
plt.xlabel("Duration (minutes)")
plt.ylabel("Frequency")
plt.show()

# 2 Boxplot popularity by year
plt.figure()
sns.boxplot(x=df['year'], y=df['popularity'])
plt.xticks(rotation=90)
plt.show()

# 3 Top 10 artists
top_artists = df['artist'].value_counts().head(10)
plt.figure()
top_artists.plot(kind='barh')
plt.show()

# 4 Violin plot
plt.figure()
sns.violinplot(x=df['year'], y=df['duration_ms'])
plt.xticks(rotation=90)
plt.show()

# 5 Average popularity per year
avg_pop = df.groupby('year')['popularity'].mean()
plt.figure()
avg_pop.plot()
plt.show()

# 6 Songs per year
songs_year = df['year'].value_counts().sort_index()
plt.figure()
songs_year.plot.area()
plt.show()

# 7 Scatter plot duration vs popularity
plt.figure()
plt.scatter(df['duration_ms'], df['popularity'])
plt.xlabel("Duration")
plt.ylabel("Popularity")
plt.show()

# 8 Longest songs
longest = df.sort_values('duration_ms', ascending=False).head(10)
plt.figure()
plt.stem(longest['duration_ms'])
plt.show()

# 9 Average duration by artist
avg_dur = df.groupby('artist')['duration_ms'].mean().sort_values(ascending=False).head(5)
plt.figure()
avg_dur.plot(kind='bar')
plt.show()

# 10 Top artists by decade
df['decade'] = (df['year']//10)*10
top_decade = df.groupby(['decade','artist']).size().unstack(fill_value=0)
top_decade.head(3).plot(kind='bar', stacked=True)
plt.show()

# 11 Top 10 songs popularity
top_tracks = df.sort_values('popularity', ascending=False).head(10)
plt.figure()
plt.barh(top_tracks['track_name'], top_tracks['popularity'])
plt.show()

# 12 Strip plot
plt.figure()
sns.stripplot(x=df['artist'].head(50), y=df['duration_ms'].head(50))
plt.xticks(rotation=90)
plt.show()

# 13 Pie chart top albums
top_albums = df['album'].value_counts().head(5)
plt.figure()
plt.pie(top_albums, labels=top_albums.index, autopct='%1.1f%%')
plt.show()

# 14 Heatmap
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# 15 Pairplot
sns.pairplot(df[['duration_ms','popularity','year']])
plt.show()

# 16 Bar songs per year
plt.figure()
songs_year.plot(kind='bar')
plt.show()

# 17 Swarm plot sample
sample = df.sample(100)
plt.figure()
sns.swarmplot(x=sample['year'], y=sample['popularity'])
plt.xticks(rotation=90)
plt.show()

# 18 Hexbin
plt.figure()
plt.hexbin(df['duration_ms'], df['popularity'], gridsize=25)
plt.show()

# 19 ECDF
x = np.sort(df['duration_ms'])
y = np.arange(1,len(x)+1)/len(x)
plt.figure()
plt.plot(x,y)
plt.show()

# 20 Grouped bar
grouped = df.groupby(['decade','artist'])['popularity'].mean().unstack()
grouped.head().plot(kind='bar')
plt.show()