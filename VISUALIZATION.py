import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv(r"C:\Users\Ayushi\Desktop\submissions\ADS\tiktok_dataset.csv")

# Display dataset info
print("Dataset Info:")
print(df.info())

# Display first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Histogram of video duration
plt.figure(figsize=(10, 6))
sns.histplot(df['video_duration_sec'], bins=30, kde=True)
plt.title('Histogram of Video Duration (seconds)')
plt.xlabel('Video Duration (seconds)')
plt.ylabel('Frequency')
plt.show()

# Line chart for video views (limited to first 1000 rows)
df_line_graph = df.head(1000)
plt.figure(figsize=(10, 6))
plt.plot(df_line_graph.index, df_line_graph['video_view_count'], marker='o', linestyle='-')
plt.title('Line Chart of Video Views Over Time (Limited Data)')
plt.xlabel('Index (or Time)')
plt.ylabel('Video Views')
plt.show()

# Pie chart of claim status
claim_status_counts = df['claim_status'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(claim_status_counts, labels=claim_status_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Pie Chart of Claim Status Distribution')
plt.show()

# Scatter plot of likes vs shares
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['video_like_count'], y=df['video_share_count'], hue=df['verified_status'])
plt.title('Scatter Plot of Video Likes vs Video Shares')
plt.xlabel('Video Likes')
plt.ylabel('Video Shares')
plt.show()

# Box plot of views by ban status
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['author_ban_status'], y=df['video_view_count'])
plt.title('Box Plot of Video Views by Author Ban Status')
plt.xlabel('Author Ban Status')
plt.ylabel('Video Views')
plt.show()

# Correlation matrix heatmap for numeric columns
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

# Bar plot of top 10 videos by view count
top_videos = df.nlargest(10, 'video_view_count')
plt.figure(figsize=(12, 6))
sns.barplot(y=top_videos['video_id'], x=top_videos['video_view_count'], palette="viridis")
plt.title('Top 10 Videos by View Count')
plt.xlabel('View Count')
plt.ylabel('Video ID')
plt.show()
