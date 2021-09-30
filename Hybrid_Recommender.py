##########################################
# Hybrid Recommender System
###########################################
# Business Problem: Make an estimate for the user whose ID is given, using the item-based and user-based recommender methods.

# Step 1: Preparing the Dataset
# Step 2: Determining the Movies Watched by the User
# Step 3: Accessing Data and Ids of Other Users Watching the Same Movies
# Step 4: Identifying Users with the Most Similar Behaviors to the User to Suggest
# Step 5: Calculating the Weighted Average Recommendation Score
# Step 6: Making an item-based suggestion based on the name of the movie that the user has watched with the highest score.


#############################################
# Step 1: Preparing the Dataset
#############################################
import pandas as pd
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)

movie = pd.read_csv('Datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Datasets/movie_lens_dataset/rating.csv')

df = movie.merge(rating, how="left", on="movieId")
# Selection of movies with over 1000 comments
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]

#New comments count
common_movies.shape
# New unique movie count
common_movies["title"].nunique()
common_movies.head()

#Creating user-movie df
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()

random_user = int(pd.Series(user_movie_df.index).sample(1).values)
#78712

#############################################
# Step 2: Determining the Movies Watched by the User Who We Will Make a Suggestion
#############################################

random_user_df = user_movie_df[user_movie_df.index == random_user]
#We have reduced the dataset to user now we will go to the non na to go to the movies he watched
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)


#############################################
# Step 3: Accessing Data and Ids of Other Users Watching the Same Movies
#############################################


movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.shape

#We are creating a variable called user movie count, and by saying this, we get how many movies each user has watched.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count.columns = ["userId","movie_count"]

#Fixing index problem
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId","movie_count"]

#I want to bring users who watch at least 65% of the same movies as our user.
percentage = len(movies_watched) * 65 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > percentage]["userId"]
users_same_movies.head()
users_same_movies.count()

#############################################
# Step 4: Identifying Users with the Most Similar Behaviors to the User to Suggest
#############################################
# For this we will perform 3 steps:
#1. We will aggregate the data of our user and other users.
#2. We will create the correlation df.
# 3. We will find the most similar users (Top Users)

#We concat random_user and other users side by side with concat
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

final_df.head()
#We get users in columns to look by users
final_df.T.corr()

#For naming and index problems that will arise
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()
# We got the users in the columns and we got the movies in the rows
# We looked at the correlations and calculated the correlation of each user with each user
corr_df.head()
# We'll find users with a 65 percent or higher correlation with random_user:
#We selected random_user from the user_id_1 column in #corrdf.
# And since we will bring the ones with a correlation equal or greater than 65% and then look at the similarity with random_user,
# Since we have no more work with userid1, we select the remaining two columns by saying reset index and create top users


top_user = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2" , "corr"]].reset_index(drop=True)

top_users = top_user.sort_values("corr", ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

#We found the most similar users with User,
# but there is no information about which movie they rate, so we bring the rating table

rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")
top_users_ratings.head()
#removing the user from here
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

#############################################
# Step 5: Calculating the Weighted Average Recommendation Score
#############################################
#We will create a single score by considering the effects of the users most similar to user and the points given at the same time.
top_users_ratings['weighted_rating'] = top_users_ratings["corr"] * top_users_ratings["rating"]
top_users_ratings.head()

#When we do #groupby, the weighted average of each movie will come
top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
#assign to variable
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
#Number of unique movie
recommendation_df[["movieId"]].nunique()
#max point
recommendation_df["weighted_rating"].max()

#Scaling according to Max point
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 5))
scaler.fit(recommendation_df[["weighted_rating"]])
recommendation_df["weighted_rating"] = scaler.transform(recommendation_df[["weighted_rating"]])

#We bring movies he can like
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 4.5].sort_values("weighted_rating", ascending=False)
movies_to_be_recommend.head(20)
user_based_movies = movies_to_be_recommend.merge(movie[["movieId", "title"]]).head()

#########################
## Step 6: Making an item-based suggestion based on the name of the movie that the user has watched with the highest score.
#########################

#USER BASED RECOMMEDATION
"""
   movieId  weighted_rating                                title
0     5613         5.000000                       8 Women (2002)
1     6874         5.000000             Kill Bill: Vol. 1 (2003)
2    38499         5.000000             Angels in America (2003)
3       36         5.000000              Dead Man Walking (1995)
4     5878         4.943272  Talk to Her (Hable con Ella) (2002)

"""

#ITEM-BASED-RECOMMENDED FILMS
movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)
["movieId"][0:1].values[0]

movie[550:555]
movie_name = "True Romance (1993)"
movie_name = user_movie_df[movie_name]
item_based_movies = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head()
item_based_movies = item_based_movies.reset_index()
item_based_movies.columns = ["title", "corr"]


"""
Exit Through the Gift Shop (2010)                      
Kalifornia (1993)                                       
Fighter, The (2010)                                     
Black Cat, White Cat (Crna macka, beli macor) (1998)    
Killing Zoe (1994)                                  
"""

hybrid_Recommend = pd.concat([user_based_movies["title"], item_based_movies["title"]], ignore_index=True)