# Travel-Review-Ratings
import numpy as np
import pandas as pd 
import os
data=pd.read_csv(r"C:\Users\kusha\OneDrive\Documents\google_review_ratings (1).csv")
data
data.info()
column_names = ['user_id', 'churches', 'resorts', 'beaches',
                'parks', 'theatres', 'museums', 'malls', 'zoo',
                'restaurants', 'pubs_bars', 'local_services',
                'burger_pizza_shops', 'hotels_other_lodgings',
                'juice_bars', 'art_galleries', 'dance_clubs',
                 'swimming_pools', 'gyms', 'bakeries', 'beauty_spas',
                'cafes', 'view_points', 'monuments', 'gardens', 'Unnamed: 25']

data.columns = column_names
data.isnull().sum()
data.shape
data.columns
data.head(100)
data.info()
data.drop('Unnamed: 25', axis = 1, inplace = True)
column_names = ['user_id', 'churches', 'resorts', 'beaches', 'parks', 'theatres', 'museums', 'malls', 'zoo', 'restaurants', 'pubs_bars', 'local_services', 'burger_pizza_shops', 'hotels_other_lodgings', 'juice_bars', 'art_galleries', 'dance_clubs', 'swimming_pools', 'gyms', 'bakeries', 'beauty_spas', 'cafes', 'view_points', 'monuments', 'gardens']
data.columns = column_names
data[column_names].isnull().sum()
data =data.fillna(0)
data.dtypes
data['local_services'][data['local_services'] == '2\t2.']
local_services_mean = data['local_services'][data['local_services'] != '2\t2.']
data['local_services'][data['local_services'] == '2\t2.'] = np.mean(local_services_mean.astype('float64'))
data['local_services'] = data['local_services'].astype('float64')
data.dtypes
data[column_names[:12]].describe()
data[column_names[12:]].describe()
data_description = data.describe()
min_val = data_description.loc['min'] > 0
min_val[min_val]
import matplotlib.pyplot as plt
import numpy as np
plt.rcdefaults()
%matplotlib inline
no_of_zeros = data[column_names[1:]].astype(bool).sum(axis=0).sort_values()

plt.figure(figsize=(20,10))
plt.barh(np.arange(len(column_names[1:])), no_of_zeros.values, align='center', alpha=1.0)
plt.yticks(np.arange(len(column_names[1:])), no_of_zeros.index)
plt.xlabel('No of reviews')
plt.ylabel('Categories')
plt.title('No of reviews under each category')
no_of_reviews = data[column_names[1:]].astype(bool).sum(axis=1).value_counts()
plt.figure(figsize=(20,10))
plt.bar(np.arange(len(no_of_reviews)), no_of_reviews.values, align='center', alpha=1.0)
plt.xticks(np.arange(len(no_of_reviews)), no_of_reviews.index)
plt.ylabel('No of reviews')
plt.xlabel('No of categories')
plt.title('No of Categories vs No of reviews')
avg_rating = data[column_names[1:]].mean()
avg_rating = avg_rating.sort_values()
plt.figure(figsize=(20,10))
plt.barh(np.arange(len(column_names[1:])), avg_rating.values, align='center', alpha=1.0)
plt.yticks(np.arange(len(column_names[1:])), avg_rating.index)
plt.xlabel('Average Rating')
plt.title('Average rating per Category')
entertainment = ['theatres', 'dance_clubs', 'malls']
food_travel = ['restaurants', 'pubs_bars', 'burger_pizza_shops', 'juice_bars', 'bakeries', 'cafes']
places_of_stay = ['hotels_other_lodgings', 'resorts']
historical = ['churches', 'museums', 'art_galleries', 'monuments']
nature = ['beaches', 'parks', 'zoo', 'view_points', 'gardens']
services = ['local_services', 'swimming_pools', 'gyms', 'beauty_spas']
df_category_reviews = pd.DataFrame(columns = ['entertainment', 'food_travel', 'places_of_stay', 'historical', 'nature', 'services'])
df_category_reviews['entertainment'] = data[entertainment].mean(axis = 1)
df_category_reviews['food_travel'] = data[food_travel].mean(axis = 1)
df_category_reviews['places_of_stay'] = data[places_of_stay].mean(axis = 1)
df_category_reviews['historical'] = data[historical].mean(axis = 1)
df_category_reviews['nature'] = data[nature].mean(axis = 2)
df_category_reviews['services'] = data[services].mean(axis = 2)
df_category_reviews.describe(
ratings_per_category_df = pd.DataFrame(data[column_names[1:]].mean()).reset_index(level=0)
ratings_per_category_df.columns = ['category', 'avg_rating']
ratings_per_category_df['no_of_ratings'] = data[column_names[1:]].astype(bool).sum(axis=0).values.tolist()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
ratings_per_category_df['avg_rating_scaled'] = scaler.fit_transform(ratings_per_category_df['avg_rating'].values.reshape(-1,1))
ratings_per_category_df['no_of_ratings_scaled'] = scaler.fit_transform(ratings_per_category_df['no_of_ratings'].values.reshape(-1,1))
def calculate_weighted_rating(x):
    return (0.5 * x['avg_rating_scaled'] + 0.5 * x['no_of_ratings_scaled'])

ratings_per_category_df['weighted_rating'] = ratings_per_category_df.apply(calculate_weighted_rating, axis = 1)
ratings_per_category_df = ratings_per_category_df.sort_values(by=['weighted_rating'], ascending = False)
data.head()
def get_recommendation_based_on_popularity(x):
    zero_cols = data[data['user_id'] == x['user_id']][column_names[1:]].astype(bool).sum(axis=0)
    zero_df = pd.DataFrame(zero_cols[zero_cols == 0]).reset_index(level = 0)
    zero_df.columns = ['category', 'rating']
    zero_df = pd.merge(zero_df, ratings_per_category_df, on = 'category', how = 'left')[['category', 'weighted_rating']]
    zero_df = zero_df.sort_values(by = ['weighted_rating'], ascending = False)
    if len(zero_df) > 0:
        return zero_df['category'].values[0]
    else:
        return ""
        data_recommendation = data.copy()
data_recommendation['recommendation_based_on_popularity'] = data_recommendation.apply(get_recommendation_based_on_popularity, axis = 1)
data_recommendation['recommendation_based_on_popularity'][data['user_id'] == "User 16"]
from sklearn.neighbors import NearestNeighbors
data_matrix = data[column_names[1:]].values
knn_model = NearestNeighbors(n_neighbors=5).fit(data_matrix)
query_index = np.random.choice(data[column_names[1:]].shape[0])
distances, indices = knn_model.kneighbors(data[column_names[1:]].iloc[query_index, :].values.reshape(1,-1), n_neighbors = 5)
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
array = scaler.fit_transform(data[column_names[1:]].values)
ratings_per_category_df['no_of_ratings_scaled'] = scaler.fit_transform(ratings_per_category_df['no_of_ratings'].values.reshape(-1,1))
input_array = data[column_names[1:]].values
kmeans = KMeans(n_clusters=6)
# fit kmeans object to data
kmeans.fit(input_array)
# print location of clusters learned by kmeans object
print(kmeans.cluster_centers_)
# save new clusters for chart
y_km = kmeans.fit_predict(input_array)
plt.scatter(input_array[y_km ==0,0], input_array[y_km == 0,1], s=100, c='red')
plt.scatter(input_array[y_km ==1,0], input_array[y_km == 1,1], s=100, c='black')
plt.scatter(input_array[y_km ==2,0], input_array[y_km == 2,1], s=100, c='blue')
plt.scatter(input_array[y_km ==3,0], input_array[y_km == 3,1], s=100, c='cyan')
Sum_of_squared_distances = []
K = range(1,30)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(input_array)
    Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
from sklearn.metrics import silhouette_score
for n_clusters in range(2,30):
    clusterer = KMeans (n_clusters=n_clusters)
    preds = clusterer.fit_predict(input_array)
    centers = clusterer.cluster_centers_

    score = silhouette_score (input_array, preds)
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
