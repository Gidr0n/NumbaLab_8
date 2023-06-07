# 1
import random
from numba import jit
import numpy as np

N = 1000000
A = np.random.randint(0, 1001, size=N)
B = A + 100
B_mean = np.mean(B)
print("Среднее значение массива B:", B_mean)

# 2
import random
import string
import numpy as np
from numba import jit

N = 2000000
table = np.zeros((N, 4))
for i in range(N):
    for j in range(4):
        table[i][j] = random.random()

letters = string.ascii_lowercase
key_column = np.random.choice(list(letters), N)
table = np.column_stack((table, key_column))

@jit(nopython=True)
def filter_table(table):
    filtered_table = []
    for row in table:
        if row[4] in letters[:5]:
            filtered_table.append(row)
    return filtered_table

filtered_table = filter_table(table)
print(len(filtered_table))
#Лабораторная работа
# 1
import pandas as pd
import time

recipes = pd.read_csv('recipes_sample.csv', index_col=0)
reviews = pd.read_csv('reviews_sample.csv', index_col=0)

recipes['id'] = recipes['id'].astype(int)
recipes['n_steps'] = recipes['n_steps'].astype(float)
recipes['n_ingredients'] = recipes['n_ingredients'].astype(float)

reviews['recipe_id'] = reviews['recipe_id'].astype(int)
reviews['user_id'] = reviews['user_id'].astype(int)
reviews['rating'] = reviews['rating'].astype(float)
reviews['date'] = pd.to_datetime(reviews['date'])

def mean_rating_A(reviews):
    total_rating = 0
    count = 0
    for index, row in reviews.iterrows():
        if row['date'].year == 2010:
            total_rating += row['rating']
            count += 1
    return total_rating/count

def mean_rating_B(reviews):
    reviews_2010 = reviews[reviews['date'].dt.year == 2010]
    total_rating = 0
    count = 0
    for index, row in reviews_2010.iterrows():
        total_rating += row['rating']
        count += 1
    return total_rating/count

def mean_rating_C(reviews):
    reviews_2010 = reviews[reviews['date'].dt.year == 2010]
    return reviews_2010['rating'].mean()

print(mean_rating_A(reviews))
print(mean_rating_B(reviews))
print(mean_rating_C(reviews))

start_time = time.time()
mean_rating_A(reviews)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
mean_rating_B(reviews)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
mean_rating_C(reviews)
print("--- %s seconds ---" % (time.time() - start_time))
# 3
def get_word_reviews_count(df):
    word_reviews = {}
    word_reviews_count = {}
    for _, row in df.dropna(subset=['review']).iterrows():
        recipe_id, review = row['recipe_id'], row['review']
        words = set(review.split(' '))
        for word in words:
            if word not in word_reviews:
                word_reviews[word] = []
            word_reviews[word].append(recipe_id)
            if word not in word_reviews_count:
                word_reviews_count[word] = 0
            word_reviews_count[word] += 1
    return word_reviews_count
# 4
import pandas as pd
import numpy as np
from numba import jit

reviews = pd.DataFrame({'recipe_id': [1, 1, 2, 2, 3, 3],
                        'rating': [4, 5, 3, 2, 1, 2]})
def MAPE_1(reviews):
    reviews = reviews[reviews['rating'] != 0]
    mape = 0
    for recipe_id in reviews['recipe_id'].unique():
        recipe_reviews = reviews[reviews['recipe_id'] == recipe_id]
        mean_rating = recipe_reviews['rating'].mean()
        for index, row in recipe_reviews.iterrows():
            mape += abs(row['rating'] - mean_rating) / mean_rating
    return mape / len(reviews)

start_time = time.time()
MAPE_1(reviews)
print("--- %s seconds ---" % (time.time() - start_time))

@jit(nopython=True)
def MAPE_2(reviews):
    reviews = reviews[reviews['rating'] != 0]
    mape = 0
    for recipe_id in reviews['recipe_id'].unique():
        recipe_reviews = reviews[reviews['recipe_id'] == recipe_id]
        mean_rating = recipe_reviews['rating'].mean()
        for index, row in recipe_reviews.iterrows():
            mape += abs(row['rating'] - mean_rating) / mean_rating
    return mape / len(reviews)

start_time = time.time()
MAPE_2(reviews)
print("--- %s seconds ---" % (time.time() - start_time))


def MAPE_3(reviews):
    reviews = reviews[reviews['rating'] != 0]
    recipe_means = reviews.groupby('recipe_id')['rating'].mean()
    recipe_ids = recipe_means.index
    mean_ratings = recipe_means.values
    ratings = reviews['rating'].values
    mape = np.sum(np.abs(ratings - mean_ratings) / mean_ratings)
    return mape / len(reviews)

start_time = time.time()
MAPE_3(reviews)
print("--- %s seconds ---" % (time.time() - start_time))

@jit(nopython=True)
def MAPE_4(reviews):
    reviews = reviews[reviews['rating'] != 0]
    recipe_means = reviews.groupby('recipe_id')['rating'].mean()
    recipe_ids = recipe_means.index
    mean_ratings = recipe_means.values
    ratings = reviews['rating'].values
    mape = np.sum(np.abs(ratings - mean_ratings) / mean_ratings)
    return mape / len(reviews)

start_time = time.time()
MAPE_4(reviews)
print("--- %s seconds ---" % (time.time() - start_time))
