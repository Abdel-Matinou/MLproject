# %%


# %%
# let's start by importing the libraries

# Dataframe manipulation library
from ast import NotIn
import random
from matplotlib import container
from nbformat import write
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.distance import hamming
import time
import streamlit as st
from datetime import date
import watchdog

# %% [markdown]
# ### Loading data

# %%

# importing data set
df_books = pd.read_csv('./dataset/Books.csv', low_memory=False)
# rating data set
df_ratings = pd.read_csv('./dataset/Ratings.csv')

# we will not use the user data set which contains features on location and Age of each user.
# we will not need it for our model

# %% [markdown]
# ### Pre processing

# %%
# books data
# keep only columns needed to reduce memory
df_books.columns = ['isbn', 'title', 'author', 'year',
                    'publisher', 'img_url_s', 'img_url_m', 'img_url_l']
books = df_books[["isbn", "title", "author"]]
books.set_index(keys='isbn', drop=True, inplace=True)


# %%
# rating data
ratings = df_ratings.copy()
ratings.columns = ['user', 'isbn', 'rating']

# %%

# %%
userid = int(date.today().toordinal()) * 100


# most popular books
# reducing the number of users and books for computational reasons.
usersPerISBN = ratings.isbn.value_counts()
ISBNsPerUser = ratings.user.value_counts()
# reduce rows
ratings = ratings[ratings["user"].isin(ISBNsPerUser[ISBNsPerUser > 30].index)]
# reduce columns
ratings = ratings[ratings["isbn"].isin(usersPerISBN[usersPerISBN > 30].index)]

# Grab Ratings - Don't let users move on if no ratings


userRatedIsbn = pd.Series(ratings[ratings['user'] == userid].isbn.unique())

st.header("Welcome to Book recommendation App")
st.subheader("Rate some books to get tuned recommendation")


st.sidebar.image("http://www.ehtp.ac.ma/images/lo.png")
max_pop_books = st.sidebar.slider(label="Number of Popular books", min_value=3,
                                  max_value=6, step=1)
#max_rd_books = st.sidebar.slider(label="Number of Random books", min_value=3,
                                 #max_value=6, step=1)
popularBooks = ratings['isbn'].value_counts().index
popularRecom = popularBooks[~popularBooks.isin(userRatedIsbn)][:max_pop_books]
# Get book info based on its isbn


def bookInfo(isbn):
    title = books.at[isbn, "title"]
    author = books.at[isbn, "author"]
    img = df_books.set_index(keys='isbn').at[isbn, "img_url_l"]
    return title, author, img

user_input = {'user': [], 'isbn': [], 'rating': []}

with st.container():
    with st.form(key='form', clear_on_submit=False, ):
        st.subheader("Please rate the following books from 1 to 10")
        # Create five rows of elements
        cols = st.columns(1)
        for i, col in enumerate(cols):

            for singlebook in popularRecom[:max_pop_books]:
                
                book_to_rate = bookInfo(singlebook)
            #   For each book create three columns
                row1_0, row1_1, row1_2 = st.columns((1, 2, 3))

            #     # The first will contain the image
                with row1_1:
                    st.image(book_to_rate[2])

            #     # Second will have the rating radio form
                with row1_2:
                    ratings_options = ("-", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
                    user_book_rating = st.radio(book_to_rate[1],
                                                ratings_options,
                                                key=i)
                if user_book_rating == '-':
                    user_book_rating = np.nan

                # # Add rating to list
                user_input['user'].append(userid)
                user_input['isbn'].append(singlebook)
                user_input['rating'].append(user_book_rating)
            
            ratings = ratings[~ratings["isbn"].isin(user_input['isbn'])]
            popularBooks = ratings['isbn'].value_counts().index
            popularRecom = popularBooks[~popularBooks.isin(userRatedIsbn)][:max_pop_books]
        # Add form submit button
        submitted = st.form_submit_button()


if submitted:
    ratings = ratings.append(pd.DataFrame(data=user_input))

    def Get_userRatings(user, N=10):
        # N = Maximum number of books to get
        UserRatings = ratings[ratings['user'] == user]
        UserRatings_Sorted = UserRatings.sort_values(
            by='rating', ascending=False)[:N]
        UserRatings_Sorted['title'] = UserRatings_Sorted['isbn'].apply(bookInfo)

        return UserRatings_Sorted

    # Build the user-item interaction matrix
    userItemRatingMatrix = pd.pivot_table(ratings,  values="rating",
                                        index=['user'], columns=['isbn'])


    def distance(user1, user2):
        try:
            user1Ratings = userItemRatingMatrix.transpose()[user1]
            user2Ratings = userItemRatingMatrix.transpose()[user2]
            distance = hamming(user1Ratings, user2Ratings)
        except:
            distance = np.nan
        return distance


    def nearestNeighbors(user, K=10):
        # fetch all users
        allUsers = pd.DataFrame(userItemRatingMatrix.index)
        allUsers = allUsers[allUsers.user != user]
        allUsers["distance"] = allUsers['user'].apply(lambda x: distance(user, x))
        KnearestUsers = allUsers.sort_values(
            ["distance"], ascending=True)['user'][:K]
        return KnearestUsers

    @st.cache(suppress_st_warning=True)
    def topN(user, N=max_pop_books):
        KnearestUsers = nearestNeighbors(user)
        NNRatings = userItemRatingMatrix[userItemRatingMatrix.index.isin(
            KnearestUsers)]
        avgRatings = NNRatings.apply(np.nanmean).dropna()
        booksAlreadyRated = userItemRatingMatrix.transpose()[user].dropna().index
        avgRatings = avgRatings[~avgRatings.index.isin(booksAlreadyRated)]

        #popularRecom

        topNISBNs = avgRatings.sort_values(ascending=False).index[:N]
        return topNISBNs, pd.Series(topNISBNs).apply(bookInfo)

    with st.spinner(text='In progress'):
        time.sleep(5)
        R_isbns, recommended = topN(userid)
        st.success('Done')
    ratings = ratings[~ratings.isbn.isin(R_isbns)]
    popularBooks = ratings['isbn'].value_counts().index
    popularRecom = popularBooks[~popularBooks.isin(userRatedIsbn)][:max_pop_books]

    st.header("You might like these books !")
    for b in recommended:
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(b[2])
            with col2:
                st.text("Title")
                st.markdown(b[0])
            with col3:
                st.text("Author")
                st.subheader(b[1])
    st.button("Rate other books", key='button')


