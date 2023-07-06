# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors 
from sklearn.preprocessing import MinMaxScaler

# The columns that will be taken into account when making item-based similarity recommendations
item_columns = []    

# Number of neighbors to take into account
N_NEIGHBORS = 10


# Handler for Item DataFrame
class ItemData:
    def __init__(self):
        self.df = pd.read_csv('./items.csv')
        self._scale_cols()
        self.item_columns = ['scaled_runtime', 'vote_scaled', 'Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 
            'Crime', 'Thriller', 'Horror', 'History','Science Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 
            'Western', 'TV Movie', 'ratio_scaled', 'pop_scaled']
        self.scaled_df = self.df[self.item_columns]

    def _scale_cols(self):
        runtime_col = self.df['runtime'].values.reshape(-1, 1)
        runtime_scaler = MinMaxScaler().fit(runtime_col)
        self.df['scaled_runtime'] = runtime_scaler.transform(runtime_col)

        vote_col = self.df['vote_average'].values.reshape(-1, 1)
        vote_scaler = MinMaxScaler().fit(vote_col)
        self.df['vote_scaled'] = vote_scaler.transform(vote_col)

        ratio_col = self.df['rb_ratio'].values.reshape(-1, 1)
        ratio_scaler = MinMaxScaler().fit(ratio_col)
        self.df['ratio_scaled'] = ratio_scaler.transform(ratio_col)

        pop_col = self.df['pop_bin'].values.reshape(-1, 1)
        ratio_scaler = MinMaxScaler().fit(pop_col)
        self.df['pop_scaled'] = ratio_scaler.transform(pop_col)
    
    def get_filtered_row_by_id(self, id):
        return self.df[self.df['id'] == int(id)][self.item_columns]
    
    def get_id_by_idx(self, idx):
        return self.df.at[idx, 'id']
    
    def get_random_id(self):
        return self.df.sample(1)['id'].values[0]
    
    def get_row_by_id(self, id):
        return self.df[self.df['id'] == id]

    def get_movie_title_by_id(self, id):
        return self.get_row_by_id(id)['title'].values[0]
    
    def get_movie_overview_by_id(self, id):
        return self.get_row_by_id(id)['overview'].values[0]
        

# Handler for User DataFrame
class UserData:
    def __init__(self):
        self.df = pd.read_csv('./users.csv')
        self.df = self.df.fillna(0)


###### Recommender System
class Recommender:
    def __init__(self):
        # Load preprocessed dataframes
        self.item_handler = ItemData()
        self.user_handler = UserData()
        print("Dataframes loaded...")

        self.preferences = pd.DataFrame(columns=self.user_handler.df.columns[1:])    # For user data
        self.preferences.loc[0] = 0    # Initialize all ratings to zero

        self.item_picks = pd.DataFrame(columns=self.item_handler.df.columns)
        
        self.n_picks = 1
        self.recommended_ids = []    # Resets every time
        self.seen_movies = []

        # Initialize nearest neighbor algorithm. With p=1, euclidean distance is our metric
        self.user_recommender = NearestNeighbors(n_neighbors=N_NEIGHBORS, p=2).fit(self.user_handler.df.drop('user_id', axis=1))
        self.item_recommender = NearestNeighbors(n_neighbors=N_NEIGHBORS, p=2).fit(self.item_handler.scaled_df)

        # Initialize recommended movies
        for i in range(3):
            self.recommended_ids.append(self.get_item_recommendation())    # Getting random movies


    def on_pick(self, idx, rating):
        '''
            Called whenever the user picks a new movie.
            idx: [0, 2] -> which one of the recommendations was picked out of the 3 suggestions
        '''
        self.n_picks += 1

        chosen_movie_id = self.recommended_ids[idx]
        self.update(chosen_movie_id, rating)

        # Recommend new movies
        self.recommended_ids[0] = self.get_item_recommendation()
        self.recommended_ids[1] = self.get_user_recommendation()
        self.recommended_ids[2] = self.get_joint_recommendation()

        return self.recommended_ids
    
    def get_descs_for_recommended(self, recs):
        descs = []
        for rec in recs:
            info = {}
            info['title'] = self.item_handler.get_movie_title_by_id(rec)
            info['overview'] = self.item_handler.get_movie_overview_by_id(rec)
            descs.append(info)
        return descs

    def update(self, movie_id, rating):
        '''
            Update user preferences based on last picked movie (and given rating)
        '''
        self.seen_movies.append(movie_id)

        # Update user data
        self.preferences.at[0, str(movie_id)] = rating

        # Update item data - but only if the user liked it
        if rating > 2.5:
            new_row = self.item_handler.get_row_by_id(movie_id)
            self.item_picks = pd.concat([self.item_picks, new_row], axis=0)

    
    def get_item_recommendation(self):
        '''
            Make recommendation based on item similarity
        '''
        # If user hasn't picked any movies they like yet, pick something random
        if not self.item_picks.empty:
            filtered_picks = self.item_picks[self.item_handler.item_columns]
            
            # Return movie that's closest to average preference
            summed_preferences = filtered_picks.sum(axis=0)
            average_preferences = summed_preferences / filtered_picks.shape[0]

            dist, idxes = self.item_recommender.kneighbors([average_preferences], min(len(self.seen_movies), self.item_handler.df.shape[0]))    # guarenteed to pick a movie that has not been seen before
            
            for idx in idxes[0]:
                new_id = self.item_handler.get_id_by_idx(idx)
                if new_id not in self.seen_movies:
                    return new_id

        # Pick a random movie if strategy did not work
        return self.item_handler.get_random_id()     

    
    def get_user_recommendation(self):
        '''
            Make recommendation based on user similarity
        '''
        # If user hasn't chosen anything yet
        if self.item_picks.empty:
            return self.item_handler.get_random_id()

        _, idx = self.user_recommender.kneighbors(self.preferences.values, 25)

        # Find the closest user's top 3 movies. If all have been seen, move onto the next user until a candidate movie is found
        for best_idx in idx[0]:
            cols_to_drop = ['user_id']
            # Find best movie
            for i in range(3):
                best_movie = self.user_handler.df.drop(cols_to_drop, axis=1).iloc[best_idx].idxmax(axis=0)
                if best_movie in self.seen_movies:
                    cols_to_drop.append(best_movie)
                    continue
                
                if self.user_handler.df.at[best_idx, best_movie] > 2.5:
                    return int(best_movie)

        # Otherwise, return random movie
        return self.item_handler.get_random_id()

    
    def get_joint_recommendation(self):
        '''
            Make recommendation based on both item and user similarity
        '''
        # If user hasn't chosen anything yet
        if self.item_picks.empty:
            return self.item_handler.get_random_id()

        # Get similar users
        _, user_idxs = self.user_recommender.kneighbors(self.preferences.values, 10)

        # Get similar items
        summed_preferences = self.item_picks[self.item_handler.item_columns].sum(axis=0)
        average_preferences = summed_preferences / self.item_picks.shape[0]

        n_movies = min(len(self.seen_movies), self.item_handler.df.shape[0])
        _, item_idxs = self.item_recommender.kneighbors([average_preferences], n_movies)    # guarenteed to pick a movie that has not been seen before

        score_sums = [0 for i in range(n_movies)]
        n_votes = [0 for i in range(n_movies)]

        # Sum ratings per movie
        for i, movie_idx in enumerate(item_idxs[0]):
            movie_id = self.item_handler.get_id_by_idx(movie_idx)
            if movie_id in self.seen_movies:
                continue

            for user_id in user_idxs[0]:
                score = self.user_handler.df.at[user_id, str(movie_id)]
                if score != 0:
                    score_sums[i] += score
                    n_votes[i] += 1
        

        # Calculate per-movie score
        final_score = []
        for i, score in enumerate(score_sums):
            if n_votes[i] > 0:
                final_score.append(score/n_votes[i])
            else:
                final_score.append(-1)

        # Find best score
        best_score_idx = final_score.index(max(final_score))
        best_movie_idx = item_idxs[0][best_score_idx]
        return self.item_handler.get_id_by_idx(best_movie_idx)