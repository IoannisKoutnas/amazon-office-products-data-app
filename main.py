# Author: Ioannis Koutnas
# Title: Amazon Office Products Analysis Dashboard
# Purpose: GUI application for similarity, recommendations, and data exploration on Amazon review data

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # Efficient vector similarity
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.figure import Figure                    # Visualization back-end
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

# ----------- Data Loading and Preprocessing -----------

df = pd.read_csv('Office_Products.csv.gz', compression='gzip')  # Import compressed dataset

N_USERS = 5000     # Use only the top N users to control matrix size and memory
N_PRODUCTS = 5000  # Use only the top N products (by rating count)

# Select only the most active users and most rated products for scalability
top_users = df['user_id'].value_counts().nlargest(N_USERS).index
top_products = df['parent_asin'].value_counts().nlargest(N_PRODUCTS).index
df_small = df[df['user_id'].isin(top_users) & df['parent_asin'].isin(top_products)]

# Construct the user-item rating (pivot) matrix, with NaN for missing ratings
user_item = df_small.pivot_table(index='user_id', columns='parent_asin', values='rating')

# Fill missing ratings with zero for similarity computation (can use mean imputation as alternative)
user_item_filled = user_item.fillna(0)

# Precompute user-user and item-item cosine similarity matrices
user_similarity = cosine_similarity(user_item_filled)
item_similarity = cosine_similarity(user_item_filled.T) 

# Get ID lists for lookup convenience
user_ids = list(user_item_filled.index)
item_ids = list(user_item_filled.columns)

# ----------- Core Analytics and Recommendation Logic -----------

def get_similar_users(user_id, top_n=5):
    """
    Retrieve the top-N most similar users to the given user, based on cosine similarity of ratings vectors.
    Excludes self-match. Returns list of (user, similarity_score).
    """
    try:
        idx = user_ids.index(user_id)
    except ValueError:
        return []  # User ID not present in sample
    scores = user_similarity[idx]
    similar_idxs = np.argsort(scores)[::-1][1:top_n+1]  # Exclude self (best match)
    return [(user_ids[i], round(scores[i], 3)) for i in similar_idxs]

def get_similar_products(product_asin, top_n=5):
    """
    Retrieve the top-N most similar products (by ASIN); uses cosine similarity of user rating profiles.
    Excludes self-match. Returns list of (ASIN, similarity_score).
    """
    try:
        idx = item_ids.index(product_asin)
    except ValueError:
        return []  # ASIN not found in sample
    scores = item_similarity[idx]
    similar_idxs = np.argsort(scores)[::-1][1:top_n+1]  # Exclude self
    return [(item_ids[i], round(scores[i], 3)) for i in similar_idxs]

def recommend_products(user_id, top_n=5):
    """
    User-based collaborative filtering recommendations.
    For a given user, predict products likely to rate highly, based on ratings of top similar users.
    Returns top N predicted (ASIN, predicted_rating).
    """
    try:
        idx = user_ids.index(user_id)
    except ValueError:
        return []
    sim_scores = user_similarity[idx]
    sim_users = np.argsort(sim_scores)[::-1][1:11]  # Top 10 similar users
    user_ratings = user_item_filled.iloc[idx]
    products_already_rated = user_ratings[user_ratings > 0].index

    # Weighted sum of ratings from similar users for each product
    scores = pd.Series(0, index=item_ids)
    sim_scores_sum = pd.Series(0, index=item_ids)
    for su in sim_users:
        sim_user_ratings = user_item_filled.iloc[su]
        weight = sim_scores[su]
        mask = sim_user_ratings > 0
        scores[mask] += sim_user_ratings[mask] * weight
        sim_scores_sum[mask] += weight

    # Normalize predicted ratings, ignoring already rated products
    predicted_ratings = scores / sim_scores_sum.replace(0, np.nan)
    predicted_ratings = predicted_ratings.drop(products_already_rated, errors='ignore')
    recommendation = predicted_ratings.dropna().sort_values(ascending=False).head(top_n)
    return list(zip(recommendation.index, recommendation.round(3)))

def display_random_samples():
    """
    Show 10 random user IDs and 10 random ASINs from the filtered sample—useful for quick testing.
    """
    rand_users = random.sample(user_ids, min(10, len(user_ids)))
    rand_items = random.sample(item_ids, min(10, len(item_ids)))
    return "Random sample users:\n" + "\n".join(rand_users) + \
           "\n\nRandom sample products (ASIN):\n" + "\n".join(rand_items)

def get_data_stats():
    """
    Compute and return a string with key dataset statistics for exploration and validation.
    """
    n_users = len(user_ids)
    n_products = len(item_ids)
    n_ratings = df_small.shape[0]
    avg_ratings_per_user = n_ratings / n_users
    avg_ratings_per_product = n_ratings / n_products
    rating_dist = df_small['rating'].value_counts().sort_index()
    return f"Data Summary:\nUsers: {n_users}\nProducts: {n_products}\nTotal Ratings: {n_ratings}\n" + \
           f"Avg Ratings/User: {avg_ratings_per_user:.2f}\nAvg Ratings/Product: {avg_ratings_per_product:.2f}\n" + \
           "Rating Distribution:\n" + "\n".join(f"{i}: {rating_dist.get(i,0)}" for i in sorted(rating_dist.index))

# ----------- GUI Setup and Event Logic -----------

root = tk.Tk()
root.title("Amazon Office Products Analysis")

# Scrollable text area for outputs and logs
output_area = scrolledtext.ScrolledText(root, width=70, height=20, wrap=tk.WORD)
output_area.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

# Entry box for inputting user ID or product ASIN
entry = tk.Entry(root, width=50)
entry.grid(row=1, column=0, columnspan=4, padx=10)

def clear_output():
    """
    Clear all output text and remove figure widgets from GUI.
    Used before rendering a new output.
    """
    output_area.delete(1.0, tk.END)
    for widget in root.grid_slaves():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

def show_histograms():
    """
    Visualize and display histogram (distribution) of product ratings in the current dataset.
    """
    clear_output()
    fig = Figure(figsize=(7,3))
    ax = fig.add_subplot(111)
    ratings = df_small['rating']
    ax.hist(ratings, bins=5, color='skyblue', edgecolor='black')
    ax.set_title('Rating Distribution')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=2, column=0, columnspan=4)
    canvas.draw()

def show_popular_products():
    """
    Show a horizontal bar chart of the top 10 most frequently rated products (by ASIN).
    """
    clear_output()
    fig = Figure(figsize=(8,3))
    ax = fig.add_subplot(111)
    popular = df_small['parent_asin'].value_counts().head(10)
    ax.bar(popular.index, popular.values)
    ax.set_title('Top 10 Popular Products')
    ax.set_ylabel('Number of Ratings')
    ax.set_xticks(range(len(popular.index)))
    ax.set_xticklabels(popular.index, rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=2, column=0, columnspan=4)
    canvas.draw()

def on_product_sim():
    """
    Callback for "Product Similarity" button. Shows similar products for entered ASIN.
    """
    clear_output()
    asin = entry.get().strip()
    if not asin:
        output_area.insert(tk.END, "Enter a product ASIN.\n")
        return
    results = get_similar_products(asin)
    if results:
        output_area.insert(tk.END, f"Similar products to {asin}:\n")
        for p, s in results:
            output_area.insert(tk.END, f"{p} (score: {s})\n")
    else:
        output_area.insert(tk.END, "Product not found or no similar products.\n")

def on_user_sim():
    """
    Callback for "User Similarity" button. Shows similar users for entered user ID.
    """
    clear_output()
    uid = entry.get().strip()
    if not uid:
        output_area.insert(tk.END, "Enter a user ID.\n")
        return
    results = get_similar_users(uid)
    if results:
        output_area.insert(tk.END, f"Similar users to {uid}:\n")
        for u, s in results:
            output_area.insert(tk.END, f"{u} (score: {s})\n")
    else:
        output_area.insert(tk.END, "User not found or no similar users.\n")

def on_recommend():
    """
    Callback for "Recommend Products" button. Shows top suggested products for entered user.
    """
    clear_output()
    uid = entry.get().strip()
    if not uid:
        output_area.insert(tk.END, "Enter a user ID for recommendations.\n")
        return
    recs = recommend_products(uid)
    if recs:
        output_area.insert(tk.END, f"Recommended products for {uid}:\n")
        for p, s in recs:
            output_area.insert(tk.END, f"{p} (predicted rating: {s})\n")
    else:
        output_area.insert(tk.END, "No recommendations available for that user.\n")

def on_random():
    """
    Callback for "Show Random Samples" button. Displays random user & product IDs.
    """
    clear_output()
    samples = display_random_samples()
    output_area.insert(tk.END, samples + "\n")

def on_data_explore():
    """
    Callback for "Data Exploration" button. Displays key statistics for current sample.
    """
    clear_output()
    stats = get_data_stats()
    output_area.insert(tk.END, stats + "\n")

def on_exit():
    """
    Callback for "Exit" button. Closes the GUI application.
    """
    root.destroy()

# ------------ GUI Button Layout ------------

btn_product = tk.Button(root, text="Product Similarity", command=on_product_sim)
btn_product.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
btn_user = tk.Button(root, text="User Similarity", command=on_user_sim)
btn_user.grid(row=3, column=1, padx=5, pady=5, sticky='ew')
btn_recommend = tk.Button(root, text="Recommend Products", command=on_recommend)
btn_recommend.grid(row=3, column=2, padx=5, pady=5, sticky='ew')
btn_random = tk.Button(root, text="Show Random Samples", command=on_random)
btn_random.grid(row=3, column=3, padx=5, pady=5, sticky='ew')

btn_hist = tk.Button(root, text="Show Rating Histogram", command=show_histograms)
btn_hist.grid(row=4, column=0, padx=5, pady=10, sticky='ew')
btn_popular = tk.Button(root, text="Show Popular Products", command=show_popular_products)
btn_popular.grid(row=4, column=1, padx=5, pady=10, sticky='ew')
btn_data = tk.Button(root, text="Data Exploration", command=on_data_explore)
btn_data.grid(row=4, column=2, padx=5, pady=10, sticky='ew')
btn_exit = tk.Button(root, text="Exit", command=on_exit)
btn_exit.grid(row=4, column=3, padx=5, pady=10, sticky='ew')

root.mainloop()
