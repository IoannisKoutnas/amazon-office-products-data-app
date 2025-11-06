# Author: Ioannis Koutnas

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

# Load and sample data
df = pd.read_csv('Office_Products.csv.gz', compression='gzip')  # Load dataset from gzip compressed CSV
N_USERS = 5000  # Limit to top 5000 users for memory/performance
N_PRODUCTS = 5000  # Limit to top 5000 products similarly

# Filter for top users and products to reduce dataset size
top_users = df['user_id'].value_counts().nlargest(N_USERS).index
top_products = df['parent_asin'].value_counts().nlargest(N_PRODUCTS).index
df_small = df[df['user_id'].isin(top_users) & df['parent_asin'].isin(top_products)]

# Create user-item rating matrix, rows are users, columns are products, values are ratings
user_item = df_small.pivot_table(index='user_id', columns='parent_asin', values='rating')

# Fill missing ratings with 0 for cosine similarity calculation
user_item_filled = user_item.fillna(0)

# Compute similarity matrices for users and items based on ratings vectors
user_similarity = cosine_similarity(user_item_filled)
item_similarity = cosine_similarity(user_item_filled.T)  # Transpose for item-based similarity

# List of unique user IDs and product ASINs (used for lookups)
user_ids = list(user_item_filled.index)
item_ids = list(user_item_filled.columns)

def get_similar_users(user_id, top_n=5):
    """
    Returns the top N most similar users to the input user_id based on cosine similarity
    """
    try:
        idx = user_ids.index(user_id)  # Find matrix row index for user
    except ValueError:
        return []  # User not found
    scores = user_similarity[idx]  # Get similarity scores for user to others
    similar_idxs = np.argsort(scores)[::-1][1:top_n+1]  # Top N excluding self
    return [(user_ids[i], round(scores[i], 3)) for i in similar_idxs]

def get_similar_products(product_asin, top_n=5):
    """
    Returns the top N most similar products (by ASIN) to the input product_asin
    """
    try:
        idx = item_ids.index(product_asin)
    except ValueError:
        return []  # Product not found
    scores = item_similarity[idx]
    similar_idxs = np.argsort(scores)[::-1][1:top_n+1]
    return [(item_ids[i], round(scores[i], 3)) for i in similar_idxs]

def recommend_products(user_id, top_n=5):
    """
    User-based collaborative filtering recommendation for given user_id
    Returns products not rated by user but liked by top similar users
    """
    try:
        idx = user_ids.index(user_id)
    except ValueError:
        return []
    sim_scores = user_similarity[idx]
    sim_users = np.argsort(sim_scores)[::-1][1:11]  # Top 10 similar users excluding self

    user_ratings = user_item_filled.iloc[idx]
    products_already_rated = user_ratings[user_ratings > 0].index  # User's rated products

    scores = pd.Series(0, index=item_ids)
    sim_scores_sum = pd.Series(0, index=item_ids)

    # Aggregate weighted ratings from similar users for each product
    for su in sim_users:
        sim_user_ratings = user_item_filled.iloc[su]
        similarity_weight = sim_scores[su]
        mask = sim_user_ratings > 0
        scores[mask] += sim_user_ratings[mask] * similarity_weight
        sim_scores_sum[mask] += similarity_weight

    # Compute predicted ratings by normalizing weighted sum
    predicted_ratings = scores / sim_scores_sum.replace(0, np.nan)
    # Remove products already rated by user
    predicted_ratings = predicted_ratings.drop(products_already_rated, errors='ignore')
    recommendation = predicted_ratings.dropna().sort_values(ascending=False).head(top_n)
    return list(zip(recommendation.index, recommendation.round(3)))

def display_random_samples():
    """
    Returns a string showing 10 random users and 10 random products for user convenience
    """
    rand_users = random.sample(user_ids, min(10, len(user_ids)))
    rand_items = random.sample(item_ids, min(10, len(item_ids)))
    return "Random sample users:\n" + "\n".join(rand_users) + \
           "\n\nRandom sample products (ASIN):\n" + "\n".join(rand_items)

def get_data_stats():
    """
    Returns summary stats about the current dataset sample
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

# ---- GUI setup ----
root = tk.Tk()
root.title("Amazon Office Products Analysis")

# Scrollable text widget for output display
output_area = scrolledtext.ScrolledText(root, width=70, height=20, wrap=tk.WORD)
output_area.grid(row=0, column=0, columnspan=4, padx=10, pady=10)

# Entry box for user input (user id or product ASIN)
entry = tk.Entry(root, width=50)
entry.grid(row=1, column=0, columnspan=4, padx=10)

def clear_output():
    """
    Clear the output text widget and any existing plot canvases
    """
    output_area.delete(1.0, tk.END)
    for widget in root.grid_slaves():
        if isinstance(widget, FigureCanvasTkAgg):
            widget.get_tk_widget().destroy()

def show_histograms():
    """
    Plot and show histogram of ratings distribution in dataset
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
    Plot and show bar chart of top 10 most rated products
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
    Handle product similarity query from user input
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
    Handle user similarity query from user input
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
    Handle product recommendations for user input
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
    Show random sample of users and products for convenience
    """
    clear_output()
    samples = display_random_samples()
    output_area.insert(tk.END, samples + "\n")

def on_data_explore():
    """
    Show dataset summary statistics
    """
    clear_output()
    stats = get_data_stats()
    output_area.insert(tk.END, stats + "\n")

def on_exit():
    """
    Exit the application cleanly
    """
    root.destroy()

# Top button row
btn_product = tk.Button(root, text="Product Similarity", command=on_product_sim)
btn_product.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
btn_user = tk.Button(root, text="User Similarity", command=on_user_sim)
btn_user.grid(row=3, column=1, padx=5, pady=5, sticky='ew')
btn_recommend = tk.Button(root, text="Recommend Products", command=on_recommend)
btn_recommend.grid(row=3, column=2, padx=5, pady=5, sticky='ew')
btn_random = tk.Button(root, text="Show Random Samples", command=on_random)
btn_random.grid(row=3, column=3, padx=5, pady=5, sticky='ew')

# Bottom button row
btn_hist = tk.Button(root, text="Show Rating Histogram", command=show_histograms)
btn_hist.grid(row=4, column=0, padx=5, pady=10, sticky='ew')
btn_popular = tk.Button(root, text="Show Popular Products", command=show_popular_products)
btn_popular.grid(row=4, column=1, padx=5, pady=10, sticky='ew')
btn_data = tk.Button(root, text="Data Exploration", command=on_data_explore)
btn_data.grid(row=4, column=2, padx=5, pady=10, sticky='ew')
btn_exit = tk.Button(root, text="Exit", command=on_exit)
btn_exit.grid(row=4, column=3, padx=5, pady=10, sticky='ew')

root.mainloop()
