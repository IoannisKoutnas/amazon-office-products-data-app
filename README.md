# amazon-office-products-data-app
A Python GUI application to explore, analyze, and recommend office products using Amazon reviews data.

## Features

- **Product & User Similarity:** Instantly find similar products (ASIN) or similar users based on ratings.
- **Personalized Recommendations:** Get product recommendations for any user using collaborative filtering.
- **Interactive GUI:** Scrollable output, entry boxes, and 4+4 intuitive buttons for easy interaction.
- **Visualizations:** Display rating distribution histograms and bar charts for top popular products.
- **Data Exploration:** See summary statistics and random valid IDs for quick test queries.
- **Clean Code:** Fully commented Python code, clear structure, easily customizable.

## Getting Started

### Prerequisites
- Python 3.8 or newer
- Install required packages:
  ```
  pip install pandas numpy scikit-learn matplotlib
  ```

### Usage
1. Place the dataset file (e.g. `Office_Products.csv.gz`) in your working folder.
2. Run:
   ```
   python project.py
   ```
3. Use the GUI dashboard for similarity queries, recommendations, stats, and visualizations.

### Dataset info
This app is built for the Amazon Office Products (5-core) ratings dataset.  
Columns required:  
- `user_id`, `parent_asin`, `rating`, `timestamp`

The dataset is not included due to size/license restrictions.  
Find it here: https://amazon-reviews-2023.github.io/data_processing/5core.html

## How Recommendations Work

Product recommendations are computed using *user-based collaborative filtering*:
1. Find top 10 most similar users to the target user (based on ratings with cosine similarity).
2. Aggregate and normalize their ratings for all products the target user hasn't rated.
3. Recommend products with the highest predicted rating scores.
