"""
Content-based recommendation system that suggests Amazon products based on product names and categories to match user input.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Load and prepare the product dataset."""
    df = pd.read_csv('processed_products.csv')
    df = df.dropna(subset=['name', 'main_category'])
    
    # simplify searchable text - just use name and categories once
    df['searchable_text'] = (
        df['name'].str.lower() + ' ' + 
        df['main_category'].str.lower() + ' ' +
        df['sub_category'].fillna('').str.lower()
    )
    
    return df

def recommend_products(user_input, df, top_n=5):
    """Recommend products based on text similarity to user input."""
    # clean user input
    user_input = user_input.lower()
    
    # create TF-IDF vectors
    tfidf = TfidfVectorizer(
        analyzer='char_wb',  # use character n-grams with word boundaries
        ngram_range=(3, 5),  # use 3-5 character sequences
        max_features=5000
    )
    
    # create document corpus including user input
    all_text = df['searchable_text'].tolist() + [user_input]
    tfidf_matrix = tfidf.fit_transform(all_text)
    
    # find similarities
    similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    df['similarity'] = similarities
    
    recommendations = (
        df[df['similarity'] > 0.1]  # only keep matches with similarity > 0.1
        .nlargest(top_n, 'similarity')
    )
    return recommendations[['name', 'main_category', 'sub_category', 'ratings', 'similarity', 'actual_price', 'discount_price']]

def format_recommendations(recommendations):
    """Format recommendations for display."""
    if len(recommendations) == 0:
        print("\nNo matching products found. Try a different search term.")
        return
        
    print("\nTop Recommendations:")
    for _, row in recommendations.iterrows():
        print(f"Product: {row['name'].title()}")
        print(f"Category: {row['main_category'].title()} > {row['sub_category'].title()}")
        print(f"Rating: {row['ratings']:.1f}")
        if pd.notna(row['discount_price']):
            # prices are in Indian Rupees, convert to USD
            usd_discount = row['discount_price'] / 86.65
            usd_actual = row['actual_price'] / 86.65
            print(f"Price: ${usd_discount:.2f} (Original: ${usd_actual:.2f})")
        else:
            usd_actual = row['actual_price'] / 86.65
            print(f"Price: ${usd_actual:.2f}")
        print(f"Similarity Score: {row['similarity']:.2f}")
        print("\n")

if __name__ == "__main__":

    df = load_data()
    
    print("\nProduct Recommendation System")
    print("Examples of what you can search for:")
    print("- 'bluetooth speaker'")
    print("- 'gaming headset'")
    print("- 'hair dryer'")
    
    # user input
    user_input = input("\nWhat kind of product are you looking for? ")
    
    # get and display recommendations
    recommendations = recommend_products(user_input, df)
    format_recommendations(recommendations)
    
