# main.py

from src.data import data_analysis, data_preprocessing as preprocess
from src.algorithms import hybrid_recommendation as algorithm


# from src.algorithms import recommendation_algorithm
# from src.ui import cli

def main():
    # Load and clean the data
    cleaned_data = preprocess.clean_data("../data/raw/raw_data.csv")
    print("\n------------- cleaned data ------------------------")
    print(cleaned_data)
    print("\n------------- expanded data ------------------------")
    expanded_data = preprocess.expand_data(cleaned_data, "../data/catalog/movies.csv", "../data/cleaned/expanded.csv")
    print(expanded_data)

    # Perform data analysis to extract insights
    user_preferences = data_analysis.extract_user_preferences(expanded_data)
    print("\n------------------- User Preferences -------------------------------")
    print(user_preferences)
    print("\n--------------------- Watch History --------------------------------")
    watch_history = data_analysis.get_watch_history(expanded_data)
    print(watch_history)

    # Train collaborative filtering model
    collaborative_model = algorithm.train_collaborative_filtering_model(expanded_data)

    # Train content-based filtering model
    content_based_matrix = algorithm.train_content_based_filtering_model(expanded_data)

    # Get hybrid recommendations for a specific user
    user_id = 'Alice'
    hybrid_recommendations = algorithm.hybrid_recommendation(user_id, collaborative_model, content_based_matrix, expanded_data,
                                                   n=5)
    print(hybrid_recommendations)

    # # Initialize the CLI interface
    # filmwise_cli = cli.FilmwiseCLI(model, user_preferences)

    # # Run the CLI
    # filmwise_cli.run()


if __name__ == "__main__":
    main()
