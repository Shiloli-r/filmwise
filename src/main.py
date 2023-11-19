# main.py

from src.data import data_preprocessing as preprocess


# from src.algorithms import recommendation_algorithm
# from src.ui import cli

def main():
    # Load and clean the data
    cleaned_data = preprocess.clean_data("../data/raw/raw_data.csv")

    # Perform data analysis to extract insights
    # user_preferences = data_analysis.extract_user_preferences(cleaned_data)
    #
    # # Train the recommendation algorithm
    # model = recommendation_algorithm.train_model(cleaned_data)
    #
    # # Initialize the CLI interface
    # filmwise_cli = cli.FilmwiseCLI(model, user_preferences)
    #
    # # Run the CLI
    # filmwise_cli.run()
    print(cleaned_data)


if __name__ == "__main__":
    main()
