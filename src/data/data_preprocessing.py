import pandas as pd


def clean_data(file_path):
    """
    Clean the movie recommendation data.

    Parameters:
    - file_path (str): Path to the raw data file.

    Returns:
    - pd.DataFrame: Cleaned data.
    """

    # Load the raw data
    raw_data = pd.read_csv(file_path)

    # 1. Convert alphanumeric ratings to numeric
    rating_mapping = {'Five': 5, 'Four': 4, 'Three': 3, 'Two': 2, 'One': 1}
    raw_data['Rating'] = raw_data['Rating'].replace(rating_mapping)

    # 2. Convert all special characters like 5x, 5y, 2? into numeric format
    raw_data['Rating'] = raw_data['Rating'].str.extract('(\d+\.?\d*)').astype(float)
    raw_data['Rating'] = pd.to_numeric(raw_data['Rating'], errors='coerce')

    # 3. Standardize movie names - fix typos and inconsistencies
    raw_data['Movie'].replace({'Matrix': 'The Matrix'}, inplace=True)
    raw_data['Movie'] = raw_data['Movie'].str.lower()  # Convert to lowercase for consistency
    raw_data['Movie'] = raw_data['Movie'].str.strip()  # Remove leading/trailing whitespaces

    # 4. Replace missing Rating values with the mean values of that particular movie
    raw_data['Rating'].fillna(raw_data.groupby('Movie')['Rating'].transform('mean'), inplace=True)

    # 5. If a user has rated the same movie multiple times, average the ratings
    raw_data = raw_data.groupby(['User', 'Movie'], as_index=False)['Rating'].mean()

    # Drop duplicates (in case any were created during the cleaning process)
    raw_data = raw_data.drop_duplicates()

    return raw_data


def expand_data(cleaned_data, movie_catalog_data_filepath, expanded_data_filepath):
    df1 = pd.DataFrame(cleaned_data)
    df2 = pd.read_csv(movie_catalog_data_filepath)

    # do some cleaning on the second dataset
    df2['Movie'] = df2['Movie'].str.lower()  # Convert to lowercase for consistency
    df2['Movie'] = df2['Movie'].str.strip()  # Remove leading/trailing whitespaces

    # For the case of Star Wars, use "Star Wars: Episode I - The Phantom Menace" - it means it has to be renamed
    # - because it is episode 1, and is pretty similar to all other occurrences of star wars
    df2['Movie'] = df2['Movie'].replace('star wars: episode i - the phantom menace', 'star wars')

    # For the case of The Godfather, pick "The Godfather: Part III"
    df2['Movie'] = df2['Movie'].replace('the godfather: part iii', 'the godfather')

    # Merge the datasets based on movie name and rating
    merged_df = pd.merge(df1, df2, on=['Movie'], how='left')
    merged_df.to_csv(expanded_data_filepath)

    # Return the merged DataFrame
    return merged_df
