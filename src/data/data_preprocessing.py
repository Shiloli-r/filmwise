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
