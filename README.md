# Filmwise Movie Recommendation System

Filmwise is a proof-of-concept movie recommendation system designed to enhance the movie streaming experience for users. The system combines collaborative and content-based filtering techniques to provide personalized movie recommendations. This README file documents the process of developing the Filmwise project.

## Data Cleaning

The initial step involved cleaning the provided dataset, which included the following steps:

1. **Handling Missing Values:**
   - Missing rating values were replaced with the mean ratings for the particular movie, ensuring a more complete and accurate dataset.

2. **Alphanumeric to Numeric Conversion:**
   - Non-numeric ratings were converted to their numeric equivalents, such as converting 'Five' to 5.

3. **Standardizing Movie Names:**
   - Typos and inconsistencies in movie names were corrected to ensure a standardized representation.

4. **Handling Duplicate Ratings:**
   - If a user had multiple ratings for the same movie, only the latest rating was retained, streamlining the dataset.

## Data Expansion

To enrich the dataset, additional information such as movie genres was incorporated. This supplementary data was sourced from a larger dataset obtained from Kaggle, enhancing the breadth of information available for analysis and recommendation.

## Data Analysis

Data analysis was conducted to gain insights into user preferences and patterns within the dataset. Key analyses include:

1. **Average Ratings per Movie:**
   - Calculated to understand the overall reception of each movie.

2. **Number of Ratings per User and per Movie:**
   - Explored to identify user engagement and popular movies.

3. **Most Popular Genres:**
   - Investigated to understand the distribution of genres and user preferences.

## Training Process

The recommendation algorithm involves a hybrid approach, combining collaborative and content-based filtering:

1. **Collaborative Filtering:**
   - A collaborative filtering model using the Surprise library's SVD algorithm was trained on the cleaned dataset. This model leverages user ratings to make personalized recommendations.

2. **Content-Based Filtering:**
   - A content-based filtering model using movie genres and TF-IDF was trained. This model considers the content of movies to enhance recommendations.

3. **Hybrid Recommendations:**
   - A hybrid recommendation function was implemented to combine collaborative and content-based scores, providing more robust and personalized movie recommendations.

The weights in the combination formula were adjusted based on experimentation and performance evaluation to optimize the recommendation system.

## Directory Structure
The directory structure of the project is as shown below:
# filmwise

- **data/**
  - **raw/**
    - raw_data.csv
  - **cleaned/**
    - expanded.csv
  - **catalog/**
    - movies.csv

- **src/**
  - \_\_init\_\_.py
  - **data/**
    - \_\_init\_\_.py
    - data_analysis.py
    - data_preprocessing.py
  - **algorithms/**
    - \_\_init\_\_.py
    - hybrid_recommendation.py
  - main.py

- requirements.txt
- README.md
- .gitignore



## How to Run
1. **Clone this repository** - Clone the repo and open it in the terminal.
2. **Create a python virtual environment** - On the terminal, run `pip install virtualenv`. Then `virtualenv env` to create a virtual environment named env
3. **Activate the virtual environment** - For linux, do `source .env/bin/activate`
4. **Install Dependencies** - install all the required dependencies. Run `pip install requirements.txt`
5. **Run the main.py** - Run the main.py file in the src directory
