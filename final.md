# BASEBALL DATA ANALYSIS
![Mookie_Betts_hitting_the_ball_(36478781664)](https://github.com/LikithaBommasani/BDA-602/assets/112617127/3dc861ed-1546-4403-ad77-00620523827c)

## Overview
[Baseball](https://en.wikipedia.org/wiki/Baseball) has always been a sport that relies heavily on statistics, and with the emergence of machine learning, the possibilities for statistical analysis have expanded significantly. With its ability to analyse vast amounts of data generated during baseball games, machine learning is becoming an increasingly critical tool in the sport's analytics toolkit. By leveraging sophisticated algorithms and statistical models, machine learning can uncover complex patterns and relationships in player performance, team strategy, and game outcomes that may have been previously difficult to detect. This opens new possibilities for informed decision-making, enabling coaches and analysts to optimize player performance, develop winning strategies, and improve a team's chances of success. Machine learning also has the potential to automate many of the repetitive and time-consuming tasks involved in statistical analysis, allowing analysts to focus on more complex analyses and generate deeper insights.

## Data Source
For this project, I have used the baseball.sql dataset provided by Prof. Julien Pierret. The database is comprised of approximately 30 tables, each containing data on various aspects of baseball such as batting, pitching, fielding, and results statistics. This data is crucial in conducting an in-depth analysis of the sport and its players.
The is Response is set to  Home Team Wins.

## Data Analysis
The dataset provided includes various tables that span the period between 2007 and 2012. These tables contain information on different aspects of baseball games, such as team pitching and batting statistics, box scores, and game-level data.
For the project I have used the following tables :
#### •	team_pitching_counts
#### •	boxscore
#### •	game
#### •	team_batting_counts
These tables provide valuable insights into team and player performance during the specified time frame.

## Data Preparation 
As part of the data preparation process, I joined four tables - team_pitching_counts, boxscore, game, and team_batting_counts - to create a comprehensive dataset that contains information on team and player performance in baseball games between 2007 and 2012. Additionally, I extracted the date from the local date field, which also includes time information. This step was necessary to ensure that the date field is consistent across all tables and can be used for analysis purposes.

## Feature Creation
Using the comprehensive dataset created during the data preparation phase, I performed rolling 100-day statistics to gain insight into team and player performance trends over time. The resulting rolling table was then used to create a set of features that would enable me to analyze and compare team and player performance during the specified time frame.
The Features Created are:
#### •	Hit_per_Strikeout_Ratio
#### •	Strikeout_to_walk_Ratio
#### •	Groundout_to_Flyout_Ratio
#### •	Walks_per_atBat_Ratio
#### •	Strikeout_per_atBat_Ratio
#### •	HR_H_Ratio
#### •	AB_HR_Ratio
#### •	plateApperance_per_Strikeout_Ratio
#### •	Strikeout_Ratio
#### •	Hit_By_Pitch_Ratio
#### •	TB_Ratio
#### •	HR_to_inning_Ratio
#### •	Batting_Average_Ratio
#### •	inning_Ratio
#### •	Strikeout_to_inning_Ratio

## Feature Analysis
To analyze the relationship between the predictors and response, I generated violin plots. These plots are a useful visualization tool for examining the distribution of data and identifying any potential patterns or trends.
<img width="510" alt="Picture1" src="https://github.com/LikithaBommasani/BDA-602/assets/112617127/ccba3c68-a8b7-4b43-b8d2-046476195aba">

<img width="510" alt="Picture2" src="https://github.com/LikithaBommasani/BDA-602/assets/112617127/30cb1396-a5e9-41fe-bd84-0ede582a2e51">

### P value and T score
Using the stats model’s package, I calculated the p-value and t-score for each feature. The p-value measures the statistical significance of each feature, with an ideal value typically below 0.05. The t-score measures the strength of the relationship between each feature and the response variable, with an ideal value typically greater than 2 or less than -2.

### Random Forest Importance
I used random forest variable importance to rank the variables and determine their importance in predicting the response variable. This analysis helps identify which variables have the greatest impact on the accuracy of the model.

### Correlation
To analyze the relationship between pairs of variables, I used a correlation matrix with Pearson correlation coefficients. This method measures the linear relationship between two variables, with a coefficient that ranges from -1 to 1. A coefficient of 1 indicates a perfect positive correlation, while a coefficient of -1 indicates a perfect negative correlation. By examining the correlation matrix, I can identify which pairs of variables have the strongest relationships and potential interactions, providing insight into the underlying patterns in the data.


## Models
To build machine learning models for prediction, I selected several different models, including. 
#### •	Random Forest
#### •	 Decision Tree Classifier
#### •	 Support Vector Machines (SVM)
#### •	Naive Bayes
#### •	 K-Nearest Neighbors. 
These models were chosen for their ability to handle various types of data and their potential to provide accurate predictions for the response variable.
The Training Data is set to Training data =< august 13, 2009
The Test Data is set to Test data > august 13, 2009











