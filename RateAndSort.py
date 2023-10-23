'''One of the most important problems in e-commerce is the correct calculation of the points given to the products
after the sale. The solution to this problem means more customer satisfaction for the e-commerce site,
product prominence for sellers and a smooth shopping experience for buyers. Another problem is the correct ranking of
the comments given to the products. Since the prominence of misleading reviews will directly affect the sales of the
product, it will cause both financial loss and customer loss. In solving these 2 basic problems, e-commerce sites and
sellers will increase their sales while customers will complete their purchasing journey smoothly.'''

#
# Variables
# reviewerID: User ID
# asin Product ID
# reviewerName: Username
# helpful: Useful evaluation rating
# reviewText: Review
# overall: Product rating
# summary: Evaluation summary
# unixReviewTime: Review time
# reviewTime: Review time Raw
# day_diff: Number of days since evaluation
# helpful_yes: Number of times the evaluation was found helpful
# total_vote: Number of votes for evaluation

"'''Task 1: Calculate the Average Rating based on current reviews and compare it with the existing average rating.'''"

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df= pd.read_csv("/Users/ataberk/Desktop/Miuul Bootcamp/week 4/Rating Product&SortingReviewsinAmazon/amazon_review.csv")


'''Step1: Calculate the average score of the product.
Step 2: Calculate the average weighted score by date.
Step 3: Compare and interpret the average of each time period in weighted scoring.'''
df["overall"].mean()
#4.587589013224822
df.head(5)

df.columns
df["reviewTime"]= pd.to_datetime(df["reviewTime"], dayfirst= True)
df.info()
current_date = df["reviewTime"].max()

df["days"] = (current_date -df["reviewTime"]).dt.days

q1= df["days"].quantile(0.25)
q2= df["days"].quantile(0.50)
q3= df["days"].quantile(0.75)

def time_based_weighted_average(dataframe, w1=18, w2=22, w3=28, w4=32):
    return dataframe.loc[(dataframe["days"] <= q1), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > q1) & (dataframe["days"] <= q2), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > q2) & (dataframe["days"] <= q3), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > q3),"overall"].mean() * w4 / 100


time_based_weighted_average(df)
#4.568060108268714


'''Task 2: Identify 20 reviews that will be displayed on the product detail page for the product.'''
'''Step 1: Generate the helpful_no variable.
- total_vote is the total number of up-down votes given to a comment.
- up means helpful.
There is no helpful_no variable in the dataset, it needs to be generated from existing variables.
- Find the number of helpful votes (helpful_no) by subtracting the number of helpful votes (helpful_yes) from the total number of votes (total_vote).'''

df["helpful_no"] =df["total_vote"] - df["helpful_yes"]
df.head()

'''Step 2: Calculate score_pos_neg_diff, score_average_rating and wilson_lower_bound scores and add them to the data.
- Define the score_pos_neg_diff, score_average_rating and wilson_lower_bound functions to calculate the score_pos_neg_diff, score_average_rating and wilson_lower_bound scores.
- Create scores according to score_pos_neg_diff. Then save them in df under the name score_pos_neg_diff.
- Create scores according to score_average_rating. Then save in df with the name score_average_rating.
- Create scores according to wilson_lower_bound. Then save as wilson_lower_bound in df.'''

def score_pos_neg_diff(up, down):
    return up - down
df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
# score_average_rating
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
# wilson_lower_bound
def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
df["wilson_lower_bound"] = df.apply(lambda x:
wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

'''Step 3: Identify 20 Comments and Interpret the results.
- Identify and rank the top 20 comments according to wilson_lower_bound.
- Interpret the results.'''

df[["overall","helpful_yes","total_vote","days","helpful_no", "score_pos_neg_diff", "score_average_rating","wilson_lower_bound"]]\
.sort_values("wilson_lower_bound", ascending= False).head(20)