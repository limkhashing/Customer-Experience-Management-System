##========================================================================================
# if want to do sentiment analysis on chinese
# workaround is translate to English with TextBlob
# and analyze it with TextBlob or NLTK.vader

# Heavily depends on the context to know whether it is positive or negative sentiment
# 1. Polarity: if the speaker express a positive or negative opinion
# 2. Subjectivity: the thing that is being talked about

# What did you like about the event?
# What did you dislike about the event?

# Answer of these can change the sentiment
# Everything of it
# Absolutely nothing!

##========================================================================================
# Download NLTK data

# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('vader_lexicon')
# nltk.download('movie_reviews')

##========================================================================================
# NLTK

# from nltk.sentiment.vader import SentimentIntensityAnalyzer
#
# sentences = ["VADER is smart, handsome, and funny.",  # positive sentence example
#              "VADER is smart, handsome, and funny!",  # punctuation emphasis handled correctly (sentiment intensity adjusted)
#              "VADER is very smart, handsome, and funny.", # booster words handled correctly (sentiment intensity adjusted)
#              "VADER is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
#              "VADER is VERY SMART, handsome, and FUNNY!!!", # combination of signals - VADER appropriately adjusts intensity
#              "VADER is VERY SMART, uber handsome, and FRIGGIN FUNNY!!!", # booster words & punctuation make this close to ceiling for score
#              "VADER is not smart, handsome, nor funny.",  # negation sentence example
#              "The book was good.",  # positive sentence
#              "At least it isn't a horrible book.",  # negated negative sentence with contraction
#              "The book was only kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
    #              "The plot was good, but the characters are uncompelling and the dialog is not great.", # mixed negation sentence
#              "Today SUX!",  # negative slang with capitalization emphasis
#              "Today only kinda sux! But I'll get by, lol", # mixed sentiment example with slang and constrastive conjunction "but"
#              "Make sure you :) or :D today!",  # emoticons handled
#              "Catch utf-8 emoji such as such as 💘 and 💋 and 😁",  # emojis handled
#              "Not bad at all"  # Capitalized negation
#              ]
# analyzer = SentimentIntensityAnalyzer()
# for sentence in sentences:
#     vs = analyzer.polarity_scores(sentence)
#     print("{:-<65} {}".format(sentence, str(vs)))

# analyzer = SentimentIntensityAnalyzer()
# vs = analyzer.polarity_scores(feedback)
# print(vs)
# print(vs['compound'])
# print(vs[max(vs.items(), key=operator.itemgetter(1))[0]])
##========================================================================================
# textblob

# from textblob import TextBlob
# blob = TextBlob(feedback)
# translated_blob = str(blob.translate(from_lang="zh-CN", to="en"))
# final_blob = TextBlob(translated_blob)
# print(blob.sentiment_assessments)
# print(blob.sentiment.polarity)
# print(blob.sentiment.subjectivity)

##========================================================================================
from textblob import TextBlob

feedback = "I am not very satisfied with the food"
score = TextBlob(feedback).polarity

if score < -0.50:
    sentiment = "Very unsatisfied", 1
elif score < 0.00 and score >= -0.50:
    sentiment = "Unsatisfied", 2
elif score >= 0.00 and score <= 0.10:
    sentiment = "Neutral", 3
elif score > 0.10 and score <= 0.50:
    sentiment = "Satisfied", 4
else:
    sentiment = "Very satisfied", 5

print(sentiment)

