#######################################################################################
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
#######################################################################################


# using ntlk / textblob

## ntlk
# import nltk
# nltk.download('vader_lexicon')

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

##========================================================================================

## Textblob
# import nltk
# nltk.download('punkt')

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# feedback = '想告诉你们，其实重新开始并不可怕，可怕的是明知道自己想要什么却迟迟不敢去行动！清楚自己要什么、选择自己想要的，这些都是你能保持热忱投入工作和生活的原因。'
# feedback = '对于该医院的服务，我感到很满意也很开心'
# feedback = '对于该医院的服务，我感到很满意。虽然有些能改善的地方（如厕所食物等等），可是还是服务周到'
feedback = "VADER is very smart, handsome, and funny"

blob = TextBlob(feedback)
# translated_blob = str(blob.translate(from_lang="zh-CN", to="en"))
# final_blob = TextBlob(translated_blob)

print(blob.sentiment_assessments)
print(blob.sentiment.polarity)
print(blob.sentiment.subjectivity)

analyzer = SentimentIntensityAnalyzer()
vs = analyzer.polarity_scores(feedback)
print(vs)
print(vs['compound'])

# print(vs[max(vs.items(), key=operator.itemgetter(1))[0]])
