# Fine-grained Sentiment Analysis
# Very Positive = 5 stars and Very Negative = 1 star
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)


def score_analyze(score):
    if score < 0.0 or 0.0 < score < 0.25:
        return "Very unsatisfied", 1
    elif 0.25 <= score < 0.50:
        return "Unsatisfied", 2
    elif 0.50 <= score < 0.75:
        return "Satisfied", 4
    elif 0.75 <= score <= 1.0:
        return "Very satisfied", 5


def lang_detection(feedback):
    return TextBlob(feedback).detect_language()


def sentiment_malay(feedback):
    # using TextBlob to perform translation from ms to En
    translated_feedback = str(TextBlob(feedback).translate(from_lang="ms", to="en"))
    sentiment_strength = SentimentIntensityAnalyzer().polarity_scores(translated_feedback)
    print(sentiment_strength)
    if sentiment_strength['neu'] > sentiment_strength['neg'] and sentiment_strength['neu'] > sentiment_strength['pos']:
        return "Neutral", 3
    return score_analyze(sentiment_strength['compound'])


def sentiment_english(feedback):
    sentiment_strength = SentimentIntensityAnalyzer().polarity_scores(feedback)
    print(sentiment_strength)
    if sentiment_strength['neu'] > sentiment_strength['neg'] and sentiment_strength['neu'] > sentiment_strength['pos']:
        return "Neutral", 3
    return score_analyze(sentiment_strength['compound'])


def sentiment_chinese(feedback):
    # using TextBlob to perform translation from zh-CN to En
    translated_feedback = str(TextBlob(feedback).translate(from_lang="zh-CN", to="en"))
    sentiment_strength = SentimentIntensityAnalyzer().polarity_scores(translated_feedback)
    print(sentiment_strength)
    if sentiment_strength['neu'] > sentiment_strength['neg'] and sentiment_strength['neu'] > sentiment_strength['pos']:
        return "Neutral", 3
    return score_analyze(sentiment_strength['compound'])


def init():
    import nltk

    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('punkt')
        nltk.download('vader_lexicon')


@app.route('/',  methods=['GET', 'POST'])
# @app.route('/api/sentiment', methods=['POST'])
def index():
    init()

    if request.method == 'POST':
        if request.form['feedback'] == '':
            return jsonify({"error": "No feedback found"})

        feedback = request.form['feedback']

        lang = lang_detection(feedback)
        print("Feedback: " + feedback)
        print("Language " + lang)

        sentiment = None
        rating = None
        if lang == "en":
            sentiment, rating = sentiment_english(feedback)
        elif lang == "ms":
            sentiment, rating = sentiment_malay(feedback)
        elif lang == "zh-CN":
            sentiment, rating = sentiment_chinese(feedback)
        return render_template('index.html', feedback=feedback, sentiment=sentiment, rating=rating)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()