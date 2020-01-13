# Fine-grained Sentiment Analysis
# Very Positive = 5 stars and Very Negative = 1 star
from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

app = Flask(__name__)


def score_analyze(score):
    if 0.0 < score < 0.25:
        return "very bad"
    elif 0.25 <= score < 0.50:
        return "bad"
    elif 0.50 <= score < 0.75:
        return "good"
    elif 0.75 <= score <= 1.0:
        return "very good"


def score_analyze_malaya(score, category):
    if category == "positive":
        sentiment = "good"
    else:
        sentiment = "bad"

    if 0.50 <= score < 0.75:
        return sentiment
    elif 0.75 <= score <= 1.0:
        return "very " + sentiment


def lang_detection(feedback):
    return TextBlob(feedback).detect_language()


def sentiment_malay(feedback):
    import malaya
    import operator

    model = malaya.sentiment.transformer(model='xlnet', size='base')
    # model = malaya.sentiment.multinomial()
    sentiment_strength = model.predict(feedback,get_proba=True,add_neutral=True)
    print(sentiment_strength)
    if sentiment_strength['neutral'] > sentiment_strength['negative'] and \
            sentiment_strength['neutral'] > sentiment_strength['positive']:
        return "Neutral"

    sentiment_category = max(sentiment_strength.items(), key=operator.itemgetter(1))[0]
    malaya.clear_cache('sentiment/xlnet/base')
    return score_analyze_malaya(sentiment_strength[sentiment_category], sentiment_category)


def sentiment_english(feedback):
    sentiment_strength = SentimentIntensityAnalyzer().polarity_scores(feedback)
    print(sentiment_strength)
    if sentiment_strength['neu'] > sentiment_strength['neg'] and sentiment_strength['neu'] > sentiment_strength['pos']:
        return "Neutral"
    return score_analyze(sentiment_strength['compound'])


def sentiment_chinese(feedback):
    # using TextBlob to perform translation from zh-CN to En
    translated_feedback = str(TextBlob(feedback).translate(from_lang="zh-CN", to="en"))
    sentiment_strength = SentimentIntensityAnalyzer().polarity_scores(translated_feedback)
    print(sentiment_strength)
    if sentiment_strength['neu'] > sentiment_strength['neg'] and sentiment_strength['neu'] > sentiment_strength['pos']:
        return "Neutral"
    return score_analyze(sentiment_strength['compound'])


@app.route('/api/sentiment', methods=['POST'])
def sentiment_api():
    if request.form['feedback'] == '':
        return jsonify({"error": "No feedback found" })

    feedback = request.form['feedback']

    lang = lang_detection(feedback)
    print("Feedback: " + feedback)
    print("Language " + lang)

    if lang == "en":
        return sentiment_english(feedback)
    elif lang == "ms":
        return sentiment_malay(feedback)
    elif lang == "zh-CN":
        return sentiment_chinese(feedback)


@app.route('/')
# @app.route('/api/sentiment', methods=['POST'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()