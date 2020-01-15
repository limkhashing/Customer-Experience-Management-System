# Fine-grained Sentiment Analysis
# Very Positive = 5 stars and Very Negative = 1 star
from flask import Flask, render_template, request, jsonify
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

app = Flask(__name__)


def score_analyze(score):
    if score < -0.50:
        return "Very unsatisfied", 1
    elif 0.00 > score >= -0.50:
        return "Unsatisfied", 2
    elif 0.00 <= score <= 0.10:
        return "Neutral", 3
    elif 0.10 < score <= 0.50:
        return "Satisfied", 4
    else:
        return "Very satisfied", 5


def lang_detection(feedback):
    return TextBlob(feedback).detect_language()


def init():
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('punkt')
        nltk.download('vader_lexicon')


def sentiment_analysis(feedback, lang):
    if lang == "ms":
        feedback = str(TextBlob(feedback).translate(from_lang="ms", to="en"))
    elif lang == "zh-CN":
        feedback = str(TextBlob(feedback).translate(from_lang="zh-CN", to="en"))

    sentiment_strength = SentimentIntensityAnalyzer().polarity_scores(feedback)

    if sentiment_strength['compound'] != 0.00:
        print("NLTK: ", sentiment_strength)
        return score_analyze(sentiment_strength['compound'])
    else:
        blob = TextBlob(feedback)
        print("Textblob: ", blob.sentiment_assessments)
        return score_analyze(blob.polarity)


@app.route('/api/sentiment',  methods=['POST'])
def sentiment_api():
    init()
    
    if request.form['feedback'] == '':
        return jsonify({"error": "No feedback found"})

    feedback = request.form['feedback']

    lang = lang_detection(feedback)
    print("Feedback: " + feedback)
    print("Language " + lang)

    sentiment, rating = sentiment_analysis(feedback, lang)

    return jsonify({"sentiment": sentiment, "rating": rating})


@app.route('/',  methods=['GET', 'POST'])
def index():
    init()

    if request.method == 'POST':
        if request.form['feedback'] == '':
            return jsonify({"error": "No feedback found"})

        feedback = request.form['feedback']

        lang = lang_detection(feedback)
        print("Feedback: " + feedback)
        print("Language " + lang)

        sentiment, rating = sentiment_analysis(feedback, lang)

        return render_template('index.html', feedback=feedback, sentiment=sentiment, rating=rating)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
