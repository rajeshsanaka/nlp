from flask import Flask, request, render_template
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Initialize Flask app
app = Flask(__name__)

# Initialize VADER
sia = SentimentIntensityAnalyzer()

# Initialize RoBERTa
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Helper function for RoBERTa sentiment analysis
def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the review text from the form
    review_text = request.form['review']
    
    # VADER Analysis
    vader_result = sia.polarity_scores(review_text)
    
    # RoBERTa Analysis
    roberta_result = polarity_scores_roberta(review_text)
    
    # Combine results
    results = {
        "VADER": vader_result,
        "RoBERTa": roberta_result
    }
    
    return render_template('index.html', review=review_text, results=results)

if __name__ == '__main__':
    app.run(debug=True)
