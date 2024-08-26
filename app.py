from flask import Flask, request, jsonify, render_template
import json
import random
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from textblob import TextBlob
from flask_cors import CORS  # Only needed if CORS issues arise
import logging
from typing import Tuple

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests if needed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model and intents
def load_model_and_intents() -> Tuple[Pipeline, dict]:  
    """Load the trained model and intents from files."""
    try:
        with open('chatbot_advanced_model.pkl', 'rb') as model_file:
            pipeline = pickle.load(model_file)
        with open('intents.json', 'r') as file:
            intents = json.load(file)
        return pipeline, intents
    except Exception as e:
        logger.error("Error loading model or intents: %s", e)
        raise

pipeline, intents = load_model_and_intents()

def correct_spelling(user_input: str) -> str:
    """Use TextBlob for spelling correction."""
    try:
        corrected_input = str(TextBlob(user_input).correct())
        return corrected_input
    except Exception as e:
        logger.warning("Spelling correction failed: %s", e)
        return user_input  # Return original input if correction fails

def preprocess_input(user_input: str) -> str:
    """Preprocess the user input by correcting spelling and handling specific keywords."""
    logger.info(f"Original input: {user_input}")

    # Preserve the original input case
    original_input = user_input

    # Convert input to lowercase for consistent processing
    user_input_lower = user_input.lower()

    # Block or ignore specific phrases
    if "im a" in user_input_lower:
        logger.info(f"Blocked input detected: {user_input_lower}")
        return "Sorry, I cannot process that request."

    # Correct spelling
    user_input = correct_spelling(user_input)
    logger.info(f"After spelling correction: {user_input}")

    # Define replacements for common errors or known keywords
    replacements = {
        'lip stick': 'lipstick',
        'lipbalm': 'lip balm',
        'lipstick': 'Lipstick details',
        'lipscrub': 'lip scrub',
        'types of': 'type of',
        'lipserum': 'lip serum',
        'kajal': 'KAJAL',  # Replace "kajal" with "KAJAL"
        # Add more replacements as needed
    }

    # Replace specific keywords in the input
    for key, value in replacements.items():
        if key in user_input_lower:
            user_input = user_input_lower.replace(key, value)

    # Handle specific uppercase cases like "KAJAL"
    if "KAJAL" in original_input:
        user_input = user_input.replace("kajal", "KAJAL")

    logger.info(f"After replacements: {user_input}")
    return user_input


def get_response(user_input: str) -> str:
    """Generate a response based on the user's input."""
    user_input = preprocess_input(user_input)
    logger.info(f"Processed input: {user_input}")
    # Handle "bye" input explicitly
    if user_input.lower() == "bye":
        return "Thank you for your visit."

    # Single word responses for specific cases
    if len(user_input.split()) == 1:
        logger.info(f"Single word input detected: {user_input.lower()}")
        if user_input.lower() not in ['what', 'kpm', 'lipstick', 'lipbalm', 'oil', 'soap', 'hi', 'hello', 'helo', 'hai', 'name', 'hey', 'location', 'kajal', 'im a', 'Bye']:
            logger.info(f"Input not in single-word list: {user_input.lower()}")
            return "Please provide more details so I can assist you better."

    try:
        # Predict the intent tag
        tag = pipeline.predict([user_input])[0]
        probabilities = pipeline.predict_proba([user_input])[0]

        # Set a confidence threshold for predictions
        threshold = 0.1
        if np.max(probabilities) < threshold:
            unknown_responses = [intent["responses"] for intent in intents["intents"] if intent["tag"] == "unknown"][0]
            return random.choice(unknown_responses)
        
        # Return a response based on the predicted intent
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
        
        return "I'm not sure how to respond to that. Can you please provide more details?"

    except Exception as e:
        logger.error("Error generating response: %s", e)
        return "Something went wrong, please try again later."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle the chat POST request."""
    try:
        user_input = request.json.get('message')
        if not user_input:
            return jsonify({'response': 'Please provide a message.'})
        response = get_response(user_input)
        return jsonify({'response': response})
    except Exception as e:
        logger.error("Error in /chat route: %s", e)
        return jsonify({'response': 'An error occurred while processing your request.'})

# Start the Flask app
if __name__ == "__main__":
    app.run(port=5000, debug=True)
