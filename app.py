from flask import Flask, render_template, request, jsonify
import pickle
import json
import random

app = Flask(__name__)

# Load the trained model and vectorizer
with open('model/chatbot_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load the intents data
with open('dataset/intents1.json', 'r') as f:
    intents = json.load(f)

def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]

    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            break
    else:
        response = "Sorry, I didn't understand that."

    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
# # from chatbot import CB
# # from flask import Flask, render_template, request

# # app = Flask(__name__)

# # @app.route("/chatbot")
# # def home():
# #     return render_template("index.html")

# # @app.route("/get")
# # def get_bot_response():
# #     userText = request.args.get('msg')
# #     return str(CB.get_response(userText))
# # app.run(debug = True)
# from flask import Flask, render_template, request
# import pickle
# import json
# import random

# app = Flask(__name__)

# # Load the trained model and vectorizer
# with open('model/chatbot_model.pkl', 'rb') as f:
#     best_model = pickle.load(f)

# with open('model/vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)

# # Load the intentsdata
# with open('dataset/intents1.json', 'r') as f:
#     intents = json.load(f)

# def chatbot_response(user_input):
#     input_text = vectorizer.transform([user_input])
#     predicted_intent = best_model.predict(input_text)[0]

#     for intent in intents['intents']:
#         if intent['tag'] == predicted_intent:
#             response = random.choice(intent['responses'])
#             break

#     return response

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.form['user_input']
#     response = chatbot_response(user_input)
#     return response

# if __name__ == '__main__':
#     app.run(debug=True)