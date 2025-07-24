from flask import Flask, request, jsonify
from flask_cors import CORS  # ✅ import CORS
from model_utils import chat_with_bot

app = Flask(__name__)
CORS(app)  # ✅ enable CORS

user_location = {'latitude': 48.8566, 'longitude': 2.3522}


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    lat = request.json.get("latitude", user_location["latitude"])
    lon = request.json.get("longitude", user_location["longitude"])

    response, updated_coords = chat_with_bot(user_input, lat, lon)
    user_location["latitude"], user_location["longitude"] = updated_coords

    return jsonify({
        "reply": response,
        "latitude": user_location["latitude"],
        "longitude": user_location["longitude"]
    })


if __name__ == "__main__":
    app.run(debug=True)
