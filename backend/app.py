from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# # Simulated user data (replace with your user database logic)
users = {
    'user1': {'password': 'password1'},
    'user2': {'password': 'password2'},
    'user3': {'password': 'password3'},
    'user4': {'password': 'password4'}
}

locations_data = {
    'id1': {'name': 'Farmacia Pescarusul'},
    'id2': {'name': 'Restaurant Catena'},
    'id3': {'name': 'Muzeul Ursilor'},
}

review_data = {
    'review_id1': {'name': 'user1', 'location': 'Farmacia Pescarusul', 'rating': '3'},
    'review_id2': {'name': 'user2', 'location': 'Restaurant Catena', 'rating': '3'},
    'review_id3': {'name': 'user3', 'location': 'Restaurant Catena', 'rating': '4'},
    'review_id4': {'name': 'user4', 'location': 'Muzeul Ursilor', 'rating': '2'}
}

current_user = None

def compute_recommendation(user):
    return 'Farmacia Pescarusul'

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get('username')
    current_user = username

    credentials = users.get(username)
    if credentials:
        return jsonify({'message': 'Login Successful'}), 200
    else:
        return jsonify({'message': 'Invalid credentials'}), 401

@app.route("/locations", methods=["GET"])
def locations():
    return jsonify(locations_data), 200

@app.route("/reviews", methods=["POST"])
def reviews():
    data = request.get_json()
    username = data.get('username')
    location = data.get('location')
    rating = data.get('rating')

    review_data['review_id' + str(len(review_data) + 1)] = {'name': username, 'location': location, 'rating': rating}
    return jsonify({'message': 'Inserted data successfully'}), 200

@app.route("/recommendation", methods=["GET"])
def recommendation():
    return jsonify(compute_recommendation(current_user)), 200

if __name__ == '__main__':
    app.run(debug=True)