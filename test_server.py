from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/test", methods=["GET", "POST"])
def test():
    return jsonify({"status": "ok", "message": "Test server is running"})

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "no message")
    return jsonify({"answer": f"You said: {msg}"})

if __name__ == '__main__':
    print("Starting test server...")
    app.run(host="0.0.0.0", port=8080, debug=False)
