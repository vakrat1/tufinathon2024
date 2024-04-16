from flask import Flask, jsonify, request
import run

app = Flask(__name__)


@app.route("/chaty", methods=['POST'])
def chaty():
    request_data = request.get_json()  # {"query": <QUERY>, "contetxt": <CONTEXT>}
    query = request_data['query']
    context = request_data['context']
    answer = run.run_chaty(query, context)
    return answer

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
