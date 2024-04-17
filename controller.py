import json
import os

from flask import Flask, jsonify, request
import run

app = Flask(__name__)


@app.route("/chaty", methods=['POST'])
def chaty():
    request_data = request.get_json()  # {query": <QUERY>, "contetxt": [<CONTEXT>]}
    # chaty_request = request_data['chatyRequest']
    query = request_data['query']
    context = "\n".join(request_data['context'])
    answer = run.run_chaty(query, context)
    answer_json = {"answer": answer}
    return json.dumps(answer_json)

if __name__ == "__main__":
    port = 8080 #int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
