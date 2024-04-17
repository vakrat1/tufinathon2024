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

    # Search for the 'Final Answer:' in the response
    answer_marker = "Final Answer:"
    if answer_marker in answer:
        # Split the answer on 'Final Answer:' and take the second part
        final_answer = answer.split(answer_marker, 1)[1].strip()
    else:
        # If 'Final Answer:' is not found, set a default message
        final_answer = answer

    # Create a JSON object with the final answer
    answer_json = {"answer": final_answer}
    return json.dumps(answer_json)

if __name__ == "__main__":
    port = 8080 #int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)
