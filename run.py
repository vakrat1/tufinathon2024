import os
import json
import logging
import datetime
import time
import yaml

from langchain.requests import Requests
from langchain import OpenAI

from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT

logger = logging.getLogger()


def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ['TUFIN_BASIC_AUTH'] = config['tufin_basic_auth']
        
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint())],
    )
    logger.setLevel(logging.INFO)

    # scenario = input("Please select a scenario (TMDB/Spotify): ")
    # scenario = scenario.lower()
    scenario = 'tufin'

    with open("specs/tufin_oas.json") as f:
        raw_api_spec = json.load(f)

    api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)

    tufin_basic_auth = os.environ["TUFIN_BASIC_AUTH"]
    headers = {
        'Authorization': f'Basic {tufin_basic_auth}'
    }

    requests_wrapper = Requests(headers=headers)

    llm = OpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0, max_tokens=700)
    # rest_gpt = RestGPT(llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=False)

    query_example = "For traffic between source IP address 172.16.100.0/30 to destination IP address 10.200.0.0/24 on service tcp:8081, check if it is blocked. If it does block get the device_info of this topology path and open an AccessRequest with ticket subject 'TATATEST' ticket with wordflow id 10 and name AR with action Accept"

    history = []  # Initialize the conversation history
    while True:
        rest_gpt = RestGPT(llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=False)
        start_time = time.time()

        # Gather the complete conversation history as context for the new query
        context = " ".join(history[-10:])  # Using the last 10 interactions for context
        prompt = input("Please input an instruction (Press ENTER to use the example instruction): ")
        if prompt == '':
            prompt = query_example

        # Concatenate context with the new query
        if context:
            full_query = f"Previous conversions {context} User question: {prompt}"
        else:
            full_query = prompt

        logger.info(f"Query: {full_query}")

        # Send the full query with context to the model
        answer = rest_gpt.run(full_query)
        logger.info(f"Answer: {answer}")
        logger.info(f"Execution Time: {time.time() - start_time}")

        # Append the latest interaction to the history
        history.extend([prompt, answer])

if __name__ == '__main__':
    main()
