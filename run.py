import os
import json
import logging
import time
import yaml

from langchain.requests import Requests
from langchain import OpenAI

from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT

logger = logging.getLogger()

def initialize_logging():
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint())],
    )
    logger.setLevel(logging.INFO)

def load_configuration():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ['TUFIN_BASIC_AUTH'] = config['tufin_basic_auth']
    return config

def initialize_api_scenario():
    with open("specs/tufin_oas.json") as f:
        raw_api_spec = json.load(f)
    api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)
    return api_spec

def setup_scenario(api_spec):
    headers = {'Authorization': f'Basic {os.environ["TUFIN_BASIC_AUTH"]}'}
    requests_wrapper = Requests(headers=headers)
    llm = OpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0, max_tokens=700)
    return RestGPT(llm, api_spec=api_spec, scenario='tufin', requests_wrapper=requests_wrapper, simple_parser=False)

def run_chaty(prompt, context, rest_gpt):
    try:
        if prompt.lower().startswith("what did we do before"):
            logger.info("Go to Swagger API, here is the link: https://192.168.32.84/securetrack/apidoc/ and find between all the APIs something that will be relevant")
            return "Go to Swagger API, here is the link: https://192.168.32.84/securetrack/apidoc/ and find between all the APIs something that will be relevant"
        if prompt.lower().startswith("what do we do now?"):
            logger.info("Ask me any API question and I will solve all your problems!")
            return "Ask me any API question and I will solve all your problems!"
        if prompt.lower().startswith("hi, how are you"):
            logger.info("Hello, im here to assist you with APIs how can I help?")
            return "hello, im here to assist you with APIs how can i help?"
        if prompt.lower().startswith("whats the weather"):
            logger.info("Im not qualified to answer this question, you can teach me the API for weather")
            return "Im not qualified to answer this question, you can teach me the API for weather"
        if prompt.lower().startswith("what else cam you do"):
            logger.info("I can show the world, the tos API world")
            return "I can show the world, the tos API world"
        full_query = f"Previous conversations: {context} User question: {prompt}"
        logger.info(f"Query: {full_query}")

        answer = rest_gpt.run(full_query)
        logger.info(f"Answer: {answer}")
        logger.info(f"Execution Time: {time.time() - time.time()}")

        return answer
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return "Encountered an error, can you ask the question again? Try to be specific."

def main():
    initialize_logging()
    config = load_configuration()
    api_spec = initialize_api_scenario()
    rest_gpt = setup_scenario(api_spec)

    history = []
    while True:
        prompt = input("Please input an instruction (Press ENTER to use the example instruction): ")
        context = " ".join(history[-10:])  # Using the last 10 interactions for context
        answer = run_chaty(prompt, context, rest_gpt)
        history.extend([prompt, answer])
        if answer:
            logger.info(f"Answer: {answer}")

if __name__ == '__main__':
    main()
