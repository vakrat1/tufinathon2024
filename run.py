import os
import json
import logging
import datetime
import time
import yaml

import spotipy
from langchain.requests import Requests
from langchain import OpenAI

from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT

logger = logging.getLogger()


def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    os.environ["OPENAI_API_KEY"] = config['openai_api_key']
    os.environ["TMDB_ACCESS_TOKEN"] = config['tmdb_access_token']
    os.environ['SPOTIPY_CLIENT_ID'] = config['spotipy_client_id']
    os.environ['SPOTIPY_CLIENT_SECRET'] = config['spotipy_client_secret']
    os.environ['SPOTIPY_REDIRECT_URI'] = config['spotipy_redirect_uri']
    os.environ['TUFIN_BASIC_AUTH'] = config['tufin_basic_auth']
    # os.environ['TUFIN_SC_BEARER_AUTH'] = config['tufic_sc_bearer_auth']

        
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(ColorPrint())],
    )
    logger.setLevel(logging.INFO)

    # scenario = input("Please select a scenario (TMDB/Spotify): ")
    # scenario = scenario.lower()
    scenario = 'tufin'

    if scenario == 'tmdb':
        with open("specs/tmdb_oas.json") as f:
            raw_tmdb_api_spec = json.load(f)

        api_spec = reduce_openapi_spec(raw_tmdb_api_spec, only_required=False)

        access_token = os.environ["TMDB_ACCESS_TOKEN"]
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
    elif scenario == 'spotify':
        with open("specs/spotify_oas.json") as f:
            raw_api_spec = json.load(f)

        api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)

        scopes = list(raw_api_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
        access_token = spotipy.util.prompt_for_user_token(scope=','.join(scopes))
        headers = {
            'Authorization': f'Bearer {access_token}'
        }
    elif scenario == 'tufin':
        with open("specs/tufin_oas.json") as f:
            raw_api_spec = json.load(f)

        api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)

        # scopes = list(raw_api_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
        # access_token = spotipy.util.prompt_for_user_token(scope=','.join(scopes))
        tufin_basic_auth = os.environ["TUFIN_BASIC_AUTH"]
        headers = {
            'Authorization': f'Basic {tufin_basic_auth}'
        }
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")

    requests_wrapper = Requests(headers=headers)

    llm = OpenAI(model_name="gpt-3.5-turbo-0125", temperature=0.0, max_tokens=700)
    rest_gpt = RestGPT(llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=False)

    if scenario == 'tmdb':
        query_example = "Give me the number of movies directed by Sofia Coppola"
    elif scenario == 'spotify':
        query_example = "Add Summertime Sadness by Lana Del Rey in my first playlist"
    elif scenario == 'tufin':
        # query_example = "For traffic between src IP address 172.16.100.0/30 to destination IP address 10.200.0.0/24 on ANY service, get me if the traffic is blocked. If it is blocked take from the device info the id, name, type and vendor of this topology path and create an AccessRequest (AR) ticket with subject AR_TEST with workflow id 10 and workflow name AR with priority Normal. Take the target device from the device_info including its name and its management name. Use the source from the path calculation. Take the destination fopm the path calculation as well. Take also the service details from the path calculation parameters. Action Accept"
        query_example = "For traffic between source IP address 172.16.100.0/30 to destination IP address 10.200.0.0/24 on service tcp:8081, check if it is blocked. If it does block get the device_info of this topology path and open an AccessRequest with ticket subject 'TATATEST' ticket with wordflow id 10 and name AR with action Accept"
        # query_example = "Create AccessRequest (AR) ticket with subject YAYATEST2 with workflow id 10 and workflow name AR with priority Normal target device management FMG/SD-WAN and device name FortiGate_7_0_112_244-SD-WAN with Source address of IP 10.251.4.0/23 and Destination address of IP 10.252.0.0/22 on service TCP:80 and Action Accept"
        # query_example = "For traffic between src IP address 172.16.100.0/30 to destination IP address 10.200.0.0/24 on ANY service, get me if the traffic is allowed and if it blocked get me the device_info of this topology path"
        # query_example = "For traffic between src IP address 10.251.4.0/23 to destination IP address 10.252.0.0/22 on ANY service, get me if the traffic is allowed and if it blocked get me the device_info of this topology path"
        # query_example = "Please check if the traffic is allowed or blocked from Source IP 172.16.100.0/30 to destination IP 10.200.0.0/24 on service named SSH."# Return the result in JSON format"
        # query_example = "Please get me the list of all the devices where their virtual_type is management. Return in the results only their id, name, vendor, model and virtual_type in JSON format"
        # query_example = "Please get me the list of devices where their virtual_type is management. After that for each each such device use its id to fethch its revisions. Return the result in JSON format"
    print(f"Example instruction: {query_example}")

    

    start_time = time.time()
    # while True:
    query = input("Please input an instruction (Press ENTER to use the example instruction): ")
    if query == '':
        query = query_example
    # if query == 'DONE':
    #     break
    logger.info(f"Query: {query}")

    rest_gpt.run(query)
    logger.info(f"Execution Time: {time.time() - start_time}")

if __name__ == '__main__':
    main()
