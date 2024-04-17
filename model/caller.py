import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from copy import deepcopy
import yaml
import time
import re
import requests
import os

import tiktoken

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.requests import RequestsWrapper
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM

from utils import simplify_json, get_matched_endpoint, ReducedOpenAPISpec, fix_json_error
from .parser import ResponseParser, SimpleResponseParser

from langchain.requests import Requests

logger = logging.getLogger(__name__)




CALLER_PROMPT = """You are an agent that gets a sequence of API calls and given their documentation, should execute them and return the final response.
If you're not able to resolve an API call, you should retry the API call again. 
The response is in a JSON format.
If the response is not a legal JSON format, perform the API call again.
You should use the entire JSON response. Do not truncate the response and do not limit the resposne.
When interacting with API objects, you should extract ids for inputs to other API calls but return all the data fields as outputs returned to the User.
If you cannot complete them and run into issues, you should explain the issue and request the missing data. 
Your task is to complete the corresponding api calls according to the plan.
You should filter the results based on the query you have been given. If there is a paramter in the query you should filter the results based on that parameter.


Here is documentation on the API:
Base url: {api_url}
Endpoints:
{api_docs}

If the API path contains "{{}}", it means that it is a variable and you should replace it with the appropriate value. For example, if the path is "/users/{{user_id}}/tweets", you should replace "{{user_id}}" with the user id. "{{" and "}}" cannot appear in the url.

You can use http request method, i.e., GET, POST, DELETE, PATCH, PUT, and generate the corresponding parameters according to the API documentation and the plan.
The input should be a JSON string which has 3 base keys: url, description, output_instructions
The value of "url" should be a string.
The value of "description" should describe what the API response is about. The description should be specific.
The value of "output_instructions" should be instructions on what information to extract from the response, for example the id(s) for a resource(s) that the POST request creates. Note "output_instructions" MUST be natural language and as verbose as possible! It cannot be "return the full response". Output instructions should faithfully contain the contents of the api calling plan and be as specific as possible. The output instructions can also contain conditions such as filtering, sorting, etc.
If you are using GET method, add "params" key, and the value of "params" should be a dict of key-value pairs.
If you are using POST, PATCH or PUT methods, add "data" key, and the value of "data" should be a dict of key-value pairs. 
When invoking a POST API map the fields in the "data" element to the corresponded fields as describe in the API schema 
Remember to add a comma after every value except the last one, ensuring that the overall structure of the JSON remains valid.

Example 1:
Operation: GET
Input: {{
    "url": "https://192.168.32.84/securetrack/api/devices.json",
    "description": "The API response is the list of all the devices ,monitored by Securetrack)",
    "output_instructions": "Filter the API response by the query given by the user. For example, filter the results by virtual_type=management"
}}

Example 2:
Operation: GET
Input: {{
    "url": "https://192.168.32.84/securetrack/api/topology/path,json?src=29.29.29.1/24&dst=25.25.25.1/32&service=Facebook,tcp:80",
    "params": {{
        "src": "29.29.29.29.1/24",
        "dct": "25.25.25.1/32",
        "service": "Facebook,tcp:80"
    }},
    "description": "Check if the traffic is allowed on the path between the source 29.29.29.29/24 to destination 25.25.25.1/32 and service Facebook on tcp:80"
    "output_instructions": "Filter the API response by the extracting the field traffic_allowed and the device_info fields. For example, traffic_allowed: false"
}}

Example 3:
Operation: GET
Input: {{
    "url": "https://192.168.32.84/securechangeworkflow/api/securechange/workflows/active_workflows.json",
    "description": "The API response is the list of all the active workflows in securechange",
    "output_instructions": "return the list of the workflows from the API response "
}}

Example 4:
Operation: POST
Input: {{
    "url": "https://192.168.32.84/securechangeworkflow/api/securechange/tickets.json",
    "data": {{
         "subject": "demo ticket",
         "priority": "Normal",
         "workflow.id": 10,
         "workflow.name": AR,
         "steps.step.name": "Open request",
         "steps.step.tasks.task.fields.field.@xsi.type": "multi_access_request",
         "steps.step.tasks.task.fields.field.name": "ar",
         "steps.step.tasks.task.fields.field.access_request.use_topology": true,
         "steps.step.tasks.task.fields.field.access_request.targets.target.@type":"Object",
         "steps.step.tasks.task.fields.field.access_request.targets.target.object_name": "FortiGate_7_0_112_244-SD-WAN",
         "steps.step.tasks.task.fields.field.access_request.targets.target.object_name": "FMG/SD-WAN",
    }},
    "description": "The API response with a 201 HTTP response code with empty body",
    "output_instructions": "Dont try to parse the response's body, Just check if the HTTP response code is 201 "
}}


Example 8:
Operation: POST
Input: {{
    "url": "https://192.168.32.84/securetrack/api/topology/generic/interface.json",
    "data": {{
            "GenericInterfaces" : [{{
            "mgmtId": "1",
            "name": "Bob1",
            "ip": "100.100.45.55",
            "mask": "255.255.0.0",
            "vrf": "",
            "mpls": false,
            "unnumbered": false,
            "type": "external"    
            }}]   
            
    }},
    "description": "The API response with a 200 HTTP response code with empty body, Add an inner generic interface with mgmtId 1,name "Bob1",ip "100.100.45.55" mask "255.255.0.0",vrf "" and other parameters as specified in request body, The API response with a 200 HTTP response code with empty body",
    "output_instructions": "Dont try to parse the response's body, Just check if the HTTP response code is 200 "
}}

I will give you the background information and the plan you should execute.
Background: background information which you can use to execute the plan, e.g., the id of a person.
Plan: the plan of API calls to execute

You should execute the plan faithfully and give the Final Answer as soon as you successfully call the planned APIs, don't get clever and make up steps that don't exist in the plan. Do not make up APIs that don't exist in the plan. For example, if the plan is "GET /search/person to search for the director "Lee Chang dong"", do not call "GET /person/{{person_id}}/movie_credits" to get the credit of the person.

Starting below, you must follow this format:

Background: background information which you can use to execute the plan, e.g., the id of a workflow.
Plan: the plan of API calls to execute
Thought: you should always think about what to do
Operation: the request method to take, should be one of the following: GET, POST, DELETE, PATCH, PUT
Input: the input to the operation
Response: the output of the operation
Thought: I am finished executing the plan (or, I cannot finish executing the plan without knowing some other information.)
Execution Result: based on the API response, the execution result of the API calling plan.

The execution result should satisfy the following conditions:
1. The execution result must contain "Execution Result:" prompt.
2. You should reorganize the response into natural language based on the plan. For example, if the plan is "GET /search/person to search for the director "Lee Chang dong"", the execution result should be "Successfully call GET /search/person to search for the director "Lee Chang dong". The id of Lee Chang dong is xxxx". Do not use pronouns if possible. For example, do not use "The id of this person is xxxx".
3. If the plan includes expressions such as "most", you should choose the first item from the response. For example, if the plan is "GET /trending/tv/day to get the most trending TV show today", you should choose the first item from the response.
4. The execution result should be natural language and as verbose as possible. It must contain the information needed in the plan.

Begin!

Background: {background}
Plan: {api_plan}
Thought: {agent_scratchpad}
"""



class Caller(Chain):
    llm: BaseLLM
    api_spec: ReducedOpenAPISpec
    scenario: str
    requests_wrapper: RequestsWrapper
    max_iterations: Optional[int] = 15
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "force"
    simple_parser: bool = False
    with_response: bool = False
    output_key: str = "result"


    def __init__(self, llm: BaseLLM, api_spec: ReducedOpenAPISpec, scenario: str, requests_wrapper: RequestsWrapper, simple_parser: bool = False, with_response: bool = False) -> None:
        super().__init__(llm=llm, api_spec=api_spec, scenario=scenario, requests_wrapper=requests_wrapper, simple_parser=simple_parser, with_response=with_response)

    @property
    def _chain_type(self) -> str:
        return "RestGPT Caller"

    @property
    def input_keys(self) -> List[str]:
        return ["api_plan"]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
            self.max_execution_time is not None
            and time_elapsed >= self.max_execution_time
        ):
            return False

        return True
    
    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Response: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought: "
    
    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]
    
    def _construct_scratchpad(
        self, history: List[Tuple[str, str]]
    ) -> str:
        if len(history) == 0:
            return ""
        scratchpad = ""
        for i, (plan, execution_res) in enumerate(history):
            scratchpad += self.llm_prefix.format(i + 1) + plan + "\n"
            scratchpad += self.observation_prefix + execution_res + "\n"
        return scratchpad

    def _get_action_and_input(self, llm_output: str) -> Tuple[str, str]:
        if "Execution Result:" in llm_output:
            return "Execution Result", llm_output.split("Execution Result:")[-1].strip()
        # \s matches against tab/newline/whitespace
        regex = r"Operation:[\s]*(.*?)[\n]*Input:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # TODO: not match, just return
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        if action not in ["GET", "POST", "DELETE", "PUT"]:
            raise NotImplementedError
        
        # avoid error in the JSON format
        action_input = fix_json_error(action_input)

        return action, action_input
    
    def _get_response(self, action: str, action_input: str) -> str:
        action_input = action_input.strip().strip('`')
        left_bracket = action_input.find('{')
        right_bracket = action_input.rfind('}')
        action_input = action_input[left_bracket:right_bracket + 1]
        try:
            data = json.loads(action_input)
        except json.JSONDecodeError as e:
            raise e
        
        desc = data.get("description", "No description")
        query = data.get("output_instructions", None)

        # if 'securetrack' in action_input:
        #     #securetrack requires Basic
        #     headers = {
        #         'Authorization': f'Basic {os.environ["TUFIN_BASIC_AUTH"]}'
        #     }
        #     self.requests_wrapper = Requests(headers=headers)
        # else:
        #     # securechange api requires Bearer authentication
        #     headers = {
        #         'Authorization': f'Basic {os.environ["TUFIN_SC_BEARER_AUTH"]}'
        #     }
        #     self.requests_wrapper = Requests(headers=headers)
        params, request_body = None, None
        if action == "GET":
            if 'params' in data:
                params = data.get("params")
                response = self.requests_wrapper.get(data.get("url"), params=params, verify=False)
            else:
                response = self.requests_wrapper.get(data.get("url"), verify=False)
        elif action == "POST":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.post(data["url"], params=params, data=request_body, verify=False)
        elif action == "PUT":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.put(data["url"], params=params, data=request_body, verify=False)
        elif action == "DELETE":
            params = data.get("params")
            request_body = data.get("data")
            response = self.requests_wrapper.delete(data["url"], params=params, json=request_body, verify=False)
        else:
            raise NotImplementedError
        
        if isinstance(response, requests.models.Response):
            if response.status_code != 200:
                return response.text
            response_text = response.text
        elif isinstance(response, str):
            response_text = response
        else:
            raise NotImplementedError
        
        return response_text, params, request_body, desc, query
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        intermediate_steps: List[Tuple[str, str]] = []

        api_plan = inputs['api_plan']
        api_url = self.api_spec.servers[0]['url']
        matched_endpoints = get_matched_endpoint(self.api_spec, api_plan)
        endpoint_docs_by_name = {name: docs for name, _, docs in self.api_spec.endpoints}
        api_doc_for_caller = ""
        assert len(matched_endpoints) == 1, f"Found {len(matched_endpoints)} matched endpoints, but expected 1."
        endpoint_name = matched_endpoints[0]
        tmp_docs = deepcopy(endpoint_docs_by_name.get(endpoint_name))
        if 'responses' in tmp_docs and 'content' in tmp_docs['responses']:
            if 'application/json' in tmp_docs['responses']['content']:
                tmp_docs['responses'] = tmp_docs['responses']['content']['application/json']['schema']['properties']
            elif 'application/json; charset=utf-8' in tmp_docs['responses']['content']:
                tmp_docs['responses'] = tmp_docs['responses']['content']['application/json; charset=utf-8']['schema']['properties']
        if not self.with_response and 'responses' in tmp_docs:
            tmp_docs.pop("responses")
        tmp_docs = yaml.dump(tmp_docs)
        encoder = tiktoken.encoding_for_model('gpt-3.5-turbo-0125')
        encoded_docs = encoder.encode(tmp_docs)
        if len(encoded_docs) > 1500:
            tmp_docs = encoder.decode(encoded_docs[:1500])
        api_doc_for_caller += f"== Docs for {endpoint_name} == \n{tmp_docs}\n"

        caller_prompt = PromptTemplate(
            template=CALLER_PROMPT,
            partial_variables={
                "api_url": api_url,
                "api_docs": api_doc_for_caller,
            },
            input_variables=["api_plan", "background", "agent_scratchpad"],
        )
        
        caller_chain = LLMChain(llm=self.llm, prompt=caller_prompt)

        while self._should_continue(iterations, time_elapsed):
            scratchpad = self._construct_scratchpad(intermediate_steps)
            caller_chain_output = caller_chain.run(api_plan=api_plan, background=inputs['background'], agent_scratchpad=scratchpad, stop=self._stop)
            logger.info(f"Caller: {caller_chain_output}")

            action, action_input = self._get_action_and_input(caller_chain_output)
            if action == "Execution Result":
                return {"result": action_input}
            response, params, request_body, desc, query = self._get_response(action, action_input)

            called_endpoint_name = action + ' ' + json.loads(action_input)['url'].replace(api_url, '')
            called_endpoint_name = get_matched_endpoint(self.api_spec, called_endpoint_name)[0]
            api_path = api_url + called_endpoint_name.split(' ')[-1]
            api_doc_for_parser = endpoint_docs_by_name.get(called_endpoint_name)
            if not self.simple_parser:
                response_parser = ResponseParser(
                    llm=self.llm,
                    api_path=api_path,
                    api_doc=api_doc_for_parser,
                )
            else:
                response_parser = SimpleResponseParser(
                    llm=self.llm,
                    api_path=api_path,
                    api_doc=api_doc_for_parser,
                )

            params_or_data = {
                "params": params if params is not None else "No parameters",
                "data": request_body if request_body is not None else "No request body",
            }
            parsing_res = response_parser.run(query=query, response_description=desc, api_param=params_or_data, json=response)
            logger.info(f"Parser: {parsing_res}")

            intermediate_steps.append((caller_chain_output, parsing_res))

            iterations += 1
            time_elapsed = time.time() - start_time

        return {"result": caller_chain_output}



# "data": {{
#          "ticket.subject": "OMERTEST",
#          "ticket.priority": "Normal",
#          "ticket.workflow.id": 10,
#          "ticket.workflow.name": AR,
#          "ticket.steps.step.name": "Open request",
#          "ticket.steps.step.tasks.task.fields.field.@xsi.type": "multi_access_request",
#          "ticket.steps.step.tasks.task.fields.field.name": "ar",
#          "ticket.steps.step.tasks.task.fields.field.access_request.use_topology": true,
#          "ticket.steps.step.tasks.task.fields.field.access_request.targets.target.@type":"Object",
#          "ticket.steps.step.tasks.task.fields.field.access_request.targets.target.object_name": "FortiGate_7_0_112_244-SD-WAN",
#          "ticket.steps.step.tasks.task.fields.field.access_request.targets.target.object_name": "FMG/SD-WAN",
#     }},