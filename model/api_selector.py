from typing import Any, Dict, List, Optional, Tuple
import re
import logging

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM

from utils import ReducedOpenAPISpec, get_matched_endpoint

logger = logging.getLogger(__name__)


icl_examples = {
    "tufin": """Example 1:

Background: The id of a device monitored by Securetrack is 14
User query: give me the device with id 14 in JSON format
API calling 1: GET /securetrack/api/devices/14.json to get the the device with id 14
API response: The device details with id 14 is {
    "device": {
        "id": "14",
        "name": "DG1.1",
        "vendor": "PaloAltoNetworks",
        "model": "Panorama_device_group",
        "domain_id": "1",
        "domain_name": "Default",
        "offline": false,
        "topology": false,
        "module_uid": "",
        "ip": "10.100.249.211",
        "latest_revision": "10",
        "parent_id": 4,
        "virtual_type": "mdom",
        "context_name": "DG1.1",
        "status": "Started",
        "module_type": "",
        "licenses": [
        ]
    }
}


Example 2:

Background: No background
User query: Get all the devices that monitored by Securetrack
API calling 1: GET /securetrack/api/devices.json to get all the monitored devices
API response: Here are all the devices that monitored by Securetrack in a JSON format {
    "devices": {
        "count": 3,
        "total": 3,
        "device": [
            {
                "id": "1",
                "name": "249.211",
                "vendor": "PaloAltoNetworks",
                "model": "Panorama_ng",
                "domain_id": "1",
                "domain_name": "Default",
                "offline": false,
                "topology": false,
                "module_uid": "",
                "ip": "10.100.249.211",
                "latest_revision": "2",
                "virtual_type": "management",
                "context_name": "Shared",
                "status": "Started",
                "module_type": ""
            },
            {
                "id": "2",
                "name": "Branches",
                "vendor": "PaloAltoNetworks",
                "model": "Panorama_device_group",
                "domain_id": "1",
                "domain_name": "Default",
                "offline": false,
                "topology": false,
                "module_uid": "",
                "ip": "10.100.249.211",
                "latest_revision": "3",
                "parent_id": 1,
                "virtual_type": "mdom",
                "context_name": "Branches",
                "status": "Started",
                "module_type": ""
            },
            {
                "id": "3",
                "name": "horizon_DG1",
                "vendor": "PaloAltoNetworks",
                "model": "Panorama_device_group",
                "domain_id": "1",
                "domain_name": "Default",
                "offline": false,
                "topology": false,
                "module_uid": "",
                "ip": "10.100.249.211",
                "latest_revision": "4",
                "parent_id": 1,
                "virtual_type": "mdom",
                "context_name": "horizon_DG1",
                "status": "Started",
                "module_type": ""
            }
        ]
    }
}

Example 3:

Background: No background
User query: Get all the devices that monitored by Securetrack in which their virtual_type is management
API calling 1: GET /securetrack/api/devices.json to get all the monitored devices
API response: Here are all the devices that monitored by Securetrack in a JSON format {
    "devices": {
        "count": 3,
        "total": 3,
        "device": [
            {
                "id": "1",
                "name": "249.211",
                "vendor": "PaloAltoNetworks",
                "model": "Panorama_ng",
                "domain_id": "1",
                "domain_name": "Default",
                "offline": false,
                "topology": false,
                "module_uid": "",
                "ip": "10.100.249.211",
                    "latest_revision": "2",
                "virtual_type": "management",
                "context_name": "Shared",
                "status": "Started",
                "module_type": ""
            },
            {
                "id": "2",
                "name": "Branches",
                "vendor": "PaloAltoNetworks",
                "model": "Panorama_device_group",
                "domain_id": "1",
                "domain_name": "Default",
                "offline": false,
                "topology": false,
                "module_uid": "",
                "ip": "10.100.249.211",
                "latest_revision": "3",
                "parent_id": 1,
                "virtual_type": "mdom",
                "context_name": "Branches",
                "status": "Started",
                "module_type": ""
            },
            {
                "id": "3",
                "name": "horizon_DG1",
                "vendor": "PaloAltoNetworks",
                "model": "Panorama_device_group",
                "domain_id": "1",
                "domain_name": "Default",
                "offline": false,
                "topology": false,
                "module_uid": "",
                "ip": "10.100.249.211",
                "latest_revision": "4",
                "parent_id": 1,
                "virtual_type": "mdom",
                "context_name": "horizon_DG1",
                "status": "Started",
                "module_type": ""
            },
        ]
    }
}
Instruction: Continue. Filter the response API calling 1 to return only those devices that their virtual_type value is management 
API response: Here is the response in the JSON format for device id that its  "virtual_type" is of type "management" {
    "devices": {
        "count": 1,
        "total": 1,
        "device": [
            {
                "id": "1",
                "name": "249.211",
                "vendor": "PaloAltoNetworks",
                "model": "Panorama_ng",
                "domain_id": "1",
                "domain_name": "Default",
                "offline": false,
                "topology": false,
                "module_uid": "",
                "ip": "10.100.249.211",
                "latest_revision": "2",
                "virtual_type": "management",
                "context_name": "Shared",
                "status": "Started",
                "module_type": ""
            }
        ]
    }
}

Example 4:
Background: There are revisions that represent the device configuration settings
User query: Get the revisions of device id 4
API calling 1: GET /securetrack/api/devices/4/revisions.json to get the list of the revision for device 4
API response: Here are the revisions of device 4 in the JSON format {
    "revision": [
        {
            "id": "33",
            "revisionId": "2",
            "action": "automatic",
            "date": "2024-04-09",
            "time": "00:05:48.000",
            "admin": "",
            "guiClient": "-",
            "auditLog": "-",
            "policyPackage": "Standard",
            "authorizationStatus": "n_a",
            "modules_and_policy": {
                "module_and_policy": [
                ]
            },
            "firewall_status": true,
            "ready": true,
            "tickets": {
                "ticket": [
                ]
            }
        },
        {
            "id": "27",
            "revisionId": "2",
            "action": "automatic",
            "date": "2024-04-08",
            "time": "19:01:06.000",
            "admin": "",
            "guiClient": "",
            "auditLog": "",
            "policyPackage": "Standard",
            "authorizationStatus": "n_a",
            "modules_and_policy": {
                "module_and_policy": [
                ]
            },
            "firewall_status": true,
            "ready": true,
            "tickets": {
                "ticket": [
                ]
            }
        },
        {
            "id": "21",
            "revisionId": "2",
            "action": "automatic",
            "date": "2024-04-08",
            "time": "17:48:22.000",
            "admin": "",
            "guiClient": "",
            "auditLog": "",
            "policyPackage": "Standard",
            "authorizationStatus": "n_a",
            "modules_and_policy": {
                "module_and_policy": [
                ]
            },
            "firewall_status": true,
            "ready": true,
            "tickets": {
                "ticket": [
                ]
            }
        },
        {
            "id": "5",
            "revisionId": "1",
            "action": "automatic",
            "date": "2024-04-08",
            "time": "16:49:17.000",
            "admin": "",
            "guiClient": "-",
            "auditLog": "-",
            "policyPackage": "Standard",
            "authorizationStatus": "n_a",
            "modules_and_policy": {
                "module_and_policy": [
                ]
            },
            "firewall_status": true,
            "ready": true,
            "tickets": {
                "ticket": [
                ]
            }
        }
    ]
}


Example 5:
Background: There are devices in an IP Network that is constructed from Switches, Routers and Firewall devices
User query: Check if the traffic between src IP address 29.29.29.1/24 to destination IP address 25.25.0.0/16 on service any is blocked or open
API calling 1: GET /securetrack/api/topology/path.json?src={source}&dst={destination}&service={service} to check if a traffic between src ip 29.29.29.29/24 to destination IP 25.25.25.25/16 on a service any is allowed
API response: Here is the path between source ip 29.29.29.1/24 to destination 25.25.0.0/16 for a Facebook service {
    "path_calc_results": {
        "traffic_allowed": false,
        "device_info": {
            "id": "17",
            "name": "PA-VM-111.6 (Cluster)",
            "type": "mgmt",
            "vendor": "Palo Alto Networks",
            "incomingInterfaces": {
                "incomingVrf": "default",
                "ip": "29.29.29.1/255.255.255.252",
                "name": "tunnel.11"
            },
            "nextDevices": {
                "name": "DIRECTLY_CONNECTED",
                "routes": {
                    "outgoingInterfaceName": "ethernet1/5",
                    "outgoingVrf": "default",
                    "routeDestination": "25.25.25.1/255.255.255.0"
                }
            },
            "bindings": {
                "name": "",
                "rules": [
                    {
                        "action": "Accept",
                        "applications": "facebook",
                        "destNegated": false,
                        "destinations": "Any",
                        "ruleIdentifier": 43,
                        "ruleUid": {000C29BB-3503-0ed3-0000-000268436494},
                        "serviceNegated": false,
                        "services": "Any",
                        "sourceNegated": false,
                        "sources": "Any",
                        "users": "Any"
                    },
                    {
                        "action": "Deny",
                        "applications": "Any",
                        "destNegated": false,
                        "destinations": "Any",
                        "ruleIdentifier": 62,
                        "ruleUid": {025E4E47-8D79-7155-7EFE-5FD7AD7E9ED6},
                        "serviceNegated": false,
                        "services": "Any",
                        "sourceNegated": false,
                        "sources": "Any",
                        "users": "Any"
                    }
                ]
            }
        }
    }
}
Instruction: Continue. Filter the response API calling 1 to return the value of traffic_allowed field indicating if the traffic is allowed or blocked and retrun also the list of devices on the path taken from the device_info field
API response: Here is the response in the JSON format for a {
    "path_calc_results": {
        "traffic_allowed": false,
        "device_info": [
            {
                "id": 45,
                "name": "FortiGate_7_0_112_244-SD-WAN",
                "type": "mgmt",
                "vendor": "Fortinet",
                "incomingInterfaces": [
                    {
                        "name": "forSdwan0",
                        "ip": "172.16.100.1/30",
                        "incomingVrf": "0"
                    }
                ],
                "nextDevices": [
                    {
                        "name": "DIRECTLY_CONNECTED",
                        "routes": [
                            {
                                "routeDestination": "10.200.0.4/24",
                                "outgoingInterfaceName": "port4",
                                "outgoingVrf": "0"
                            }
                        ]
                    }
                ],
                "natList": [
                ],
                "ipsecList": [
                ],
                "pbrEntryList": [
                ],
                "sdwanEntryJsonList": [
                ],
            }
        ]
    }
}

Example 6:

Background: No Background
User query: give me all the active workflows in securechange
API calling 1: GET /securechangeworkflow/api/securechange/workflows/active_workflows.json to get all the active workflows in securetrack
API response: The active workflows are  {
    "workflows":{
        "workflow":{
            "id": 10,
            "name": "AR",
            "description": "",
            "type": "ACCESS_REQUEST"
        }
    }
}

Example 7:

Background: There is a blocked traffic on device management FMG/SD-WAN and device name FortiGate_7_0_112_244-SD-WAN between Source address 10.10.10.1/24 and Destination address 3.3.3.3/30 on service TCP:80
User query: Create AccessRequest (AR) ticket with subject OMERTEST with workflow id 10 and name AR with target device management FMG/SD-WAN and device name FortiGate_7_0_112_244-SD-WAN with Source address of IP 10.251.4.0/23 and Destination address of IP 10.252.0.0/22 on service TCP:80 and Action Accept
API calling 1: POST /securechangeworkflow/api/securechange/tickets.json with request body content as JSON { "ticket": { "subject": "blocked traffic on FortiGate_7_0_112_244-SD-WAN", "priority": "Normal", "domain_name": "", "workflow": { "id": 7, "name": "AR", "uses_topology": true }, "steps": { "step": [ { "name": "Open request", "tasks": { "task": { "fields": { "field": { "@xsi.type": "multi_access_request", "name": "ar", "access_request": { "use_topology": true, "targets": { "target": { "@type": "Object", "object_name": "FortiGate_7_0_112_244-SD-WAN", "management_name": "FMG/SD-WAN" } }, "users": { "user": [ "Any" ] }, "sources": { "source": [ { "@type": "IP", "ip_address": "172.16.100.0", "netmask": "255.255.255.252", "cidr": 30 } ] }, "destinations": { "destination": [ { "@type": "IP", "ip_address": "10.200.0.0", "netmask": "255.255.255.0", "cidr": 24 } ] }, "services": { "service": [ { "@type": "PROTOCOL", "protocol": "TCP", "port": 80 } ] }, "action": "Accept", "labels": "" } } } } } } ] }, "comments": "" } }
API response: response status code with 201 Created
"""
}

# Thought: I am finished executing the plan and have the information the user asked for or the data the used asked to create
# Final Answer: the final output from executing the plan. If the user's query contains filter conditions, you need to filter the results as well. For example, if the user query is "Search for the first person whose name is 'Tom Hanks'", you should filter the results and only output the first person whose name is 'Tom Hanks'.
API_SELECTOR_PROMPT = """You are a planner that plans a sequence of RESTful API calls to assist with user queries against an API.
Another API caller will receive your plan call the corresponding APIs and finally give you the result in natural language.
The API caller also has filtering, sorting functions to post-process the response of APIs. Therefore, if you think the API response should be post-processed, just tell the API caller to do so.
If you think you have got the final answer, do not make other API calls and just output the answer immediately. For example, the query is search for a person, you should just return the id and name of the person.

----

Here are name and description of available APIs.
Do not use APIs that are not listed here.

{endpoints}

----

Starting below, you should follow this format:

Background: background information which you can use to execute the plan, e.g., the id of a person, the id of tracks by Faye Wong. In most cases, you must use the background information instead of requesting these information again. For example, if the query is "get the poster for any other movie directed by Wong Kar-Wai (12453)", and the background includes the movies directed by Wong Kar-Wai, you should use the background information instead of requesting the movies directed by Wong Kar-Wai again.
User query: the query a User wants help with related to the API
API calling 1: the first api call you want to make. Note the API calling can contain conditions such as filtering, sorting, etc. For example, "GET /movie/18329/credits to get the director of the movie Happy Together", "GET /movie/popular to get the top-1 most popular movie". If user query contains some filter condition, such as the latest, the most popular, the highest rated, then the API calling plan should also contain the filter condition. If you think there is no need to call an API, output "No API call needed." and then output the final answer according to the user query and background information.
API response: the response of API calling 1
Instruction: Another model will evaluate whether the user query has been fulfilled. If the instruction contains "continue", then you should make another API call following this instruction.
... (this API calling n and API response can repeat N times, but most queries can be solved in 1-2 step)


{icl_examples}

Note, if you are missing value to construct the API you can return result requesting the missing value
Note, if the API path contains "{{}}", it means that it is a variable and you should replace it with the appropriate value. For example, if the path is "/users/{{user_id}}/tweets", you should replace "{{user_id}}" with the user id. "{{" and "}}" cannot appear in the url. In most cases, the id value is in the background or the API response. Just copy the id faithfully. If the id is not in the background, instead of creating one, call other APIs to query the id. For example, before you call "/users/{{user_id}}/playlists", you should get the user_id via "GET /me" first. Another example is that before you call "/person/{{person_id}}", you should get the movie_id via "/search/person" first.

Begin!

Background: {background}
User query: {plan}
API calling 1: {agent_scratchpad}"""



class APISelector(Chain):
    llm: BaseLLM
    api_spec: ReducedOpenAPISpec
    scenario: str
    api_selector_prompt: BasePromptTemplate
    output_key: str = "result"

    def __init__(self, llm: BaseLLM, scenario: str, api_spec: ReducedOpenAPISpec) -> None:
        api_name_desc = [f"{endpoint[0]} {endpoint[1].split('.')[0] if endpoint[1] is not None else ''}" for endpoint in api_spec.endpoints]
        api_name_desc = '\n'.join(api_name_desc)
        api_selector_prompt = PromptTemplate(
            template=API_SELECTOR_PROMPT,
            partial_variables={"endpoints": api_name_desc, "icl_examples": icl_examples[scenario]},
            input_variables=["plan", "background", "agent_scratchpad"],
        )
        super().__init__(llm=llm, api_spec=api_spec, scenario=scenario, api_selector_prompt=api_selector_prompt)

    @property
    def _chain_type(self) -> str:
        return "RestGPT API Selector"

    @property
    def input_keys(self) -> List[str]:
        return ["plan", "background"]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "API response: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "API calling {}: "
    
    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]
    
    def _construct_scratchpad(
        self, history: List[Tuple[str, str]], instruction: str
    ) -> str:
        if len(history) == 0:
            return ""
        scratchpad = ""
        for i, (plan, api_plan, execution_res) in enumerate(history):
            if i != 0:
                scratchpad += "Instruction: " + plan + "\n"
            scratchpad += self.llm_prefix.format(i + 1) + api_plan + "\n"
            scratchpad += self.observation_prefix + execution_res + "\n"
        scratchpad += "Instruction: " + instruction + "\n"
        return scratchpad
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        # inputs: background, plan, (optional) history, instruction
        if 'history' in inputs:
            scratchpad = self._construct_scratchpad(inputs['history'], inputs['instruction'])
        else:
            scratchpad = ""
        api_selector_chain = LLMChain(llm=self.llm, prompt=self.api_selector_prompt)
        api_selector_chain_output = api_selector_chain.run(plan=inputs['plan'], background=inputs['background'], agent_scratchpad=scratchpad, stop=self._stop)
        
        api_plan = re.sub(r"API calling \d+: ", "", api_selector_chain_output).strip()

        logger.info(f"API Selector: {api_plan}")

        finish = re.match(r"No API call needed.(.*)", api_plan)
        if finish is not None:
            return {"result": api_plan}

        iterations = 0
        while get_matched_endpoint(self.api_spec, api_plan) is None and iterations < 3:
            iterations = iterations + 1
            logger.info("API Selector: The API you called is not in the list of available APIs. Please use another API.")
            scratchpad += api_selector_chain_output + "\nThe API you called is not in the list of available APIs. Please use another API.\n"
            api_selector_chain_output = api_selector_chain.run(plan=inputs['plan'], background=inputs['background'], agent_scratchpad=scratchpad, stop=self._stop)
            api_plan = re.sub(r"API calling \d+: ", "", api_selector_chain_output).strip()
            logger.info(f"API Selector: {api_plan}")

        return {"result": api_plan}
