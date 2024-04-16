from typing import Any, Dict, List, Optional, Tuple
import re

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM

icl_examples = {
    "tmdb": """Example 1:
User query: give me some movies performed by Tony Leung.
Plan step 1: search person with name "Tony Leung"
API response: Tony Leung's person_id is 1337
Plan step 2: collect the list of movies performed by Tony Leung whose person_id is 1337
API response: Shang-Chi and the Legend of the Ten Rings, In the Mood for Love, Hero
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: Tony Leung has performed in Shang-Chi and the Legend of the Ten Rings, In the Mood for Love, Hero

Example 2:
User query: Who wrote the screenplay for the most famous movie directed by Martin Scorsese?
Plan step 1: search for the most popular movie directed by Martin Scorsese
API response: Successfully called GET /search/person to search for the director "Martin Scorsese". The id of Martin Scorsese is 1032
Plan step 2: Continue. search for the most popular movie directed by Martin Scorsese (1032)
API response: Successfully called GET /person/{{person_id}}/movie_credits to get the most popular movie directed by Martin Scorsese. The most popular movie directed by Martin Scorsese is Shutter Island (11324)
Plan step 3: search for the screenwriter of Shutter Island
API response: The screenwriter of Shutter Island is Laeta Kalogridis (20294)
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: Laeta Kalogridis wrote the screenplay for the most famous movie directed by Martin Scorsese.
""",
    "spotify": """Example 1:
User query: set the volume to 20 and skip to the next track.
Plan step 1: set the volume to 20
API response: Successfully called PUT /me/player/volume to set the volume to 20.
Plan step 2: skip to the next track
API response: Successfully called POST /me/player/next to skip to the next track.
Thought: I am finished executing a plan and completed the user's instructions
Final Answer: I have set the volume to 20 and skipped to the next track.

Example 2:
User query: Make a new playlist called "Love Coldplay" containing the most popular songs by Coldplay
Plan step 1: search for the most popular songs by Coldplay
API response: Successfully called GET /search to search for the artist Coldplay. The id of Coldplay is 4gzpq5DPGxSnKTe4SA8HAU
Plan step 2: Continue. search for the most popular songs by Coldplay (4gzpq5DPGxSnKTe4SA8HAU)
API response: Successfully called GET /artists/4gzpq5DPGxSnKTe4SA8HAU/top-tracks to get the most popular songs by Coldplay. The most popular songs by Coldplay are Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b).
Plan step 3: make a playlist called "Love Coldplay"
API response: Successfully called GET /me to get the user id. The user id is xxxxxxxxx.
Plan step 4: Continue. make a playlist called "Love Coldplay"
API response: Successfully called POST /users/xxxxxxxxx/playlists to make a playlist called "Love Coldplay". The playlist id is 7LjHVU3t3fcxj5aiPFEW4T.
Plan step 5: Add the most popular songs by Coldplay, Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b), to playlist "Love Coldplay" (7LjHVU3t3fcxj5aiPFEW4T)
API response: Successfully called POST /playlists/7LjHVU3t3fcxj5aiPFEW4T/tracks to add Yellow (3AJwUDP919kvQ9QcozQPxg), Viva La Vida (1mea3bSkSGXuIRvnydlB5b) in playlist "Love Coldplay" (7LjHVU3t3fcxj5aiPFEW4T). The playlist id is 7LjHVU3t3fcxj5aiPFEW4T.
Thought: I am finished executing a plan and have the data the used asked to create
Final Answer: I have made a new playlist called "Love Coldplay" containing Yellow and Viva La Vida by Coldplay.
""",
    "tufin": """Example 1:
User query: get the details about device 12 in JSON format
Plan step 1: get the device with id=12
API response: Successfully called GET /securetrack/api/devices/12.json to get the device with id=12
Thought: I am finished executing a plan and completed the user's instructions
Final Answer: Here is the device 12 details in JSON format {"device":{"id":"12","name":"testDG","vendor":"PaloAltoNetworks","model":"Panorama_device_group","domain_id":"1","domain_name":"Default","offline":false,"topology":false,"module_uid":"","ip":"10.100.249.211","latest_revision":"41","parent_id":13,"virtual_type":"mdom","context_name":"testDG","status":"Started","module_type":"","licenses":[]}}

Example 2:
User query: get me the management devices
Plan step 1: First get all the devices in the system
API response: Successfully called GET /securetrack/api/devices.json to get all the devices in the system in JSON format
Plan step 2: Continue. Do not call another API. Instead, Take from the response JSON from step 1 the device that its virtual_type=management
Thought: I have finished executing a plan and now I need to take only the element from the JSON where the virtual_type is management 
Final Answer: Here is the device where its virtual_type is management {"id":"1","name":"249.211","vendor":"PaloAltoNetworks","model":"Panorama_ng","domain_id":"1","domain_name":"Default","offline":false,"topology":false,"module_uid":"","ip":"10.100.249.211","latest_revision":"2","virtual_type":"management","context_name":"Shared","status":"Started","module_type":""}


Example 3:
User query: Get list of waiting provisioning tasks in queue for all the management devices of the PaloAltoNetworks devices tree
Plan step 1: get all the PaloAltoNetworks management devices 
API response: Successfully called GET /securetrack/api/devices.json?vendor=PaloAltoNetworks to get all the PaloAltoNetworks devices
PLan step 2:  Continue. No need to call another API in this step. Take from the response only those devices that their  "virtual_type" field's value is "management". The device is is 26
Plan step 2: Continue. For each device id from step 1, get its list of waiting tasks in queue
API response: Successfully called GET /securetrack/api/devices/provisioning/waiting_tasks/{{id}} to get the list of waiting tasks in queue for a management device id
Thought: I am finished executing a plan and have the data the used asked to create
Final Answer: I have returned the list of devices names and their ids together with their list of the provisioning tasks in the queue


Example 4:
User query: Get the revision of all the devices in the system
Plan step 1: get all the devices 
API response: Successfully called GET /securetrack/api/devices.json to get all the devices
Plan step 2: Continue. For each device id from step 1, get its latest revision
API response: Successfully called GET securetrack/api/devices/{{device_id}}/revisions.json to get the list of the device's revisions
Thought: I am finished executing a plan and have the data the used asked to create
Final Answer: I have returned the list of revisions for all the device sin the system

Example 5:
User query: Check if the traffic between source 29.29.29.1/24 and destination 25.25.25.1/32 on service SSH is allowed
Plan step 1: Take the source 29.29.29.1/24, destination 25.25.25.1/32 and service SSH from the user query and use them to query the Topology path calculation API
API response: Successfully called GET /securetrack/api/topology/path.json?src={source}&dst={destination}&service={service} to get details about the topology path between the {source} and the {destination} on the given {service}
Thought: In the results from Step 1 I need to extract the traffic_allowed field and the device_info fields and then it means that I finished executing the plan and have the data the used asked me for
Final Answer: The traffic is allowed and here are the devices on the path as taken from the device_info field in the API response

Example 6:
User query: get me the list of active workflows from scurechange
Plan step 1: Get all the active workflows from securechange
API response: Successfully called GET /securechangeworkflow/api/securechange/workflows/active_workflows.json to get all the active workflow
Thought: I am finished executing a plan and have the data the used asked to retrieve
Final Answer: I have returned the list of all active workflows in securechange


"""
}

PLANNER_PROMPT = """You are an agent that plans solution to user queries.
You should always give your plan in natural language.
Another model will receive your plan and find the right API calls and give you the result in natural language.
If you assess that the current plan has not been fulfilled, you can output "Continue" to let the API selector select another API to fulfill the plan.
If you think you have got the final answer or the user query has been fulfilled, just output the answer immediately. If the query has not been fulfilled, you should continue to output your plan.
In most case, search, filter, and sort should be completed in a single step.
The plan should be as specific as possible. It is better not to use pronouns in plan, but to use the corresponding results obtained previously. For example, instead of "Get the most popular movie directed by this person", you should output "Get the most popular movie directed by Martin Scorsese (1032)". If you want to iteratively query something about items in a list, then the list and the elements in the list should also appear in your plan.
The plan should be straightforward. If you want to search, sort or filter, you can put the condition in your plan. For example, if the query is "Who is the lead actor of In the Mood for Love (id 843)", instead of "get the list of actors of In the Mood for Love", you should output "get the lead actor of In the Mood for Love (843)".
If 


Starting below, you should follow this format:

User query: the query a User wants help with related to the API.
Plan step 1: the first step of your plan for how to solve the query
API response: the result of executing the first step of your plan, including the specific API call made.
Plan step 2: based on the API response, the second step of your plan for how to solve the query. If the last step result is not what you want, you can output "Continue" to let the API selector select another API to fulfill the plan. For example, the last plan is "add a song (id xxx) in my playlist", but the last step API response is calling "GET /me/playlists" and getting the id of my playlist, then you should output "Continue" to let the API selector select another API to add the song to my playlist. Pay attention to the specific API called in the last step API response. If a inproper API is called, then the response may be wrong and you should give a new plan.
API response: the result of executing the second step of your plan
... (this Plan step n and API response can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: the final output from executing the plan


{icl_examples}

Begin!

User query: {input}
Plan step 1: {agent_scratchpad}"""


# If the API selector returns an answer that "No API call needed", you should use the "API Selector" Final answer as your Final Answer as well

class Planner(Chain):
    llm: BaseLLM
    scenario: str
    planner_prompt: str
    output_key: str = "result"

    def __init__(self, llm: BaseLLM, scenario: str, planner_prompt=PLANNER_PROMPT) -> None:
        super().__init__(llm=llm, scenario=scenario, planner_prompt=planner_prompt)

    @property
    def _chain_type(self) -> str:
        return "RestGPT Planner"

    @property
    def input_keys(self) -> List[str]:
        return ["input"]
    
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
        return "Plan step {}: "
    
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

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        scratchpad = self._construct_scratchpad(inputs['history'])
        # print("Scrachpad: \n", scratchpad)
        planner_prompt = PromptTemplate(
            template=self.planner_prompt,
            partial_variables={
                "agent_scratchpad": scratchpad,
                "icl_examples": icl_examples[self.scenario],
            },
            input_variables=["input"]
        )
        planner_chain = LLMChain(llm=self.llm, prompt=planner_prompt)
        planner_chain_output = planner_chain.run(input=inputs['input'], stop=self._stop)

        planner_chain_output = re.sub(r"Plan step \d+: ", "", planner_chain_output).strip()

        return {"result": planner_chain_output}

#Example 7:
# User query: For traffic between src IP address 172.16.100.0/30 to destination IP address 10.200.0.0/24 on ANY service, check if the traffic is blocked. If it is blocked take from the device info the id, name, type and vendor of this topology path and create an AccessRequest (AR) ticket with subject AR_TEST with workflow id 10 and workflow name AR with priority Normal. Take the target device from the device_info including its name and its management name. Use the source from the path calculation. Take the destination fopm the path calculation as well. Take also the service details from the path calculation parameters. Action Accept
# Plan step 1: For traffic between src IP address 172.16.100.0/30 to destination IP address 10.200.0.0/24 on ANY service, check if the traffic is blocked
# API response: Successfully called GET /securetrack/api/topology/path.json?src={source}&dst={destination}&service={service} to check if a traffic between src ip  172.16.100.0/30 to destination IP 10.200.0.0/24 on a service any is blocked
# Plan step 2: Continue. Take the device_info from the response of step 1. device_info contains the information about the devices on the path. For each such device retrive the devices id, type, name, vendor. Take also the source ip, destination ip and the service to open a ticket on secure change
# API response: Successfully called POST /securechangeworkflow/api/securechange/tickets.json to open a ticket in securechange
# Thought: I am finished executing a plan and have the data the used asked to create
# Final Answer: I have returned response status code with 201 Created ticket successfully
