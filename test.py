import json

# JSON response snippet
data = `{"devices":{"count":7,"total":7,"device":[{"id":"1","name":"249.211","vendor":"PaloAltoNetworks","model":"Panorama_ng","domain_id":"1","domain_name":"Default","offline":false,"topology":false,"module_uid":"","ip":"10.100.249.211","latest_revision":"2","virtual_type":"management","context_name":"Shared","status":"Started","module_type":""},{"id":"2","name":"DG1","vendor":"PaloAltoNetworks","model":"Panorama_device_group","domain_id":"1","domain_name":"Default","offline":false,"topology":false,"module_uid":"","ip":"10.100.249.211","latest_revision":"3","parent_id":1,"virtual_type":"mdom","context_name":"DG1","status":"Started","module_type":""},{"id":"3","name":"DG1.1.1.1","vendor":"PaloAltoNetworks","model":"Panorama_device_group","domain_id":"1","domain_name":"Default","offline":false,"topology":false,"module_uid":"","ip":"10.100.249.211","latest_revision":"6","parent_id":6,"virtual_type":"mdom","context_name":"DG1.1.1.1","status":"Started","module_type":""},{"id":"4","name":"PA-VM_7","vendor":"PaloAltoNetworks","model":"Panorama_ng_fw","domain_id":"1","domain_name":"Default","offline":false,"topology":true,"module_uid":"","ip":"10.100.249.211","parent_id":3,"virtual_type":"context","context_name":"vsys1","status":"Error: Unable to get configuration","module_type":""},{"id":"5","name":"testDG","vendor":"PaloAltoNetworks","model":"Panorama_device_group","domain_id":"1","domain_name":"Default","offline":false,"topology":false,"module_uid":"","ip":"10.100.249.211","latest_revision":"7","parent_id":6,"virtual...`

# Parse JSON response
response = json.loads(data)

# Extract names and ids of all management devices
management_devices = [(device['name'], device['id']) for device in response['devices']['device'] if device['virtual_type'] == 'management']

# Print the result
for name, device_id in management_devices:
    print(f"The name of the management device is {name} and the id is {device_id}")

