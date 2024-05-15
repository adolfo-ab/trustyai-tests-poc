import http
import json
import os
import subprocess

import requests
from ocp_resources.pod import Pod
from ocp_resources.route import Route

from utils.constants import TRUSTYAI_SERVICE, TRUSTYAI_SPD_ENDPOINT, TRUSTYAI_NAMES_ENDPOINT, INFERENCE_ENDPOINT, \
    TRUSTYAI_MODEL_METADATA_ENDPOINT


class TrustyAIPodNotFoundError(Exception):
    pass


def send_data_to_inference_service(client, namespace, inference_service, data_path, ):
    trustyai_service_pod = get_trustyai_pod(client=client, namespace=namespace)
    inference_route = next(Route.get(client=client, namespace=namespace.name, name=inference_service.name))
    token = get_ocp_token()

    responses = []
    errors = []

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if os.path.isfile(file_path):
            start_obs = get_trustyai_service_datapoint_counter(trustyai_service_pod=trustyai_service_pod, file=file_path)

            with open(file_path, "r") as file:
                data = file.read()

            url = f"https://{inference_route.host}{inference_route.instance.spec.path}{INFERENCE_ENDPOINT}"

            headers = {"Authorization": f"Bearer {token}"}
            response = requests.post(url=url, headers=headers, data=data, verify=False)

            end_obs = get_trustyai_service_datapoint_counter(trustyai_service_pod=trustyai_service_pod, file=file_path)

            if end_obs > start_obs:
                responses.append(response)
            else:
                errors.append(f"Data from file {file_name} not received by TrustyAI service")

    if errors:
        return responses, errors
    else:
        return responses, None


def get_ocp_token():
    token = subprocess.check_output(["oc", "whoami", "-t"]).decode().strip()
    return token


def get_trustyai_pod(client, namespace):
    pod = next((pod for pod in Pod.get(client=client, namespace=namespace.name) if TRUSTYAI_SERVICE in pod.name), None)
    if pod is None:
        raise TrustyAIPodNotFoundError(f"No TrustyAI pod found in namespace {namespace.name}")
    return pod


def get_trustyai_service_route(client, namespace):
    return next(Route.get(client=client, namespace=namespace.name, name=TRUSTYAI_SERVICE))


def get_trustyai_service_datapoint_counter(trustyai_service_pod, file):
    command_output = trustyai_service_pod.execute(
        command=["bash", "-c", f"ls /inputs/ | grep \"{file}\" || echo \"\""],
        container=TRUSTYAI_SERVICE,
    )

    if not command_output.strip():
        return 0
    else:
        metadata_output = trustyai_service_pod.execute(
            command=["bash", "-c", f"cat /inputs/{file}-metadata.json"],
            container=TRUSTYAI_SERVICE,
        )

        metadata_json = json.loads(metadata_output)
        observations = metadata_json.get("observations", 0)

        return observations


def get_trustyai_model_metadata(client, namespace):
    return send_trustyai_service_request(client=client, namespace=namespace, endpoint=TRUSTYAI_MODEL_METADATA_ENDPOINT,
                                         method=http.HTTPMethod.GET)


def apply_trustyai_name_mappings(client, namespace, inference_service):
    data = {
        "modelId": inference_service.name,
        "inputMapping": {
            "customer_data_input-0": "Number of Children",
            "customer_data_input-1": "Total Income",
            "customer_data_input-2": "Number of Total Family Members",
            "customer_data_input-3": "Is Male-Identifying?",
            "customer_data_input-4": "Owns Car?",
            "customer_data_input-5": "Owns Realty?",
            "customer_data_input-6": "Is Partnered?",
            "customer_data_input-7": "Is Employed?",
            "customer_data_input-8": "Live with Parents?",
            "customer_data_input-9": "Age",
            "customer_data_input-10": "Length of Employment?"
        },
        "outputMapping": {
            "predict": "Will Default?"
        }
    }

    return send_trustyai_service_request(client=client, namespace=namespace, endpoint=TRUSTYAI_NAMES_ENDPOINT,
                                         method=http.HTTPMethod.POST, data=data)


def get_fairness_metrics(client, namespace, inference_service):
    data = {
        "modelId": inference_service.name,
        "protectedAttribute": "Is Male-Identifying?",
        "privilegedAttribute": 1.0,
        "unprivilegedAttribute": 0.0,
        "outcomeName": "Will Default?",
        "favorableOutcome": 0,
        "batchSize": 5000
    }

    return send_trustyai_service_request(client=client, namespace=namespace, endpoint=TRUSTYAI_SPD_ENDPOINT,
                                         method=http.HTTPMethod.POST, data=data)


def send_trustyai_service_request(client, namespace, endpoint, method, data=None):
    trustyai_service_route = get_trustyai_service_route(client=client, namespace=namespace)
    token = get_ocp_token()

    url = f"https://{trustyai_service_route.host}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = None
    if method == http.HTTPMethod.GET:
        response = requests.get(f"https://{trustyai_service_route.host}/info",
                                headers=headers, verify=False)
    elif method == http.HTTPMethod.POST:
        response = requests.post(url, headers=headers, json=data, verify=False)

    return response
