import http
import os
import subprocess
from time import time, sleep

import kubernetes
import requests
from ocp_resources.pod import Pod
from ocp_resources.route import Route

from utils.constants import TRUSTYAI_SERVICE, TRUSTYAI_SPD_ENDPOINT, TRUSTYAI_NAMES_ENDPOINT, INFERENCE_ENDPOINT, \
    TRUSTYAI_MODEL_METADATA_ENDPOINT, MM_PAYLOAD_PROCESSORS

import logging

logger = logging.getLogger(__name__)


class TrustyAIPodNotFoundError(Exception):
    pass


def send_data_to_inference_service(client, namespace, inference_service, data_path):
    inference_route = next(Route.get(client=client, namespace=namespace.name, name=inference_service.name))
    token = get_ocp_token()

    responses = []
    errors = []

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if os.path.isfile(file_path):
            start_obs = get_trustyai_service_datapoint_counter(client=client,
                                                               namespace=namespace,
                                                               inference_service=inference_service)

            with open(file_path, "r") as file:
                data = file.read()

            url = f"https://{inference_route.host}{inference_route.instance.spec.path}{INFERENCE_ENDPOINT}"

            headers = {"Authorization": f"Bearer {token}"}
            response = requests.post(url=url, headers=headers, data=data, verify=False)

            end_obs = get_trustyai_service_datapoint_counter(client=client,
                                                             namespace=namespace,
                                                             inference_service=inference_service)

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


def get_trustyai_service_datapoint_counter(client, namespace, inference_service):
    model_metadata_response = get_trustyai_model_metadata(client=client, namespace=namespace)

    if model_metadata_response:
        model_metadata = model_metadata_response.json()

        model_data = next((item for item in model_metadata if item.get("modelId") == inference_service.name), None)

        if model_data:
            observations = model_data.get("observations", 0)
            return observations
        else:
            return 0
    else:
        raise Exception("Failed to retrieve model metadata")


def get_trustyai_model_metadata(client, namespace):
    return send_trustyai_service_request(client=client, namespace=namespace, endpoint=TRUSTYAI_MODEL_METADATA_ENDPOINT,
                                         method=http.HTTPMethod.GET)


def apply_trustyai_name_mappings(client, namespace, inference_service, input_mappings, output_mappings):
    data = {
        "modelId": inference_service.name,
        "inputMapping": input_mappings,
        "outputMapping": output_mappings
    }

    return send_trustyai_service_request(client=client, namespace=namespace, endpoint=TRUSTYAI_NAMES_ENDPOINT,
                                         method=http.HTTPMethod.POST, data=data)


def get_fairness_metrics(client,
                         namespace,
                         inference_service,
                         protected_attribute,
                         privileged_attribute,
                         unprivileged_attribute,
                         outcome_name,
                         favorable_outcome,
                         batch_size):
    data = {
        "modelId": inference_service.name,
        "protectedAttribute": protected_attribute,
        "privilegedAttribute": privileged_attribute,
        "unprivilegedAttribute": unprivileged_attribute,
        "outcomeName": outcome_name,
        "favorableOutcome": favorable_outcome,
        "batchSize": batch_size
    }

    return send_trustyai_service_request(client=client, namespace=namespace, endpoint=TRUSTYAI_SPD_ENDPOINT,
                                         method=http.HTTPMethod.POST, data=data)


def send_trustyai_service_request(client, namespace, endpoint, method, data=None):
    trustyai_service_route = get_trustyai_service_route(client=client, namespace=namespace)
    token = get_ocp_token()

    url = f"https://{trustyai_service_route.host}{endpoint}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    response = None
    if method == http.HTTPMethod.GET:
        response = requests.get(url=url, headers=headers, verify=False)
    elif method == http.HTTPMethod.POST:
        response = requests.post(url=url, headers=headers, json=data, verify=False)

    return response


def wait_for_model_pods(client, namespace):
    pods_with_env_var = False
    all_pods_running = False
    timeout = 60 * 3
    start_time = time()
    while not pods_with_env_var or not all_pods_running:
        if time() - start_time > timeout:
            raise TimeoutError("Not all model pods are ready in time")

        model_pods = [pod for pod in Pod.get(client=client, namespace=namespace.name)
                      if "modelmesh-serving" in pod.name]

        pods_with_env_var = False
        all_pods_running = True
        for pod in model_pods:
            try:
                has_env_var = False
                for container in pod.instance.spec.containers:
                    if container.env is not None and any(env.name == MM_PAYLOAD_PROCESSORS for env in container.env):
                        has_env_var = True
                        break

                if has_env_var:
                    pods_with_env_var = True
                    if pod.status != Pod.Status.RUNNING:
                        all_pods_running = False
                        break
            except kubernetes.dynamic.exceptions.NotFoundError:
                # Ignore the error if the pod is not found (deleted during the process)
                continue

        if not pods_with_env_var or not all_pods_running:
            sleep(5)
