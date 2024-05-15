from time import time

import pytest
import kubernetes
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.configmap import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.service_account import ServiceAccount

from resources.inference_service import InferenceService
from resources.minio.minio_pod import MinioPod
from resources.minio.minio_secret import MinioSecret
from resources.minio.minio_service import MinioService
from resources.serving_runtime import ServingRuntime
from resources.trustyai_service import TrustyAIService
from utils.constants import TRUSTYAI_SERVICE, OVMS_RUNTIME, OVMS_QUAY_IMAGE, OVMS, OPENVINO_MODEL_FORMAT, ONNX, \
    MINIO_IMAGE


@pytest.fixture(scope="session")
def client():
    yield DynamicClient(client=kubernetes.config.new_client_from_config())


@pytest.fixture(scope="session")
def model_namespace(client):
    with Namespace(
            client=client,
            name="model-namespace",
            label={"modelmesh-enabled": "true"},
            teardown=True,
            delete_timeout=600,
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        yield ns


@pytest.fixture(scope="function")
def modelmesh_serviceaccount(client, model_namespace):
    with ServiceAccount(client=client, name="modelmesh-serving-sa", namespace=model_namespace.name):
        yield


@pytest.fixture(scope="session")
def cluster_monitoring_config(client):
    config_yaml = yaml.dump({'enableUserWorkload': 'true'})
    with ConfigMap(name="cluster-monitoring-config",
                   namespace="openshift-monitoring",
                   data={'config.yaml': config_yaml
                         }) as cm:
        print(cm.data)
        yield cm


@pytest.fixture(scope="session")
def user_workload_monitoring_config(client):
    config_yaml = yaml.dump({
        'prometheus': {
            'logLevel': 'debug',
            'retention': '15d'
        }
    })
    with ConfigMap(name="user-workload-monitoring-config",
                   namespace="openshift-user-workload-monitoring",
                   data={'config.yaml': config_yaml
                         }) as cm:
        yield cm


@pytest.fixture(scope="function")
def trustyai_service(client, model_namespace, modelmesh_serviceaccount,
                     cluster_monitoring_config, user_workload_monitoring_config):
    with TrustyAIService(name=TRUSTYAI_SERVICE, namespace=model_namespace.name, client=client) as trusty:
        yield trusty


@pytest.fixture(scope="function")
def minio_service(client, model_namespace):
    with MinioService(name="minio", port=9000, target_port=9000, namespace=model_namespace.name, client=client) as ms:
        yield ms


@pytest.fixture(scope="function")
def minio_pod(client, model_namespace):
    with MinioPod(client=client, name="minio", namespace=model_namespace.name, image=MINIO_IMAGE) as mp:
        yield mp


@pytest.fixture(scope="function")
def aws_minio_data_connection_secret(client, model_namespace):
    with MinioSecret(client=client, name="aws-connection-minio-data-connection",
                     namespace=model_namespace.name,
                     aws_access_key_id="VEhFQUNDRVNTS0VZ",
                     aws_default_region="dXMtc291dGg=",
                     aws_s3_bucket="bW9kZWxtZXNoLWV4YW1wbGUtbW9kZWxz",
                     aws_s3_endpoint="aHR0cDovL21pbmlvOjkwMDA=",
                     aws_secret_access_key="VEhFU0VDUkVUS0VZ") as ms:
        yield ms


@pytest.fixture(scope="function")
def minio_bucket(minio_service, minio_pod, aws_minio_data_connection_secret):
    yield minio_service, minio_pod, aws_minio_data_connection_secret


@pytest.fixture(scope="function")
def ovms_runtime(client, minio_bucket, model_namespace):
    supported_model_formats = [
        {
            "name": OPENVINO_MODEL_FORMAT,
            "version": "opset1",
            "autoSelect": True
        },
        {
            "name": ONNX,
            "version": "1"
        }
    ]
    containers = [
        {
            "name": OVMS,
            "image": OVMS_QUAY_IMAGE,
            "args": [
                "--port=8001",
                "--rest_port=8888",
                "--config_path=/models/model_config_list.json",
                "--file_system_poll_wait_seconds=0",
                "--grpc_bind_address=127.0.0.1",
                "--rest_bind_address=127.0.0.1"
            ],
            "resources": {
                "requests": {
                    "cpu": "500m",
                    "memory": "1Gi"
                },
                "limits": {
                    "cpu": "5",
                    "memory": "1Gi"
                }
            }
        }
    ]

    with ServingRuntime(client=client,
                        name=OVMS_RUNTIME,
                        namespace=model_namespace.name,
                        image=OVMS_QUAY_IMAGE,
                        supported_model_formats=supported_model_formats,
                        containers=containers,
                        grpc_endpoint=8085,
                        grpc_data_endpoint=8001,
                        server_type=OVMS,
                        ) as ovms:
        yield ovms


@pytest.fixture(scope="function")
def onnx_loan_model_alpha_inference_service(client, model_namespace, ovms_runtime):
    with InferenceService(client=client,
                          name="demo-loan-nn-onnx-alpha",
                          namespace=model_namespace.name,
                          path="onnx/loan_model_alpha_august.onnx",
                          storage_name="aws-connection-minio-data-connection",
                          model_format_name=ONNX,
                          runtime=OVMS_RUNTIME) as inference_service:
        yield inference_service


@pytest.fixture(scope="function")
def onnx_loan_model_alpha_pod(client, model_namespace, onnx_loan_model_alpha_inference_service):
    is_pod_running = False
    pod = None
    timeout = 60*3
    start_time = time()
    while not is_pod_running:
        if time() - start_time > timeout:
            raise TimeoutError("Model pop did not start in time")
        pod = next((pod for pod in Pod.get(client=client, namespace=model_namespace.name)
                    if "modelmesh-serving-ovms-1.x" in pod.name), None)
        if pod is not None:
            pod.wait_for_status(status=Pod.Status.RUNNING)
            is_pod_running = True
    return pod
