import json
import logging
import time

from utils.utils import send_data_to_inference_service, get_trustyai_model_metadata, apply_trustyai_name_mappings, \
    get_fairness_metrics

logger = logging.getLogger(__name__)


def test_basic(client, model_namespace, trustyai_service, onnx_loan_model_alpha_inference_service,
               onnx_loan_model_alpha_pod):
    time.sleep(60)
    logger.info("Sending training data...")
    responses, errors = send_data_to_inference_service(client=client,
                                                       inference_service=onnx_loan_model_alpha_inference_service,
                                                       namespace=model_namespace,
                                                       data_path="../data/training")
    logger.info(responses)
    logger.info(errors)

    logger.info("Getting TrustyAI Model metadata:")
    response = get_trustyai_model_metadata(client=client, namespace=model_namespace)
    logger.info(response)
    if response.status_code == 200:
        data = response.json()[0]["data"]
        logger.info(json.dumps(data, indent=2))
    else:
        logger.info(f"Request failed with status code: {response.status_code}")
        logger.error(f"Response headers: {response.headers}")

    response = apply_trustyai_name_mappings(client=client, namespace=model_namespace,
                                            inference_service=onnx_loan_model_alpha_inference_service)
    logger.info(response)
    if response.status_code == 200:
        logger.info(response.content)
    else:
        logger.info(f"Request failed with status code: {response.status_code}")
        logger.error(f"Response headers: {response.headers}")

    response = get_fairness_metrics(client=client, namespace=model_namespace,
                                    inference_service=onnx_loan_model_alpha_inference_service)
    logger.info(response)
    if response.status_code == 200:
        logger.info(response.content)
    else:
        logger.info(f"Request failed with status code: {response.status_code}")
        logger.error(f"Response headers: {response.headers}")

    time.sleep(60)
    assert True
