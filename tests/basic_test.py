import http
import logging

from utils.utils import send_data_to_inference_service, get_trustyai_model_metadata, apply_trustyai_name_mappings, \
    get_fairness_metrics

logger = logging.getLogger(__name__)


def test_basic(client, model_namespace, trustyai_service, onnx_loan_model_alpha_inference_service):
    logger.info("Sending training data...")
    send_data_to_inference_service(client=client,
                                   inference_service=onnx_loan_model_alpha_inference_service,
                                   namespace=model_namespace,
                                   data_path="./data/training")

    logger.info("Getting TrustyAI Model metadata:")
    response = get_trustyai_model_metadata(client=client, namespace=model_namespace)
    logger.info(response.content)

    logger.info("Applying name mappings...")
    input_mappings = {
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
    }
    output_mappings = {"predict": "Will Default?"}
    response = apply_trustyai_name_mappings(client=client,
                                            namespace=model_namespace,
                                            inference_service=onnx_loan_model_alpha_inference_service,
                                            input_mappings=input_mappings,
                                            output_mappings=output_mappings)
    logger.info(response.content)

    logger.info("Getting fairness metrics...")
    response = get_fairness_metrics(client=client,
                                    namespace=model_namespace,
                                    inference_service=onnx_loan_model_alpha_inference_service,
                                    protected_attribute="Is Male-Identifying?",
                                    privileged_attribute=1.0,
                                    unprivileged_attribute=0.0,
                                    outcome_name="Will Default?",
                                    favorable_outcome=0,
                                    batch_size=5000)
    assert response.status_code == http.HTTPStatus.OK
    logger.info(response.content)
