# Serving Runtime
OVMS = "ovms"
OVMS_RUNTIME = f"{OVMS}-1.x"
OVMS_QUAY_IMAGE = "quay.io/opendatahub/openvino_model_server:stable"
OPENVINO_MODEL_FORMAT = "openvino_ir"

# Model Format
ONNX = "onnx"

# API Groups
KSERVE_API_GROUP = "serving.kserve.io"
TRUSTYAI_API_GROUP = "trustyai.opendatahub.io"

# TrustyAI
TRUSTYAI_SERVICE = "trustyai-service"

# TrustyAI Endpoints
TRUSTYAI_SPD_ENDPOINT = "/metrics/group/fairness/spd/"
TRUSTYAI_NAMES_ENDPOINT = "/info/names"
TRUSTYAI_MODEL_METADATA_ENDPOINT = "/info"

# InferenceService
INFERENCE_ENDPOINT = "/infer"

# Minio
MINIO_IMAGE = "quay.io/trustyai/modelmesh-minio-examples:gauss"