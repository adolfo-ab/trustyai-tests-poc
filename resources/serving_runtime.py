from ocp_resources.resource import NamespacedResource

from utils.constants import KSERVE_API_GROUP


class ServingRuntime(NamespacedResource):
    api_group = KSERVE_API_GROUP

    def __init__(
        self,
        client,
        name,
        namespace,
        image,
        supported_model_formats,
        containers,
        grpc_endpoint,
        grpc_data_endpoint,
        server_type,
        **kwargs,
    ):
        super().__init__(name=name, namespace=namespace, client=client, **kwargs)
        self.image = image
        self.supported_model_formats = supported_model_formats
        self.container_specs = containers
        self.grpc_endpoint = grpc_endpoint
        self.grpc_data_endpoint = grpc_data_endpoint
        self.server_type = server_type

    def to_dict(self):
        super().to_dict()
        self.res["metadata"]["annotations"] = {
            "enable-route": "true",
        }
        self.res["metadata"]["labels"] = {
            "name": "modelmesh-serving-ovms-1.x-SR",
        }
        self.res["spec"] = {
            "supportedModelFormats": self.supported_model_formats,
            "protocolVersions": [
                "grpc-v1",
            ],
            "multiModel": True,
            "grpcEndpoint": f"port:{self.grpc_endpoint}",
            "grpcDataEndpoint": f"port:{self.grpc_data_endpoint}",
            "containers": self.container_specs,
            "builtInAdapter": {
                "serverType": self.server_type,
                "runtimeManagementPort": 8888,
                "memBufferBytes": 134217728,
                "modelLoadingTimeoutMillis": 90000,
            },
        }