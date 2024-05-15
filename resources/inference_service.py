from ocp_resources.resource import NamespacedResource

from utils.constants import KSERVE_API_GROUP


class InferenceService(NamespacedResource):
    api_group = KSERVE_API_GROUP

    def __init__(
            self,
            name,
            namespace,
            path,
            storage_name,
            model_format_name,
            runtime,
            client,
            **kwargs,
    ):
        super().__init__(name=name, namespace=namespace, client=client, **kwargs)
        self.name = name
        self.path = path
        self.runtime = runtime
        self.model_format_name = model_format_name
        self.storage_name = storage_name

    def to_dict(self):
        super().to_dict()

        self.res["metadata"]["annotations"] = {
            "serving.kserve.io/deploymentMode": "ModelMesh"
        }
        self.res["spec"] = {
            "predictor": {
                "model": {
                    "modelFormat": {
                        "name": self.model_format_name,
                    },
                    "runtime": self.runtime,
                    "storage": {
                        "key": self.storage_name,
                        "path": self.path,
                    },
                },
            },
        }
