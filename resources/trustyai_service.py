from ocp_resources.resource import NamespacedResource

from utils.constants import TRUSTYAI_API_GROUP


class TrustyAIService(NamespacedResource):
    api_group = TRUSTYAI_API_GROUP

    def __init__(
            self,
            name,
            namespace,
            client,
            **kwargs,
    ):
        super().__init__(name=name, namespace=namespace, client=client, **kwargs)

    def to_dict(self):
        super().to_dict()
        self.res["apiVersion"] = "trustyai.opendatahub.io/v1alpha1"
        self.res["kind"] = "TrustyAIService"
        self.res["metadata"] = {
            "name": self.name,
        }
        self.res["spec"] = {
            "replicas": 1,
            "image": "quay.io/trustyaiservice/trustyai-service",
            "tag": "latest",
            "storage": {
                "format": "PVC",
                "folder": "/inputs",
                "size": "1Gi",
            },
            "data": {
                "filename": "data.csv",
                "format": "CSV",
            },
            "metrics": {
                "schedule": "5s",
            },
        }
