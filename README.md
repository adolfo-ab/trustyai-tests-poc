# TrustyAI Backend Integration/E2E Tests
A simple proof of concept of a testing framework to be used in [TrustyAI](https://github.com/trustyai-explainability).

## Overview
- Leverages [kubernetes python client](https://github.com/kubernetes-client/python) and [openshift-python-wrapper](https://github.com/RedHatQE/openshift-python-wrapper)
- The idea is to have the flexibility to do anything we would do through the command line, but programmatically through the K8S/OpenShift APIs, and to be able to work with cluster resources using simple Python objects.
- In this PoC, pytest was used (since pytest fixtures integrate nicely with openshift-python-wrapper), but similar results could be achieved using other testing frameworks
- These tests in principle could be run in different environments (vanilla K8S and OpenShift with OpenDataHub or OpenShift AI) with minimum effort, since they use the K8S/OpenShift API directly

## Directory
- [data](https://github.com/adolfo-ab/trustyai-tests/tree/main/data): train/test data to feed the models
- [resources](https://github.com/adolfo-ab/trustyai-tests/tree/main/resources): classes that define different K8S/OpenShift resources used in the tests. Most of these could be moved directly to openshift-python-wrapper.
- [tests](https://github.com/adolfo-ab/trustyai-tests/tree/main/tests): tests and pytest fixtures used in the PoC. Only a very simple test is provided here, just to demonstrate the possibilities of this approach.
- [utils](https://github.com/adolfo-ab/trustyai-tests/tree/main/utils): constants and util functions (send data to model, apply name mappings, etc.) used in the tests.

## Running the tests
- Log in an OpenShift cluster with OpenDataHub
- Make sure you have Poetry installed and install the project's dependencies with `poetry install`
- Run the tests with `pytest -s --log-cli-level=DEBUG tests/basic_test.py`

