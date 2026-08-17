"""
Azure ML deployment skeleton.

Prerequisites:
    pip install azure-ai-ml azure-identity
    az login

Set these environment variables:
    AZURE_SUBSCRIPTION_ID
    AZURE_RESOURCE_GROUP
    AZURE_ML_WORKSPACE

This script intentionally does not contain real subscription/resource identifiers.
"""

import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, ManagedOnlineEndpoint, ManagedOnlineDeployment, CodeConfiguration, Model
from azure.identity import DefaultAzureCredential

SUBSCRIPTION_ID = os.environ["AZURE_SUBSCRIPTION_ID"]
RESOURCE_GROUP = os.environ["AZURE_RESOURCE_GROUP"]
WORKSPACE = os.environ["AZURE_ML_WORKSPACE"]

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE
)

print("Connected to Azure ML workspace:", WORKSPACE)
print("Upload/register models and create an online endpoint using the Azure ML SDK.")
print("Keep API keys and customer data outside source control.")
