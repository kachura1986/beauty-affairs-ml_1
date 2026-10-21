import logging
from functools import lru_cache

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from fastapi import Request
from openai import AsyncAzureOpenAI

from application import AgentService
from core.logger import setup_logging
from services.embedding import AzureOpenAIEmbedder
from services.llm import LLMModel


# Logger
@lru_cache(maxsize=1)
def get_logger() -> logging.Logger:
    return setup_logging("app.log")


# Dependency providers
# LLM
def get_azure_openai_client(request: Request) -> AsyncAzureOpenAI:
    return request.app.state.azure_openai_client


def get_azure_model(request: Request) -> LLMModel:
    return request.app.state.azure_model


def get_azure_embedder(request: Request) -> AzureOpenAIEmbedder:
    return request.app.state.azure_embedder


# Agent
def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


# Searching
def get_search_index_client(request: Request) -> SearchIndexClient:
    return request.app.state.search_index_client


def get_search_credential(request: Request) -> AzureKeyCredential:
    return request.app.state.search_credential
