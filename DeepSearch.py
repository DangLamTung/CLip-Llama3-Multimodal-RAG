

from opendeepsearch import OpenDeepSearchTool
from smolagents import CodeAgent, LiteLLMModel
import os

# Set environment variables for API keys
os.environ["SERPER_API_KEY"] = "b6ff922f76ebe3c41760ce93f6dfca48ee3cf3c6"
os.environ["OPENROUTER_API_KEY"] = "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"
os.environ["JINA_API_KEY"] = "jina_1255762ef83e40b9a7469eb8f4f4e146qaA4WxUxW8_E5Mpp6UcRF3yyeazS"

search_agent = OpenDeepSearchTool(model_name="openrouter/google/gemini-2.5-pro-exp-03-25:free", reranker="jina") # Set reranker to "jina" or "infinity"
model = LiteLLMModel(
    "openrouter/google/gemini-2.0-flash-001",
    temperature=0.2
)

code_agent = CodeAgent(tools=[search_agent], model=model)
query = "Research about flash attention, what make it so effective"
result = code_agent.run(query)

print(result)