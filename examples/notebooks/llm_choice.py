"""
Very simple utility to manage the choice of LLM service
"""

import os

azureOpenAIRequiredVariables = {
    'AZURE_OPENAI_API_VERSION',
    'AZURE_OPENAI_API_BASE',
    'AZURE_OPENAI_API_KEY',
    'AZURE_OPENAI_LLM_DEPLOYMENT',
    'AZURE_OPENAI_LLM_MODEL',
    'AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT',
    'AZURE_OPENAI_EMBEDDINGS_MODEL',
}

# functions from env var map to boolean (whereby True means valid)
providerValidator = {
    'Azure_OpenAI': lambda envMap: all(k in envMap for k in azureOpenAIRequiredVariables),
    'GCP_VertexAI': lambda envMap: 'GOOGLE_APPLICATION_CREDENTIALS' in envMap,
    'OpenAI': lambda envMap: 'OPENAI_API_KEY' in envMap,
}

def suggestLLMProvider():
    #
    preferredProvider = os.environ.get('PREFERRED_LLM_PROVIDER')
    if preferredProvider and providerValidator.get(preferredProvider, lambda envMap: False)(os.environ):
        return preferredProvider
    else:
        for pName, pValidator in sorted(providerValidator.items()):
            if pValidator(os.environ):
                return pName
        raise ValueError('No available credentials for LLMs')