# Project aimed to demonstrate Google's ADK functionality

https://google.github.io/adk-docs/get-started/python/

## 1. Getting Started

### 1.1 Build Project Dependencies
From the root directory of the code repository, run the following commands in a terminal window.

```
conda env create -f requirements.yaml
conda activate google_adk_env
``` 

If you want to update the environment to add extra dependencies, update the *requirements.yaml* then run the following:

```
conda activate google_adk_env
conda env update --file requirements.yaml
```

Run the adk create command to start a new agent project. Need to create a Google AI Studio/ Vertex AI Studio API key https://aistudio.google.com/api-keys

```
adk create my_agent
```

The following models are available in Google AI Studio https://ai.google.dev/gemini-api/docs/models.

### 1.2 Build Project Dependencies
Make sure the project structure follows the guidelines here https://github.com/google/adk-docs/tree/main/examples/python/tutorial/agent_team/adk-tutorial.

A *root_agent* variable needs to be defined in *agent.py* in order for the adk-prefixed commands to work.

```text
root/
├── agent_project 1/
│   ├── __init__.py
│   ├── agent.py      # Agent definition for project 1
│   └── .env          # API Key configuration for all LLMs in project 1

```

### 1.3 Run The Application Locally

**CD** into the root project directory **Google-ADK-Project** and run your agent using the adk run command-line tool.

``` 
adk run my_agent
```

Run with web interface

``` 
adk web --port 8000
```

Expose server as an API 

``` 
adk api_server
``` 


### 1.4 Setting up open-source models

https://google.github.io/adk-docs/agents/models/#ollama-integration