"""
Main script running the multi-agent weather app https://google.github.io/adk-docs/tutorials/agent-team/
"""

import asyncio
from typing import Optional  # Make sure to import Optional

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm  # For multi-model support
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types  # For creating message Content/Parts

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.ERROR)


# @title Define the get_weather Tool
def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"--- Tool: get_weather called for city: {city} ---")  # Log tool execution
    city_normalized = city.lower().replace(" ", "")  # Basic normalization

    # Mock weather data
    mock_weather_db = {
        "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
        "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
        "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}


print("\nWeather retrieval tool defined.")


def say_hello(name: Optional[str] = None) -> str:
    """Provides a simple greeting. If a name is provided, it will be used.

    Args:
        name (str, optional): The name of the person to greet. Defaults to a generic greeting if not provided.

    Returns:
        str: A friendly greeting message.
    """
    if name:
        greeting = f"Hello, {name}!"
        print(f"--- Tool: say_hello called with name: {name} ---")
    else:
        greeting = "Hello there!"  # Default greeting if name is None or not explicitly passed
        print(f"--- Tool: say_hello called without a specific name (name_arg_value: {name}) ---")
    return greeting


def say_goodbye() -> str:
    """Provides a simple farewell message to conclude the conversation."""
    print(f"--- Tool: say_goodbye called ---")
    return "Goodbye! Have a great day."


print("\nGreeting and Farewell tools defined.")

# --- Define Model Constants for easier use ---

# More supported models can be referenced here: https://ai.google.dev/gemini-api/docs/models#model-variations
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

# More supported models can be referenced here: https://docs.litellm.ai/docs/providers/openai#openai-chat-completion-models
MODEL_GPT_4O = "openai/gpt-4.1"  # You can also try: gpt-4.1-mini, gpt-4o etc.

# More supported models can be referenced here: https://docs.litellm.ai/docs/providers/anthropic
MODEL_CLAUDE_SONNET = "anthropic/claude-sonnet-4-20250514"  # You can also try: claude-opus-4-20250514 , claude-3-7-sonnet-20250219 etc

MODEL_LLAMA_3_1 = "ollama_chat/llama3.1-modified:latest"

MODEL_GPT_OSS_20b = "ollama_chat/gpt-oss:20b-modified"

print("\nEnvironment configured.")

# @title Define the Weather Agent
# Use one of the model constants defined earlier
# Starting with Gemini (MODEL_GEMINI_2_0_FLASH) and can swap to locally hosted llama (MODEL_LLAMA_3_1)
AGENT_MODEL = MODEL_GEMINI_2_0_FLASH

# Create a welcome and a farewell sub-agent
# If you want to use models other than Gemini, Ensure LiteLlm is imported and API keys are set
# from google.adk.models.lite_llm import LiteLlm
# MODEL_GPT_4O, MODEL_CLAUDE_SONNET etc. should be defined
# Or else, continue to use: model = MODEL_GEMINI_2_0_FLASH

# --- Greeting Agent ---
greeting_agent = None
try:
    greeting_agent = Agent(
        # Using a potentially different/cheaper model for a simple task
        model=MODEL_GEMINI_2_0_FLASH,  # LiteLlm(MODEL_LLAMA_3_1) systematically calls the wrong tools/passes to incorrect agent
        # model=MODEL_GEMINI_2_0_FLASH, # If you would like to use the default flash gemini
        name="greeting_agent",
        instruction="You are the Greeting Agent  powered by Llama-3.1. "
                    "Your ONLY task is to provide a friendly greeting to the user. "
                    "Use the 'say_hello' tool to generate the greeting. "
                    "If the user provides their name, make sure to pass it to the tool. "
                    "Do not perform any other actions.",
        description="Handles simple greetings and hellos using the 'say_hello' tool and passes back to "
                    "weather orchestrator agent.",  # Crucial for delegation
        tools=[say_hello],
    )
    print(f"✅ Agent '{greeting_agent.name}' created using model '{greeting_agent.model}'.")
except Exception as e:
    print(f"❌ Could not create Greeting agent. Check API Key ({greeting_agent.model}). Error: {e}")

# --- Farewell Agent ---
farewell_agent = None
try:
    farewell_agent = Agent(
        # Can use the same or a different model
        model=MODEL_GEMINI_2_0_FLASH,  # LiteLlm(MODEL_LLAMA_3_1) systematically calls the wrong tools/passes to incorrect agent
        # model=MODEL_GEMINI_2_0_FLASH, # If you would like to use the default flash gemini
        name="farewell_agent",
        instruction="You are the Farewell Agent powered by Llama-3.1. "
                    "Your ONLY task is to provide a polite goodbye message. "
                    "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the "
                    "conversation (e.g., using words like 'bye', 'goodbye', 'thanks bye', 'see you'). "
                    "Do not perform any other actions.",
        description="Handles simple farewells and goodbyes using the 'say_goodbye' tool and passes back to "
                    "weather orchestrator agent.",  # Crucial for delegation
        tools=[say_goodbye],
    )
    print(f"✅ Agent '{farewell_agent.name}' created using model '{farewell_agent.model}'.")
except Exception as e:
    print(f"❌ Could not create Farewell agent. Check API Key ({farewell_agent.model}). Error: {e}")

# --- Root Agent --- It is mandatory to define a root_agent in order for the adk to run!
# Ensure sub-agents were created successfully before defining the root agent.
# Also ensure the original 'get_weather' tool is defined.
root_agent = None

if greeting_agent and farewell_agent and 'get_weather' in globals():
    # Let's use a capable Gemini model for the root agent to handle orchestration
    root_agent_model = MODEL_GEMINI_2_0_FLASH

    root_agent = Agent(
        name="weather_orchestrator_agent",
        model=root_agent_model,
        description="The main coordinator agent. Handles weather requests and delegates greetings/farewells "
                    "to specialists.",
        instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is "
                    "to provide weather information. Use the 'get_weather' tool ONLY for specific weather requests "
                    "(e.g., 'weather in London'). You have specialized sub-agents: "
                    "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
                    "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
                    "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. "
                    "If it's a farewell, delegate to 'farewell_agent'. "
                    "If it's a weather request, handle it yourself using 'get_weather'. "
                    "For anything else, respond appropriately or state you cannot handle it.",
        tools=[get_weather],  # Root agent still needs the weather tool for its core task
        # Key change: Link the sub-agents here!
        sub_agents=[greeting_agent, farewell_agent]
    )
    print(f"✅ Root Agent '{root_agent.name}' created using model '{root_agent_model}' "
          f"with sub-agents: {[sa.name for sa in root_agent.sub_agents]}")

else:
    print("❌ Cannot create root agent because one or more sub-agents failed to initialize "
          "or 'get_weather' tool is missing.")
    if not greeting_agent:
        print(" - Greeting Agent is missing.")
    if not farewell_agent:
        print(" - Farewell Agent is missing.")
    if 'get_weather' not in globals():
        print(" - get_weather function is missing.")


# Self-hosted ollama model example
# root_agent = Agent(
#     model=LiteLlm(model=AGENT_MODEL),
#     name="weather_agent_llama",
#     description=(
#         "Provides weather information (using Llama-3.1 with custom prompt)"
#     ),
#     instruction="You are a helpful weather assistant powered by Llama-3.1. "
#                 "Use the 'get_weather' tool for city weather requests. "
#                 "Clearly present successful reports or polite error messages based on the tool's output status.",
#     tools=[
#         get_weather,
#     ],
# )
#
# print(f"Agent '{root_agent.name}' created using model '{AGENT_MODEL}'.")


# @title Define Agent Interaction Function
async def call_agent_async(query: str, runner, user_id, session_id):
    """Sends a query to the agent and prints the final response."""
    print(f"\n>>> User Query: {query}")

    # Prepare the user's message in ADK format
    content = types.Content(role='user', parts=[types.Part(text=query)])

    final_response_text = "Agent did not produce a final response."  # Default

    # Key Concept: run_async executes the agent logic and yields Events.
    # We iterate through events to find the final answer.
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # You can uncomment the line below to see *all* events during execution
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        # Key Concept: is_final_response() marks the concluding message for the turn.
        if event.is_final_response():
            if event.content and event.content.parts:
                # Assuming text response in the first part
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:  # Handle potential errors/escalations
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            # Add more checks here if needed (e.g., specific error codes)
            break  # Stop processing events once the final response is found

    print(f"<<< Agent Response: {final_response_text}")


async def main():
    """
    Main app contents
    :return:
    """

    try:

        # --- Session Management ---
        # Key Concept: SessionService stores conversation history & state.
        # InMemorySessionService is simple, non-persistent storage for this tutorial.
        session_service = InMemorySessionService()

        # Define constants for identifying the interaction context
        APP_NAME = "weather_tutorial_app"
        USER_ID = "user_1"
        SESSION_ID = "session_001"  # Using a fixed ID for simplicity

        # Create the specific session where the conversation will happen
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        print(f"Session created: App='{APP_NAME}', User='{USER_ID}', Session='{SESSION_ID}'")

        # --- Runner ---
        # Key Concept: Runner orchestrates the agent execution loop.
        runner = Runner(
            agent=root_agent,  # The agent we want to run
            app_name=APP_NAME,  # Associates runs with our app
            session_service=session_service  # Uses our session manager
        )
        print(f"Runner created for agent '{runner.agent.name}'.")

        await call_agent_async(query='Tell me the weather in London',
                               runner=runner,
                               user_id=USER_ID,
                               session_id=SESSION_ID)

    except Exception as ex:
        print(f"An error occurred: {ex}")


if __name__ == '__main__':
    asyncio.run(main())

    # Example tool usage (optional test)
    # print("\n Testing the get_weather tool")
    # print(get_weather("New York"))
    # print(get_weather("Paris"))
    # print("\n get_weather tool test completed")
    # print(say_hello(name='Dan'))
    # print(say_hello(name=None))
    # print(say_goodbye())
