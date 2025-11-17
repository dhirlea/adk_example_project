"""
Main script running the multi-agent weather app https://google.github.io/adk-docs/tutorials/agent-team/
"""

import os
import asyncio
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


# --- Define Model Constants for easier use ---

# More supported models can be referenced here: https://ai.google.dev/gemini-api/docs/models#model-variations
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

# More supported models can be referenced here: https://docs.litellm.ai/docs/providers/openai#openai-chat-completion-models
MODEL_GPT_4O = "openai/gpt-4.1"  # You can also try: gpt-4.1-mini, gpt-4o etc.

# More supported models can be referenced here: https://docs.litellm.ai/docs/providers/anthropic
MODEL_CLAUDE_SONNET = "anthropic/claude-sonnet-4-20250514"  # You can also try: claude-opus-4-20250514 , claude-3-7-sonnet-20250219 etc

MODEL_LLAMA_3_1 = "ollama_chat/llama3.1-modified:latest"

print("\nEnvironment configured.")

# @title Define the Weather Agent
# Use one of the model constants defined earlier
AGENT_MODEL = MODEL_GEMINI_2_0_FLASH  # Starting with Gemini (MODEL_GEMINI_2_0_FLASH) and can swap to locally hosted llama (MODEL_LLAMA_3_1)

root_agent = Agent(
    name="weather_agent_gemini_flash",
    model=AGENT_MODEL,  # Can be a string for Gemini or a LiteLlm object
    description="Provides weather information for specific cities.",
    instruction="You are a helpful weather assistant. "
                "When the user asks for the weather in a specific city, "
                "use the 'get_weather' tool to find the information. "
                "If the tool returns an error, inform the user politely. "
                "If the tool is successful, present the weather report clearly.",
    tools=[get_weather],  # Pass the function directly
)

print(f"Agent '{root_agent.name}' created using model '{AGENT_MODEL}'.")

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

    # Example tool usage (optional test)
    print("\n Testing the get_weather tool")
    print(get_weather("New York"))
    print(get_weather("Paris"))
    print("\n get_weather tool test completed")

    # @title Setup Session Service and Runner

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

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())
