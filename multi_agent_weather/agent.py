"""
Main script running the multi-agent weather app https://google.github.io/adk-docs/tutorials/agent-team/
"""

import os
import asyncio
from typing import Optional, Dict, Any  # Make sure to import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
# from google.adk.models.lite_llm import LiteLlm  # For multi-model support
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types  # For creating message Content/Parts

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.ERROR)

# Gemini API Key (Get from Google AI Studio: https://aistudio.google.com/app/apikey)
# Set this directly in the script to run it as a script instead of using adk run.
# Useful when debugging agentic team state via the async main function
os.environ["GOOGLE_API_KEY"] = ""


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


def get_weather_stateful(city: str, tool_context: ToolContext) -> dict:
    """Retrieves weather, converts temp unit based on session state."""
    print(f"--- Tool: get_weather_stateful called for {city} ---")

    # --- Read preference from state ---
    preferred_unit = tool_context.state.get("user_preference_temperature_unit", "Celsius")  # Default to Celsius
    print(f"--- Tool: Reading state 'user_preference_temperature_unit': {preferred_unit} ---")

    city_normalized = city.lower().replace(" ", "")

    # Mock weather data (always stored in Celsius internally)
    mock_weather_db = {
        "newyork": {"temp_c": 25, "condition": "sunny"},
        "london": {"temp_c": 15, "condition": "cloudy"},
        "tokyo": {"temp_c": 18, "condition": "light rain"},
    }

    if city_normalized in mock_weather_db:
        data = mock_weather_db[city_normalized]
        temp_c = data["temp_c"]
        condition = data["condition"]

        # Format temperature based on state preference
        if preferred_unit == "Fahrenheit":
            temp_value = (temp_c * 9 / 5) + 32  # Calculate Fahrenheit
            temp_unit = "°F"
        else:  # Default to Celsius
            temp_value = temp_c
            temp_unit = "°C"

        report = f"The weather in {city.capitalize()} is {condition} with a temperature of {temp_value:.0f}{temp_unit}."
        result = {"status": "success", "report": report}
        print(f"--- Tool: Generated report in {preferred_unit}. Result: {result} ---")

        # Example of writing back to state (optional for this tool)
        tool_context.state["last_city_checked_stateful"] = city
        print(f"--- Tool: Updated state 'last_city_checked_stateful': {city} ---")

        return result
    else:
        # Handle city not found
        error_msg = f"Sorry, I don't have weather information for '{city}'."
        print(f"--- Tool: City '{city}' not found. ---")

        tool_context.state["last_city_checked_stateful"] = None
        print(f"--- Tool: Updated state 'last_city_checked_stateful': None ---")

        return {"status": "error", "error_message": error_msg}


print("✅ State-aware 'get_weather_stateful' tool defined.")


def change_state_temperature_units(units: str, tool_context: ToolContext) -> None:
    """Changes the state temperature units between Celsius and Fahrenheit

        Args:
        units (str): The unit to measure temperature in. Can be either Fahrenheit or Celsius

    Returns:
        None"""
    tool_context.state['user_preference_temperature_unit'] = units


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


print("\nGreeting, Farewell and Change Tem Units tools defined.")


def block_keyword_guardrail(
        callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Inspects the latest user message for 'BLOCK'. If found, blocks the LLM call
    and returns a predefined LlmResponse. Otherwise, returns None to proceed.
    """
    agent_name = callback_context.agent_name  # Get the name of the agent whose model call is being intercepted
    print(f"--- Callback: block_keyword_guardrail running for agent: {agent_name} ---")

    # Extract the text from the latest user message in the request history
    last_user_message_text = ""
    if llm_request.contents:
        # Find the most recent message with role 'user'
        for content in reversed(llm_request.contents):
            if content.role == 'user' and content.parts:
                # Assuming text is in the first part for simplicity
                if content.parts[0].text:
                    last_user_message_text = content.parts[0].text
                    break  # Found the last user message text

    print(f"--- Callback: Inspecting last user message: '{last_user_message_text[:100]}...' ---")  # Log first 100 chars

    # --- Guardrail Logic ---
    keyword_to_block = "BLOCK"
    if keyword_to_block in last_user_message_text.upper():  # Case-insensitive check
        print(f"--- Callback: Found '{keyword_to_block}'. Blocking LLM call! ---")
        # Optionally, set a flag in state to record the block event
        callback_context.state["guardrail_block_keyword_triggered"] = True
        print(f"--- Callback: Set state 'guardrail_block_keyword_triggered': True ---")

        # Construct and return an LlmResponse to stop the flow and send this back instead
        return LlmResponse(
            content=types.Content(
                role="model",  # Mimic a response from the agent's perspective
                parts=[types.Part(
                    text=f"I cannot process this request because it contains the blocked keyword '{keyword_to_block}'.")],
            )
            # Note: You could also set an error_message field here if needed
        )
    else:
        # Keyword not found, allow the request to proceed to the LLM
        print(f"--- Callback: Keyword not found. Allowing LLM call for {agent_name}. ---")
        return None  # Returning None signals ADK to continue normally


print("✅ block_keyword_guardrail function defined.")


def block_paris_tool_guardrail(
        tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    """
    Checks if 'get_weather_stateful' is called for 'Paris'.
    If so, blocks the tool execution and returns a specific error dictionary.
    Otherwise, allows the tool call to proceed by returning None.
    """
    tool_name = tool.name
    agent_name = tool_context.agent_name  # Agent attempting the tool call
    print(f"--- Callback: block_paris_tool_guardrail running for tool '{tool_name}' in agent '{agent_name}' ---")
    print(f"--- Callback: Inspecting args: {args} ---")

    # --- Guardrail Logic ---
    target_tool_name = "get_weather_stateful"  # Match the function name used by FunctionTool
    blocked_city = "paris"

    # Check if it's the correct tool and the city argument matches the blocked city
    if tool_name == target_tool_name:
        city_argument = args.get("city", "")  # Safely get the 'city' argument
        if city_argument and city_argument.lower() == blocked_city:
            print(f"--- Callback: Detected blocked city '{city_argument}'. Blocking tool execution! ---")
            # Optionally update state
            tool_context.state["guardrail_tool_block_triggered"] = True
            print(f"--- Callback: Set state 'guardrail_tool_block_triggered': True ---")

            # Return a dictionary matching the tool's expected output format for errors
            # This dictionary becomes the tool's result, skipping the actual tool run.
            return {
                "status": "error",
                "error_message": f"Policy restriction: Weather checks for '{city_argument.capitalize()}' are currently disabled by a tool guardrail."
            }
        else:
            print(f"--- Callback: City '{city_argument}' is allowed for tool '{tool_name}'. ---")
    else:
        print(f"--- Callback: Tool '{tool_name}' is not the target tool. Allowing. ---")

    # If the checks above didn't return a dictionary, allow the tool to execute
    print(f"--- Callback: Allowing tool '{tool_name}' to proceed. ---")
    return None  # Returning None allows the actual tool function to run


print("✅ block_paris_tool_guardrail function defined.")

# --- Define Model Constants for easier use ---

# More supported models can be referenced here: https://ai.google.dev/gemini-api/docs/models#model-variations
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"

# More supported models can be referenced here: https://docs.litellm.ai/docs/providers/openai#openai-chat-completion-models
MODEL_GPT_4O = "openai/gpt-4.1"  # You can also try: gpt-4.1-mini, gpt-4o etc.

# More supported models can be referenced here: https://docs.litellm.ai/docs/providers/anthropic
MODEL_CLAUDE_SONNET = "anthropic/claude-sonnet-4-20250514"  # You can also try: claude-opus-4-20250514 , claude-3-7-sonnet-20250219 etc

MODEL_LLAMA_3_2 = "ollama_chat/llama3.2-modified"

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
        model=MODEL_GEMINI_2_0_FLASH,
        # LiteLlm(MODEL_LLAMA_3_1) systematically calls the wrong tools/passes to incorrect agent
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
        model=MODEL_GEMINI_2_0_FLASH,
        # LiteLlm(MODEL_LLAMA_3_1) systematically calls the wrong tools/passes to incorrect agent
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
        description="Main agent: Provides weather (state-aware unit), delegates greetings/farewells to sub-agents, "
                    "saves report to state.",
        instruction="You are the main Weather Agent. Your jobs are to provide weather using the 'get_weather_stateful' "
                    "tool and to change the temperature units using the 'change_state_temperature_units' tool. "
                    "Delegate simple greetings to 'greeting_agent' and farewells to 'farewell_agent'. "
                    "Handle only weather requests, temperature unit changes, greetings, and farewells.",
        tools=[get_weather_stateful, change_state_temperature_units],
        # Root agent needs the weather tool for its core task
        sub_agents=[greeting_agent, farewell_agent],  # Include sub-agents
        output_key="last_weather_report",
        # Write out to state dictionary final root agent response under last_weather_report key
        before_model_callback=block_keyword_guardrail,  # <<< Assign the guardrail callback
        before_tool_callback=block_paris_tool_guardrail  # <<< Add tool guardrail
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

    # --- Session Management ---
    # Key Concept: SessionService stores conversation history & state.
    # InMemorySessionService is simple, non-persistent storage for this tutorial.
    session_service = InMemorySessionService()

    # Define constants for identifying the interaction context
    APP_NAME = "weather_tutorial_app"
    USER_ID = "user_1"
    SESSION_ID = "session_001"  # Using a fixed ID for simplicity

    # Define initial state data - user prefers Celsius initially
    initial_state = {"user_preference_temperature_unit": "Celsius"}

    # Create the specific session where the conversation will happen
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state  # <<< Initialize state during session creation
    )

    # --- Runner ---
    # Key Concept: Runner orchestrates the agent execution loop.
    runner = Runner(
        agent=root_agent,  # The agent we want to run
        app_name=APP_NAME,  # Associates runs with our app
        session_service=session_service  # Uses our session manager
    )

    await call_agent_async(query="BLOCK the weather in London",  # should trigger model callback
                           runner=runner,
                           user_id=USER_ID,
                           session_id=SESSION_ID)

    await call_agent_async(query="What's it like in Paris?",  # should trigger tool callback
                           runner=runner,
                           user_id=USER_ID,
                           session_id=SESSION_ID)

    await call_agent_async(query="What's it like in Tokyo?", # should know the answer
                           runner=runner,
                           user_id=USER_ID,
                           session_id=SESSION_ID)

    await call_agent_async(query='Goodbye',  # should delegate and say goodbye
                           runner=runner,
                           user_id=USER_ID,
                           session_id=SESSION_ID)

    # Verify that the BLOCK keyword guardrail flag has been set correctly

    final_session = await session_service.get_session(app_name=APP_NAME,
                                                      user_id=USER_ID,
                                                      session_id=SESSION_ID)

    print(f"Guardrail Triggered Flag: "
          f"{final_session.state.get('guardrail_block_keyword_triggered', 'Not Set (or False)')}")

    print(f"Guardrail Triggered Flag: "
          f"{final_session.state.get('guardrail_tool_block_triggered', 'Not Set (or False)')}")

    print(
        f"Last Weather Report: {final_session.state.get('last_weather_report', 'Not Set')}")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")

    # Example tool usage (optional test)
    # print("\n Testing the get_weather tool")
    # print(get_weather("New York"))
    # print(get_weather("Paris"))
    # print("\n get_weather tool test completed")
    # print(say_hello(name='Dan'))
    # print(say_hello(name=None))
    # print(say_goodbye())
