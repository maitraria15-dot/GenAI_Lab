"""A YouTube-content AI assistant built with LangGraph and Gemini.

Run it as a terminal chat:
    set -a; source .env; set +a
    python assistant.py

Run it with LangGraph Studio (local deployment):
    set -a; source .env; set +a
    langgraph dev
"""

import os
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

import os
from dotenv import load_dotenv

load_dotenv()

print("Tavily configured:", bool(os.getenv("TAVILY_API_KEY")))

from langchain_tavily import TavilySearch

# tool = TavilySearch(max_results=2)

# from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# TOOL 1: Create a reusable YouTube video brief from the user's idea.
# Tools are normal Python functions. The @tool decorator makes them available to Gemini.

# Define a Python function that creates a structured YouTube video brief.
# topic, target_audience, and video_length_minutes are inputs that the function needs.
# ": str" indicates that topic and target_audience should be strings.
# ": int" indicates that video_length_minutes should be an integer.
# "-> str" indicates that the function will return a string.
# The triple-quoted text is the function's docstring, which describes what the function does.
# The f"""...""" creates a multi-line formatted string and inserts the input values using { }.
# The function returns a complete YouTube brief containing a title, hook, outline, and call to action.
@tool
def create_video_brief(topic: str, target_audience: str, video_length_minutes: int) -> str:
    """Create a practical YouTube video brief with a title, hook, outline, and call to action."""
    return f"""YouTube video brief
Topic: {topic}
Target audience: {target_audience}
Suggested length: {video_length_minutes} minutes

Suggested title: {topic}: a practical guide for {target_audience}
Hook: In the first 15 seconds, explain the result the viewer will get and why it matters.
Outline: Hook → problem → 3 useful teaching points → example → recap → call to action.
Call to action: Ask viewers to comment with their biggest question about {topic}."""


# TOOL 2: Save a script or outline locally as a Markdown file.
@tool
def save_draft(title: str, content: str) -> str:
    """Save a completed YouTube outline, script, or description as a local Markdown draft."""
    
    # Convert the YouTube title into a safe filename.
    # title.lower() converts the title to lowercase.
    # re.sub() finds one or more characters that are NOT letters, numbers, "_" or "-"   
    # and replaces them with "-".
    # .strip("-") removes any "-" from the beginning or end.
    # If the resulting name is empty, use "youtube-draft" as the default filename.
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", title.lower()).strip("-") or "youtube-draft"

    # Create a path pointing to the "drafts" folder located in the same directory as this Python file.
    # Path(__file__) gets the path of the current Python file.
    # .with_name("drafts") replaces the current file name with "drafts", creating the path to that folder.
    drafts_folder = Path(__file__).with_name("drafts")

    # Create the "drafts" folder if it does not already exist.
    # .mkdir() creates the directory at the specified path.
    # exist_ok=True prevents an error if the folder already exists.
    drafts_folder.mkdir(exist_ok=True)

    # Create the full file path for the YouTube draft.
    # drafts_folder specifies where the file will be stored.
    # f"{safe_name}.md" creates the filename using safe_name and adds the ".md" Markdown extension.
    # The "/" combines the folder path and filename into one complete file path.
    draft_path = drafts_folder / f"{safe_name}.md"

    # Write the YouTube title and content into the Markdown file.
    # write_text() creates the file and writes the specified text into it.
    # f"# {title}\n\n{content}\n" formats the title as a Markdown heading, adds the content, and inserts line breaks.
    # encoding="utf-8" ensures the file can correctly store characters such as emojis and non-English text.
    draft_path.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")

    # Return a confirmation message showing the name of the saved draft file.
    # f"..." creates a formatted string and inserts the file name using draft_path.name.
    # draft_path.name gets only the file name, not the complete folder path.
    return f"Saved the draft to {draft_path.name}."


# TOOL 3: Let the assistant see which drafts already exist.
@tool
def list_drafts() -> str:
    """List the locally saved YouTube draft files."""

    # Create a path pointing to the "drafts" folder located in the same directory as this Python file.
    # Path(__file__) gets the path of the current Python file.
    # .with_name("drafts") replaces the current file name with "drafts", creating the path to that folder.
    drafts_folder = Path(__file__).with_name("drafts")

    # Create the "drafts" folder if it does not already exist.
    # .mkdir() creates a directory at the specified path.
    # exist_ok=True prevents an error if the folder already exists instead of raising an error.
    drafts_folder.mkdir(exist_ok=True)

    # Find all Markdown (.md) files inside the drafts folder and store their names in sorted order.
    # drafts_folder.glob("*.md") finds all files that have the ".md" extension.
    # path.name extracts only the filename from each matching file path.
    # sorted() arranges the filenames in alphabetical order.
    # The "for path in ..." syntax processes each matching file one by one.
    drafts = sorted(path.name for path in drafts_folder.glob("*.md"))

    # Return a message based on whether any drafts were found.
    # "if not drafts" checks whether the drafts list is empty.
    # If the list is empty, return "No saved drafts yet."
    # Otherwise, "\n- ".join(drafts) combines all draft names into a bulleted list.
    # "\n" creates a new line, and "- " adds a bullet before each draft name.
    # The conditional expression follows the pattern: value_if_true if condition else value_if_false.
    return "No saved drafts yet." if not drafts else "Saved drafts:\n- " + "\n- ".join(drafts)


# TOOL 4: Give the assistant a reliable timestamp for planning content.
@tool
def current_time() -> str:
    """Get the current local date and time."""
    # Return the current local date and time in a readable format.
    # datetime.now() gets the current date and time.
    # .astimezone() converts it to the local time zone.
    # .strftime(...) formats the date and time into a human-readable string.
    # "%A" gives the full weekday name, "%d" the day, "%B" the full month name,
    # "%Y" the year, "%H:%M" the time in 24-hour format, and "%Z" the time zone.
    return datetime.now().astimezone().strftime("%A, %d %B %Y, %H:%M %Z")


# TOOL 5: Search the web for current ideas, trends, or facts.
# This works only after you add TAVILY_API_KEY to .env.
@tool
def search_web(query: str) -> str:
    """Search the web for current, reliable information to use in YouTube content research."""

    # Check whether the TAVILY_API_KEY environment variable is available.
    # os.getenv("TAVILY_API_KEY") retrieves the API key from the environment.
    # "if not" checks whether the value is missing or empty.
    # If the API key is not configured, return a message asking the user to add it to the .env file.
    # This prevents the code from trying to use Tavily without the required API key.
    if not os.getenv("TAVILY_API_KEY"):
        return "Web search is not configured. Add TAVILY_API_KEY to .env to enable Tavily search."

    # Create a Tavily web-search tool configured to return up to 3 results.
    # TavilySearch(...) creates a Tavily search object with the specified settings.
    # max_results=3 limits the search to a maximum of 3 results.
    # search_depth="basic" uses a basic search depth for faster results.
    # topic="general" tells Tavily to perform a general-purpose web search.
    search = TavilySearch(max_results=3, search_depth="basic", topic="general")

    # Run the Tavily search using the user's query and store the returned results.
    # search.invoke(...) executes the configured Tavily search.
    # {"query": query} passes the user's search query as a dictionary to Tavily.
    # The returned search response is stored in the "response" variable.
    response = search.invoke({"query": query})

    # Extract the search results from the Tavily response.
    # isinstance(response, dict) checks whether the response is a dictionary.
    # If it is a dictionary, .get("results", []) retrieves the "results" value.
    # If "results" does not exist, .get() returns an empty list [] instead.
    # If the response is not a dictionary, use an empty list [].
    results = response.get("results", []) if isinstance(response, dict) else []

    # Check whether any search results were returned.
    # "if not results" checks whether the results list is empty or has no value.
    # If there are no results, return a message saying that no web-search results were found.
    if not results:
        return "No web-search results were returned."

    # Format all web-search results into one readable text string.
    # "\n\n".join() combines each search result and adds a blank line between results.
    # f"..." creates a formatted string containing the title, URL, and summary for each result.
    # item.get("title", "Untitled") gets the result title, or "Untitled" if no title is available.
    # item.get("url", "") gets the result URL, or an empty string if no URL is available.
    # item.get("content", "") gets the result summary/content, or an empty string if unavailable.
    # "for item in results" processes each search result one by one.
    return "\n\n".join(
        f"Title: {item.get('title', 'Untitled')}\nURL: {item.get('url', '')}\n"
        f"Summary: {item.get('content', '')}"
        for item in results
    )


# Put all available tools in one list. Add new tools here in future lessons.
tools = [create_video_brief, save_draft, list_drafts, current_time, search_web]

# Create the Gemini model and connect the tools to it.
# Create a Gemini LLM (Large Language Model) that the assistant will use to generate responses.
# ChatGoogleGenerativeAI(...) creates a connection to Google's Gemini model through LangChain.
# model="gemini-2.5-flash" specifies which Gemini model to use.
# temperature=0 makes the model's responses more deterministic and less random.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Connect the available tools to the Gemini model so it can request them when needed.
# model.bind_tools(tools) tells Gemini which tools it is allowed to use.
# "tools" contains the Python functions decorated with @tool.
# The model does not execute the tools itself; it can decide when to request a tool call.
# The tool-enabled model is stored in "model_with_tools" for use in the LangGraph workflow.
model_with_tools = model.bind_tools(tools)

# This instruction defines the assistant's role. Change it to change the assistant.
SYSTEM_PROMPT = """You are a helpful YouTube content assistant.
Help creators plan videos, write outlines, improve hooks, and draft scripts.
Use create_video_brief when a user needs a structured plan. Use save_draft only when
the user asks to save something. Use search_web when the user needs current facts,
recent trends, or sources. Be concise, practical, and friendly."""


# NODE: Send the current conversation to Gemini.
def call_model(state: MessagesState):

# Send the system instructions and the current conversation history to Gemini and get its response.
# SystemMessage(content=SYSTEM_PROMPT) tells Gemini the role and behavior it should follow.
# state["messages"] contains the conversation messages stored in the LangGraph state.
# The "*" unpacks all messages from state["messages"] into the list.
# model_with_tools.invoke(...) sends these messages to Gemini, along with the tools it can use.
# The model's response is stored in the "response" variable.
# Return the response as a "messages" list so LangGraph can add it to the conversation state.
# Here, state["messages"] retrieves the list defined by MessagesState
    response = model_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"]])
    return {"messages": [response]}
 

# GRAPH: This is the LangGraph workflow.
# Gemini responds first. If it requests a tool, ToolNode runs it, then Gemini reads
# the result and produces a final answer. If no tool is needed, the graph ends.

# Create a LangGraph workflow builder using MessagesState to manage the conversation state.
# StateGraph(...) creates the structure where we will define nodes, edges, and the flow of the agent.
# MessagesState is a pre-built state structure from LangGraph designed to store conversation messages.
# "builder" is the object we use to construct the graph before compiling it into an executable workflow.
builder = StateGraph(MessagesState)

# Add the "assistant" node to the LangGraph workflow.
# add_node() adds a step (node) to the graph.
# "assistant" is the name used to identify this node.
# call_model is the function that LangGraph will execute when this node is reached.
# In this workflow, the assistant node sends the conversation to Gemini and gets its response.
builder.add_node("assistant", call_model)

# Add a "tools" node to the LangGraph workflow to execute the tools requested by Gemini.
# ToolNode(tools) creates a pre-built LangGraph node that knows how to run the tools in the "tools" list.
# "tools" contains the Python functions decorated with @tool.
# "tools" is the name used to identify this node in the graph.
# When Gemini requests a tool, LangGraph routes the request to this node, which executes the appropriate tool.
builder.add_node("tools", ToolNode(tools))

# Connect the START point of the LangGraph workflow to the "assistant" node.
# START is the special entry point of the graph.
# add_edge() creates a connection that defines the direction of the workflow.
# This means the workflow always begins by running the "assistant" node.
builder.add_edge(START, "assistant")

# Add a conditional connection after the "assistant" node to decide what happens next.
# add_conditional_edges() allows LangGraph to choose the next node based on the current state.
# "assistant" is the node where Gemini has just responded.
# tools_condition checks whether Gemini's response contains a request to call a tool.
# If a tool is requested, LangGraph routes the workflow to the "tools" node.
# If no tool is requested, the workflow moves toward the end of the graph.
builder.add_conditional_edges("assistant", tools_condition)

# Connect the "tools" node back to the "assistant" node.
# After a tool is executed, its result is added to the conversation state.
# LangGraph then sends the updated state back to the assistant so Gemini can use the tool result.
# This creates the tool-calling loop: assistant → tools → assistant.
builder.add_edge("tools", "assistant")

# Compile the LangGraph workflow into an executable graph.
# builder contains the nodes and edges that define how the workflow should run.
# .compile() validates and assembles these components into a runnable graph.
# The compiled graph is stored in "graph", which can now be executed using graph.invoke().
graph = builder.compile()


# This section only runs when you start `python assistant.py` in the terminal.
# LangGraph Studio imports `graph` above and does not start this chat loop.

# Define a function that runs the YouTube assistant as a terminal-based chat.
# -> None indicates that this function does not return a value.
# os.getenv("GOOGLE_API_KEY") checks whether the Google API key is available in the environment.
# "if not" checks whether the API key is missing or empty.
# raise SystemExit(...) stops the program and displays an error message if the API key is not configured.
# This prevents the assistant from running without the required Gemini API key.
def run_terminal_chat() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is missing. Load .env before running this file.")


    # Create an empty list to store the conversation history.
    # chat_history will keep track of previous user and assistant messages.
    # print(...) displays a message in the terminal telling the user that the assistant is ready.
    # "\n" adds an extra blank line after the message.
    chat_history = []
    print("YouTube assistant ready! Type 'quit' to stop.\n")

    # Keep the terminal chat running continuously until the user chooses to quit.
    # while True creates an infinite loop that keeps asking the user for input.
    # input("You: ") waits for the user to type a message in the terminal.
    # .strip() removes unnecessary spaces from the beginning and end of the input.
    # user_text.lower() converts the input to lowercase so "QUIT", "Quit", etc. are also recognized.
    # If the user types "quit", print a goodbye message and break out of the loop.
    # If the user enters nothing, continue skips the rest of the loop and asks for input again.

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() == "quit":
            print("Assistant: Goodbye!")
            break
        if not user_text:
            continue

        # Invoke the complete LangGraph workflow, including any tools Gemini chooses.
        
        # Run the complete LangGraph workflow with the user's new message and the previous conversation history.
        # graph.invoke(...) executes the compiled LangGraph workflow.
        # {"messages": [...]} provides the conversation messages as the graph's input state.
        # *chat_history unpacks all previous messages so the assistant remembers the conversation.
        #  HumanMessage(content=user_text) converts the user's latest input into a LangChain message object.
        # The result returned by the graph is stored in "result".
        # chat_history = result["messages"] updates the conversation history with the latest state returned by LangGraph.
        # chat_history[-1] gets the last message in the conversation, which is the assistant's latest response.
        # .content extracts the actual text from that message.
        # print(...) displays the assistant's response in the terminal.
        #The expression [*chat_history, HumanMessage(content=user_text)] is not two separate items when invoke receives it. The * symbol takes everything out of the chat_history list and pours it into a brand-new list alongside the new message.
        result = graph.invoke({"messages": [*chat_history, HumanMessage(content=user_text)]})
        chat_history = result["messages"]
        print(f"Assistant: {chat_history[-1].content}\n")

# Run the terminal chat only when this Python file is executed directly.
# __name__ is a special Python variable that tells us how the file is being run.
# When the file is run directly, Python sets __name__ to "__main__".
# If the file is imported into another Python file, this condition is False.
# run_terminal_chat() starts the terminal-based YouTube assistant.
if __name__ == "__main__":
    run_terminal_chat()
