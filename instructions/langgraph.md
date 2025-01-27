LangGraph Tutorial: Building Stateful Multi-Actor Applications with Language Models
Welcome to the comprehensive tutorial on LangGraph, a powerful library designed for building stateful, multi-actor applications using Large Language Models (LLMs). Whether you're creating a single-agent chatbot or a complex multi-agent workflow, LangGraph provides the tools and infrastructure to streamline your development process.
Table of Contents
Introduction to LangGraph
Overview
Why Use LangGraph?
LangGraph Platform
Installation
LangGraph Quickstart
Part 1: Building a Basic Chatbot
Defining the State
Adding Nodes
Running the Chatbot
Full Code Example
Part 2: Enhancing the Chatbot with Tools
Integrating a Search Tool
Tool Nodes and Conditional Edges
Running the Enhanced Chatbot
Full Code Example
Part 3: Adding Memory to the Chatbot
Implementing Checkpointing
Persistent State Management
Running with Memory
Full Code Example
Part 4: Human-in-the-loop
Incorporating Human Assistance
Using Interrupts and Commands
Running with Human Oversight
Full Code Example
Part 5: Customizing State
Extending the State Schema
State Updates and Management
Running with Custom State
Full Code Example
Part 6: Time Travel
Understanding Time Travel
Rewinding and Resuming Execution
Running with Time Travel
Full Code Example
Advanced Concepts
Workflows and Agents
The Augmented LLM
Prompt Chaining
Parallelization
Routing
Orchestrator-Worker Pattern
Evaluator-Optimizer Workflow
Pre-built Agents
Next Steps
Exploring LangChain Academy
Leveraging LangSmith for Development

Introduction to LangGraph
Overview
LangGraph is a robust library developed by LangChain Inc. that facilitates the creation of stateful, multi-actor applications using LLMs. Inspired by frameworks like Pregel and Apache Beam, LangGraph offers a graph-based approach to building agents and multi-agent workflows, drawing design inspiration from NetworkX. It seamlessly integrates with LangChain and LangSmith but can also be used independently.
Why Use LangGraph?
LangGraph stands out due to its:
Fine-Grained Control: Manage both the flow and state of your agent applications with precision.
Central Persistence Layer: Enables memory across interactions and human-in-the-loop workflows.
Tooling Support: Through LangGraph Platform, it offers development, deployment, debugging, and monitoring tools.
Integration Flexibility: Works with or without LangChain, allowing you to choose your preferred tools.
LangGraph Platform
The LangGraph Platform is a commercial infrastructure built on the open-source LangGraph framework. It includes:
LangGraph Server: Provides APIs for interacting with your LangGraph applications.
LangGraph SDKs: Client libraries for various programming languages.
LangGraph CLI: Command-line tools for building and managing the server.
LangGraph Studio: A UI and debugger for monitoring and debugging your applications.
The platform addresses common deployment challenges such as streaming support, background runs, long-running agents, double texting, and handling burstiness.
Figure: LangGraph Platform Components

Installation
Before diving into building applications with LangGraph, ensure you have the necessary packages installed. You can install LangGraph and its dependencies using pip:
bash
Copy
pip install -U langgraph

For integrating with LangChain and LangSmith (optional but recommended for enhanced functionality):
bash
Copy
pip install -U langgraph langsmith langchain_anthropic


LangGraph Quickstart
In this tutorial, we'll build a support chatbot using LangGraph that can:
Answer common questions by searching the web.
Maintain conversation state across calls.
Route complex queries to a human for review.
Use custom state to control its behavior.
Rewind and explore alternative conversation paths.
We'll start with a basic chatbot and progressively add more sophisticated capabilities, introducing key LangGraph concepts along the way.

Part 1: Building a Basic Chatbot
In this section, we'll create a simple chatbot using LangGraph. This chatbot will respond directly to user messages, illustrating the core concepts of building with LangGraph.
Step 1: Define the State
The StateGraph defines the structure of our chatbot as a state machine. We'll start by defining the state schema using Python's TypedDict and Annotated for state updates.
python
Copy
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages are stored as a list. The `add_messages` function appends new messages.
    messages: Annotated[list, add_messages]

Explanation:
TypedDict: Defines the schema of the state, ensuring type safety.
messages: A list that holds the conversation history. The add_messages reducer appends new messages instead of overwriting them.
Step 2: Add the Chatbot Node
Nodes represent units of work in the graph. Here, we'll add a chatbot node that uses an LLM to generate responses.
python
Copy
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

def chatbot(state: State):
    # Invoke the LLM with the current messages
    return {"messages": [llm.invoke(state["messages"])]}

# Initialize the graph builder with the state schema
graph_builder = StateGraph(State)

# Add the chatbot node to the graph
graph_builder.add_node("chatbot", chatbot)

Explanation:
ChatAnthropic: An LLM from LangChain's Anthropic integration.
chatbot function: Takes the current state and returns an updated messages list by invoking the LLM.
Step 3: Define Entry and Exit Points
Specify where the graph starts and ends its execution.
python
Copy
# Define the starting edge
graph_builder.add_edge(START, "chatbot")

# Define the finishing edge
graph_builder.add_edge("chatbot", END)

# Compile the graph into an executable form
graph = graph_builder.compile()

Step 4: Visualize the Graph
Visualizing the graph helps in understanding its structure.
python
Copy
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

Note: Visualization requires additional dependencies. If not installed, you can skip this step.
Step 5: Run the Chatbot
Create a simple loop to interact with the chatbot.
python
Copy
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config={"configurable": {"thread_id": 42}}, stream_mode="values"):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # Fallback if input() is not available (e.g., in certain environments)
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

Sample Interaction:
vbnet
Copy
User: What is LangGraph?
Assistant: LangGraph is a library designed to help build stateful multi-agent applications using language models. It provides tools for creating workflows and state machines to coordinate multiple AI agents or language model interactions. LangGraph is built on top of LangChain, leveraging its components while adding graph-based coordination capabilities. It's particularly useful for developing more complex, stateful AI applications that go beyond simple query-response interactions.
Goodbye!

Full Code Example
Below is the complete code for building and running the basic chatbot:
python
Copy
# Import necessary libraries
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from IPython.display import Image, display

# Define the state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Define the chatbot node function
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# Initialize the graph builder
graph_builder = StateGraph(State)

# Add the chatbot node
graph_builder.add_node("chatbot", chatbot)

# Define entry and exit points
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
graph = graph_builder.compile()

# (Optional) Visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Function to stream updates from the graph
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config={"configurable": {"thread_id": 42}}, stream_mode="values"):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Chat loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # Fallback interaction
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break


Part 2: Enhancing the Chatbot with Tools
Now that we have a basic chatbot, let's enhance its capabilities by integrating a web search tool. This will allow the chatbot to fetch real-time information beyond its training data.
Step 1: Install Additional Packages and Set API Keys
We'll use the Tavily Search Engine for fetching search results.
bash
Copy
pip install -U tavily-python langchain_community

Set the required environment variables:
python
Copy
import os
import getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("TAVILY_API_KEY")

Step 2: Define the Search Tool
Use Tavily to create a search tool that the chatbot can invoke.
python
Copy
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize the search tool with a maximum of 2 results
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Test the tool
search_results = tool.invoke("What's a 'node' in LangGraph?")
print(search_results)

Sample Output:
python
Copy
[
    {
        'url': 'https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141',
        'content': 'Nodes: Nodes are the building blocks of your LangGraph. Each node represents a function or a computation step. You define nodes to perform specific tasks, such as processing input, making ...'
    },
    {
        'url': 'https://saksheepatil05.medium.com/demystifying-langgraph-a-beginner-friendly-dive-into-langgraph-concepts-5ffe890ddac0',
        'content': 'Nodes (Tasks): Nodes are like the workstations on the assembly line. Each node performs a specific task on the product. In LangGraph, nodes are Python functions that take the current state, do some work, and return an updated state. Next, we define the nodes, each representing a task in our sandwich-making process.'
    }
]

Step 3: Bind Tools to the LLM
Modify the chatbot to be aware of the search tool.
python
Copy
from langchain_anthropic import ChatAnthropic

# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)

Step 4: Update the Chatbot Node
Modify the chatbot node to use the augmented LLM with tools.
python
Copy
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

Step 5: Implement Tool Execution
Add a tool node that executes the requested tools when invoked by the chatbot.
python
Copy
from langchain_core.messages import ToolMessage
import json

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# Initialize the tool node
tool_node = BasicToolNode(tools=tools)

# Add the tool node to the graph
graph_builder.add_node("tools", tool_node)

Step 6: Define Conditional Edges
Route the control flow based on whether the chatbot has invoked any tools.
python
Copy
from langgraph.prebuilt import tools_condition

def route_tools(state: State):
    """
    Route to 'tools' node if there are tool calls, else to END.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# Add conditional edges
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

# Define the edge from tools back to chatbot
graph_builder.add_edge("tools", "chatbot")

Step 7: Compile and Visualize the Enhanced Graph
python
Copy
graph = graph_builder.compile()

# (Optional) Visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

Step 8: Run the Enhanced Chatbot
Interact with the chatbot, which can now utilize the search tool.
python
Copy
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # Fallback interaction
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

Sample Interaction:
vbnet
Copy
User: What is the weather in sf?
Assistant: Based on the search results, I can tell you that the current weather in San Francisco is:
Temperature: 60 degrees Fahrenheit
Conditions: Foggy
San Francisco is known for its microclimates and frequent fog, especially during the summer months. The temperature of 60°F (about 15.5°C) is quite typical for the city, which tends to have mild temperatures year-round. The fog, often referred to as "Karl the Fog" by locals, is a characteristic feature of San Francisco's weather, particularly in the mornings and evenings.
Is there anything else you'd like to know about the weather in San Francisco or any other location?

Full Code Example
Below is the complete code for enhancing the chatbot with a search tool:
python
Copy
# Import necessary libraries
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import tools_condition
from langchain_core.messages import ToolMessage
import json
from IPython.display import Image, display

# Define the state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

# Define the search tool
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Define the tool node
class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# Initialize the tool node
tool_node = BasicToolNode(tools=tools)

# Initialize the graph builder
graph_builder = StateGraph(State)

# Add the chatbot and tool nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Define the routing function
def route_tools(state: State):
    """
    Route to 'tools' node if there are tool calls, else to END.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# Add conditional edges based on tool calls
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

# Define the edge from tools back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
graph = graph_builder.compile()

# (Optional) Visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Function to stream updates from the graph
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}, config={"configurable": {"thread_id": 42}}, stream_mode="values"):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Chat loop
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # Fallback interaction
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break


Part 3: Adding Memory to the Chatbot
While the enhanced chatbot can utilize tools to fetch information, it lacks the ability to remember past interactions, limiting its ability to engage in coherent, multi-turn conversations. LangGraph addresses this through persistent checkpointing.
Step 1: Implement Checkpointing with MemorySaver
LangGraph's MemorySaver enables state persistence across graph invocations.
python
Copy
from langgraph.checkpoint.memory import MemorySaver

# Initialize MemorySaver for state persistence
memory = MemorySaver()

Step 2: Update the Graph with Checkpointer
When compiling the graph, provide the checkpointer to enable memory.
python
Copy
graph = graph_builder.compile(checkpointer=memory)

Step 3: Run the Chatbot with Memory
Use a thread_id to maintain conversation context.
python
Copy
config = {"configurable": {"thread_id": "1"}}

# Initial user input
user_input = "Hi there! My name is Will."

# Invoke the graph with the initial input and config
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()

# Follow-up user input
user_input = "Remember my name?"

# Invoke the graph again with the same thread_id to retain context
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()

Sample Interaction:
vbnet
Copy
User: Hi there! My name is Will.
Assistant: Hello Will! It's nice to meet you. How can I assist you today? Is there anything specific you'd like to know or discuss?

User: Remember my name?
Assistant: Of course, I remember your name, Will. I always try to pay attention to important details that users share with me. Is there anything else you'd like to talk about or any questions you have? I'm here to help with a wide range of topics or tasks.

Step 4: Inspecting the State
You can inspect the current state or retrieve snapshots for debugging.
python
Copy
# Get the current state snapshot
snapshot = graph.get_state(config)
print(snapshot)

# Check the next nodes to execute (should be empty if the conversation ended)
print(snapshot.next)

Sample Output:
python
Copy
StateSnapshot(
    values={
        'messages': [
            HumanMessage(content='Hi there! My name is Will.', ...),
            AIMessage(content="Hello Will! It's nice to meet you...", ...),
            HumanMessage(content='Remember my name?', ...),
            AIMessage(content="Of course, I remember your name, Will...", ...)
        ]
    },
    next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '...'}},
    metadata={...},
    created_at='2024-09-27T19:30:10.820758+00:00',
    parent_config={...},
    tasks=()
)

()

Full Code Example
Below is the complete code for adding memory to the chatbot:
python
Copy
# Import necessary libraries
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
import json
from IPython.display import Image, display

# Define the state schema with messages
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the search tool
tool = TavilySearchResults(max_results=2)
tools = [tool]

# Initialize the LLM and bind tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node function
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Ensure only one tool call per message to avoid duplication
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

# Define the prebuilt ToolNode
tool_node = ToolNode(tools=tools)

# Initialize the graph builder
graph_builder = StateGraph(State)

# Add the chatbot and tool nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Use prebuilt tools_condition for routing
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define the edge from tools back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Define the starting edge
graph_builder.add_edge(START, "chatbot")

# Initialize the checkpointer
memory = MemorySaver()

# Compile the graph with the checkpointer
graph = graph_builder.compile(checkpointer=memory)

# (Optional) Visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Function to stream updates from the graph
def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
        stream_mode="values",
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Initial interaction
user_input = "Hi there! My name is Will."
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    {"configurable": {"thread_id": "1"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

# Follow-up interaction
user_input = "Remember my name?"
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    {"configurable": {"thread_id": "1"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

# Inspect the state
snapshot = graph.get_state({"configurable": {"thread_id": "1"}})
print(snapshot)

print(snapshot.next)  # Should be empty since the conversation ended


Part 4: Human-in-the-loop
To handle complex queries or tasks requiring human oversight, LangGraph supports human-in-the-loop workflows. This allows execution to pause and resume based on human input.
Step 1: Define the Human Assistance Tool
We'll add a tool that requests human input when needed.
python
Copy
from langchain_core.tools import tool
from langgraph.types import Command, interrupt

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

Step 2: Integrate Human Assistance into Tools
Add the new tool to the tools list and bind it to the LLM.
python
Copy
# Add the human_assistance tool
tools = [tool, human_assistance]

# Bind the updated tools to the LLM
llm_with_tools = llm.bind_tools(tools)

Step 3: Update the Chatbot Node
Modify the chatbot to handle tool calls appropriately.
python
Copy
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Ensure only one tool call per message
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

Step 4: Use Prebuilt ToolNode and Condition
Leverage LangGraph's prebuilt components for tool execution and routing.
python
Copy
from langgraph.prebuilt import ToolNode, tools_condition

# Define the prebuilt ToolNode with the updated tools
tool_node = ToolNode(tools=tools)

# Update the graph builder with the new tool node
graph_builder.add_node("tools", tool_node)

# Use prebuilt tools_condition for routing
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define the edge from tools back to chatbot
graph_builder.add_edge("tools", "chatbot")

Step 5: Compile the Graph with Checkpointer
Ensure memory is maintained across interactions.
python
Copy
# Compile the graph with the memory checkpointer
graph = graph_builder.compile(checkpointer=memory)

Step 6: Run the Chatbot with Human-in-the-loop
Interact with the chatbot, triggering human assistance when needed.
python
Copy
user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

# Invoke the graph with user input
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# Simulate human response
human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

# Create a Command to resume execution with human input
human_command = Command(resume={"data": human_response})

# Resume the graph with human input
events = graph.stream(human_command, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

Sample Interaction:
vbnet
Copy
User: I need some expert guidance for building an AI agent. Could you request assistance for me?
Assistant: [{'text': "Certainly! I'd be happy to request expert assistance for you regarding building an AI agent. To do this, I'll use the human_assistance function to relay your request. Let me do that for you now.", 'type': 'text'}, {'id': 'toolu_...', 'input': {'query': 'A user is requesting expert guidance for building an AI agent. Could you please provide some expert advice or resources on this topic?'}, 'name': 'human_assistance', 'type': 'tool_use'}]
Tool Calls:
  human_assistance (toolu_...)
 Call ID: toolu_...
  Args:
    query: A user is requesting expert guidance for building an AI agent. Could you please provide some expert advice or resources on this topic?

Tool Message:
Name: human_assistance

We, the experts are here to help! We'd recommend you check out LangGraph to build your agent. It's much more reliable and extensible than simple autonomous agents.

Assistant: Thank you for your patience. I've received some expert advice regarding your request for guidance on building an AI agent. Here's what the experts have suggested:
...

Full Code Example
Below is the complete code for adding human-in-the-loop functionality to the chatbot:
python
Copy
# Import necessary libraries
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
import json
from IPython.display import Image, display

# Define the state schema with messages
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Define the human_assistance tool
@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

# Initialize the search tool
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool, human_assistance]

# Initialize the LLM and bind tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node function
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Ensure only one tool call per message to avoid duplication
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

# Define the prebuilt ToolNode
tool_node = ToolNode(tools=tools)

# Initialize the graph builder
graph_builder = StateGraph(State)

# Add the chatbot and tool nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Use prebuilt tools_condition for routing
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define the edge from tools back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Define the starting edge
graph_builder.add_edge(START, "chatbot")

# Initialize the checkpointer
memory = MemorySaver()

# Compile the graph with the checkpointer
graph = graph_builder.compile(checkpointer=memory)

# (Optional) Visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Function to stream updates from the graph
def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
        stream_mode="values",
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Initial interaction
user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# Simulate human response
human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

# Create a Command to resume execution with human input
human_command = Command(resume={"data": human_response})

# Resume the graph with human input
events = graph.stream(human_command, config, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


Part 5: Customizing State
To implement more complex behaviors, you can extend the state schema with additional fields. This allows different nodes to access and modify shared data seamlessly.
Step 1: Extend the State Schema
Add custom fields such as name and birthday to the state.
python
Copy
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

Step 2: Update Human Assistance Tool
Modify the human_assistance tool to interact with the new state fields.
python
Copy
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

@tool
def human_assistance(
    name: str, 
    birthday: str, 
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # Process human response
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # Update the state with the verified information
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)

Step 3: Update the Graph
Ensure the graph accommodates the updated state and tool.
python
Copy
# Re-initialize the graph builder with the updated state schema
graph_builder = StateGraph(State)

# Add the chatbot and updated tool nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Use prebuilt tools_condition for routing
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define the edge from tools back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Define the starting edge
graph_builder.add_edge(START, "chatbot")

# Compile the graph with the checkpointer
graph = graph_builder.compile(checkpointer=memory)

Step 4: Run the Chatbot with Custom State
Interact with the chatbot to see how it manages and updates custom state fields.
python
Copy
config = {"configurable": {"thread_id": "1"}}

# Initial user input with name and birthday
user_input = "Hi there! My name is Will."

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()

# Follow-up user input to check memory
user_input = "Remember my name?"

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()

# Inspect the updated state
snapshot = graph.get_state(config)
print({k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})

Sample Output:
python
Copy
{'name': 'LangGraph', 'birthday': 'Jan 17, 2024'}

Step 5: Manually Updating State
LangGraph allows manual state updates at any point.
python
Copy
# Manually update the 'name' field
graph.update_state(config, {"name": "LangGraph (library)"})

# Verify the update
snapshot = graph.get_state(config)
print({k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})

Sample Output:
python
Copy
{'name': 'LangGraph (library)', 'birthday': 'Jan 17, 2024'}

Full Code Example
Below is the complete code for customizing the state:
python
Copy
# Import necessary libraries
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
import json
from IPython.display import Image, display

# Define the extended state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

# Define the human_assistance tool with state updates
@tool
def human_assistance(
    name: str, 
    birthday: str, 
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    # Process human response
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    # Update the state with the verified information
    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)

# Initialize the search tool and human_assistance tool
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool, human_assistance]

# Initialize the LLM and bind tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node function
def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Ensure only one tool call per message
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

# Define the prebuilt ToolNode
tool_node = ToolNode(tools=tools)

# Initialize the graph builder
graph_builder = StateGraph(State)

# Add the chatbot and tool nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Use prebuilt tools_condition for routing
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define the edge from tools back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Define the starting edge
graph_builder.add_edge(START, "chatbot")

# Initialize the checkpointer
memory = MemorySaver()

# Compile the graph with the checkpointer
graph = graph_builder.compile(checkpointer=memory)

# (Optional) Visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Function to stream updates from the graph
def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
        stream_mode="values",
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Initial interaction with name and birthday
user_input = "Hi there! My name is Will."
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    {"configurable": {"thread_id": "1"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

# Follow-up interaction to check memory
user_input = "Remember my name?"
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    {"configurable": {"thread_id": "1"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

# Inspect the updated state
snapshot = graph.get_state({"configurable": {"thread_id": "1"}})
print({k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})

# Simulate human correction
human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
)

# Resume the graph with human correction
events = graph.stream(human_command, {"configurable": {"thread_id": "1"}}, stream_mode="values")

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# Manually update the state
graph.update_state({"configurable": {"thread_id": "1"}}, {"name": "LangGraph (library)"})

# Verify the manual update
snapshot = graph.get_state({"configurable": {"thread_id": "1"}})
print({k: v for k, v in snapshot.values.items() if k in ("name", "birthday")})


Part 6: Time Travel
LangGraph's time travel functionality allows you to rewind the graph to previous states, enabling exploration of alternative conversation paths or error recovery.
Step 1: Generate State History
Every step in the graph execution is checkpointed. You can access the history of states to rewind to a specific point.
python
Copy
# Fetch and print the state history
for state in graph.get_state_history(config):
    print("Num Messages:", len(state.values["messages"]), "Next:", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # Select a specific state to replay
        to_replay = state

Step 2: Rewind and Resume Execution
Use the selected checkpoint_id to rewind the graph and resume execution from that point.
python
Copy
# Check the selected state
print(to_replay.next)
print(to_replay.config)

# Resume execution from the selected checkpoint
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()

Sample Interaction:
sql
Copy
Num Messages:  6 Next:  ('tools',)
--------------------------------------------------------------------------------
...
('tools',)
{'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '...'}}
Assistant: [{'text': "...", 'type': 'text'}, {'id': 'toolu_...', 'input': {...}, 'name': 'tavily_search_results_json', 'type': 'tool_use'}]
Tool Calls:
  tavily_search_results_json (toolu_...)
 Call ID: toolu_...
  Args:
    query: Building autonomous agents with LangGraph examples and tutorials

Tool Message:
Name: tavily_search_results_json

[
    {"url": "...", "content": "..."},
    {"url": "...", "content": "..."}
]

Assistant: Great idea! Building an autonomous agent with LangGraph is indeed an exciting project...

Full Code Example
Below is the complete code for implementing time travel in LangGraph:
python
Copy
# Import necessary libraries
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import ToolMessage
import json
from IPython.display import Image, display

# Define the state schema
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the search tool
search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

# Initialize the LLM and bind tools
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot node function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Define the prebuilt ToolNode
tool_node = ToolNode(tools=tools)

# Initialize the graph builder
graph_builder = StateGraph(State)

# Add the chatbot and tool nodes
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Use prebuilt tools_condition for routing
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Define the edge from tools back to chatbot
graph_builder.add_edge("tools", "chatbot")

# Define the starting edge
graph_builder.add_edge(START, "chatbot")

# Initialize the checkpointer
memory = MemorySaver()

# Compile the graph with the checkpointer
graph = graph_builder.compile(checkpointer=memory)

# (Optional) Visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Function to stream updates from the graph
def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
        stream_mode="values",
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# Initial interactions to generate state history
user_input = "I'm learning LangGraph. Could you do some research on it for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# Additional interaction
user_input = "Ya that's helpful. Maybe I'll build an autonomous agent with it!"
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)

for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

# Fetch and select a state to replay
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages:", len(state.values["messages"]), "Next:", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        to_replay = state

# Check the selected state
print(to_replay.next)
print(to_replay.config)

# Resume execution from the selected checkpoint
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()


Advanced Concepts
LangGraph offers a variety of advanced workflows and agent patterns to handle complex tasks efficiently. Below are some of the key concepts:
Workflows and Agents
Workflows are predefined code paths orchestrating LLMs and tools, suitable for tasks that can be decomposed into fixed subtasks. Agents, on the other hand, are dynamic systems where LLMs direct their own processes and tool usage based on environmental feedback.
Figure: Difference between Agents and Workflows
The Augmented LLM
Augmented LLMs support structured outputs and tool calling, enabling seamless integration with LangGraph's workflows.
python
Copy
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

# Define structured output schema
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query optimized for web search.")
    justification: str = Field(None, description="Why this query is relevant.")

# Initialize and augment the LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("How does Calcium CT score relate to high cholesterol?")
print(output.search_query)
print(output.justification)

Prompt Chaining
Break tasks into a sequence of steps where each LLM call processes the output of the previous one.
python
Copy
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Define schemas for structured output
class Joke(BaseModel):
    joke: str

class State(TypedDict):
    joke: str
    improved_joke: str
    final_joke: str

# Define node functions
def generate_joke(state: State):
    return {"joke": llm.invoke(f"Write a short joke about {state['topic']}").content}

def improve_joke(state: State):
    return {"improved_joke": llm.invoke(f"Make this joke funnier: {state['joke']}").content}

def polish_joke(state: State):
    return {"final_joke": llm.invoke(f"Add a surprising twist: {state['improved_joke']}").content}

# Build the workflow
workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", "improve_joke")
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

# Compile and run
chain = workflow.compile()
state = chain.invoke({"topic": "cats"})
print("Initial joke:", state["joke"])
print("Improved joke:", state["improved_joke"])
print("Final joke:", state["final_joke"])

Parallelization
Run multiple LLM tasks in parallel to speed up processing or gather diverse perspectives.
python
Copy
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Define schemas
class JokeState(TypedDict):
    subject: str

class OverallState(TypedDict):
    topic: str
    subjects: list
    jokes: list
    best_selected_joke: str

# Define node functions
def generate_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

def generate_joke(state: JokeState):
    return {"jokes": [llm.invoke(f"Generate a joke about {state['subject']}").content]}

def select_best_joke(state: OverallState):
    # Logic to select the best joke
    best_joke = max(state["jokes"], key=lambda x: len(x))  # Example criterion
    return {"best_selected_joke": best_joke}

# Build the workflow
parallel_builder = StateGraph(OverallState)
parallel_builder.add_node("generate_jokes", generate_jokes)
parallel_builder.add_node("generate_joke", generate_joke)
parallel_builder.add_node("select_best_joke", select_best_joke)
parallel_builder.add_edge(START, "generate_jokes")
parallel_builder.add_edge("generate_jokes", "generate_joke")
parallel_builder.add_edge("generate_joke", "select_best_joke")
parallel_builder.add_edge("select_best_joke", END)

# Compile and run
parallel_workflow = parallel_builder.compile()
state = parallel_workflow.invoke({"topic": "cats"})
print(state["best_selected_joke"])

Routing
Classify inputs and route them to specialized nodes for handling.
python
Copy
from pydantic import BaseModel, Field
from typing import Literal

# Define schemas
class Route(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(description="Next step in routing.")

class State(TypedDict):
    input: str
    decision: str
    output: str

# Define node functions
def router(state: State):
    decision = router_llm.invoke([
        SystemMessage(content="Route to story, joke, or poem based on user's request."),
        HumanMessage(content=state["input"])
    ])
    return {"decision": decision.step}

def write_story(state: State):
    return {"output": llm.invoke(f"Write a story about {state['input']}").content}

def write_joke(state: State):
    return {"output": llm.invoke(f"Write a joke about {state['input']}").content}

def write_poem(state: State):
    return {"output": llm.invoke(f"Write a poem about {state['input']}").content}

# Define conditional routing
def route_decision(state: State):
    if state["decision"] == "story":
        return "write_story"
    elif state["decision"] == "joke":
        return "write_joke"
    elif state["decision"] == "poem":
        return "write_poem"

# Build the workflow
router_builder = StateGraph(State)
router_builder.add_node("router", router)
router_builder.add_node("write_story", write_story)
router_builder.add_node("write_joke", write_joke)
router_builder.add_node("write_poem", write_poem)
router_builder.add_edge(START, "router")
router_builder.add_conditional_edges(
    "router",
    route_decision,
    {"write_story": "write_story", "write_joke": "write_joke", "write_poem": "write_poem"}
)
router_builder.add_edge("write_story", END)
router_builder.add_edge("write_joke", END)
router_builder.add_edge("write_poem", END)

# Compile and run
router_workflow = router_builder.compile()
state = router_workflow.invoke({"input": "Write me a joke about cats"})
print(state["output"])

Orchestrator-Worker Pattern
Delegate tasks to multiple worker nodes dynamically, enabling scalable and flexible workflows.
python
Copy
from langgraph.types import Send

# Define schemas
class OverallState(TypedDict):
    topic: str
    sections: list
    completed_sections: Annotated[list, operator.add]
    final_report: str

class WorkerState(TypedDict):
    section: str

# Define node functions
def orchestrator(state: OverallState):
    sections = planner_llm.invoke(f"Generate report sections for {state['topic']}").sections
    return {"sections": sections}

def worker(state: WorkerState):
    report_section = llm.invoke(f"Write a report section on {state['section']}").content
    return {"completed_sections": [report_section]}

def synthesizer(state: OverallState):
    final_report = "\n\n".join(state["completed_sections"])
    return {"final_report": final_report}

def assign_workers(state: OverallState):
    return [Send("worker", {"section": s}) for s in state["sections"]]

# Build the workflow
orchestrator_worker_builder = StateGraph(OverallState)
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("worker", worker)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)
orchestrator_worker_builder.add_edge(START, "orchestrator")
orchestrator_worker_builder.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ["worker"]
)
orchestrator_worker_builder.add_edge("worker", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# Compile and run
orchestrator_worker = orchestrator_worker_builder.compile()
state = orchestrator_worker.invoke({"topic": "LLM Scaling Laws"})
print(state["final_report"])

Evaluator-Optimizer Workflow
Iteratively improve responses based on evaluations, akin to the human writing process.
python
Copy
from pydantic import BaseModel, Field
from typing import Literal

# Define schemas
class Feedback(BaseModel):
    grade: Literal["funny", "not funny"] = Field(description="Is the joke funny?")
    feedback: str = Field(description="Feedback for improvement if not funny.")

class State(TypedDict):
    joke: str
    feedback: str
    funny_or_not: str

# Define node functions
def generate_joke(state: State):
    return {"joke": llm.invoke(f"Write a joke about {state['topic']}").content}

def evaluate_joke(state: State):
    evaluation = evaluator_llm.invoke(f"Grade this joke: {state['joke']}")
    return {"funny_or_not": evaluation.grade, "feedback": evaluation.feedback}

def improve_joke(state: State):
    return {"joke": llm.invoke(f"Make this joke funnier based on feedback: {state['feedback']}").content}

def route_joke(state: State):
    if state["funny_or_not"] == "funny":
        return "end"
    else:
        return "improve_joke"

# Build the workflow
optimizer_builder = StateGraph(State)
optimizer_builder.add_node("generate_joke", generate_joke)
optimizer_builder.add_node("evaluate_joke", evaluate_joke)
optimizer_builder.add_node("improve_joke", improve_joke)
optimizer_builder.add_edge(START, "generate_joke")
optimizer_builder.add_edge("generate_joke", "evaluate_joke")
optimizer_builder.add_conditional_edges(
    "evaluate_joke",
    route_joke,
    {"improve_joke": "improve_joke", "end": END},
)
optimizer_builder.add_edge("improve_joke", "evaluate_joke")
optimizer_builder.add_edge("evaluate_joke", END)

# Compile and run
optimizer_workflow = optimizer_builder.compile()
state = optimizer_workflow.invoke({"topic": "Cats"})
print(state["joke"])


Next Steps
Congratulations! You've successfully built a stateful, tool-using chatbot with LangGraph, enhanced it with memory, incorporated human-in-the-loop workflows, customized its state, and explored advanced concepts like time travel and various workflow patterns.
Exploring Further
LangChain Academy: Dive deeper into LangGraph's capabilities with courses and tutorials.
LangSmith Integration: Utilize LangSmith for enhanced observability, debugging, and monitoring of your LangGraph applications.
Official Documentation: Refer to LangGraph Docs for detailed API references and guides.
Community and Support: Join the LangChain community forums or GitHub repositories to seek help, share projects, and collaborate with other developers.
Building More Complex Applications
With the foundational knowledge from this tutorial, you can now tackle more complex projects such as:
Multi-Agent Systems: Coordinate multiple agents to perform distributed tasks.
Automated Workflows: Create intricate workflows that handle various business processes.
Interactive Applications: Develop applications that respond dynamically to user inputs and environmental changes.

Happy coding with LangGraph! 🚀

