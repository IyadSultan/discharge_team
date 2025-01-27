from typing import Annotated, TypedDict, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Define the state schema based on the workflow
class AgentStore(TypedDict):
    admission_reason: str
    events: List[str]
    laboratory: List[str]
    microbiology: List[str]
    vad: List[str]
    surgery: List[str]
    chemotherapy: List[str]
    medications: List[str]
    discharge_instructions: List[str]

class State(TypedDict):
    messages: Annotated[list, add_messages]
    agent_store: AgentStore
    scratchpad: AgentStore
    current_document: str
    documents: List[str]
    is_consistent: bool

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True
)

def read_file_content(filepath: str) -> str:
    """Read the content of a file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return ""

# Node functions
def retrieve_documents(state: State):
    """Retrieve relevant documents for the patient."""
    documents = [
        "sample_notes/admission_note.txt",
        "sample_notes/progress_notes.txt"
    ]
    return {"documents": documents}

def create_agent_store(state: State):
    """Initialize the AgentStore with the first document."""
    if state["documents"]:
        return {
            "current_document": state["documents"][0],
            "agent_store": {
                "admission_reason": "",
                "events": [],
                "laboratory": [],
                "microbiology": [],
                "vad": [],
                "surgery": [],
                "chemotherapy": [],
                "medications": [],
                "discharge_instructions": []
            }
        }
    return {}

def read_document(state: State):
    """Read the current document and create scratchpad."""
    document_content = read_file_content(state["current_document"])
    
    prompt = """
    Extract relevant information from the following medical note and categorize it into these sections.
    Return the information in a strict JSON format with these exact keys:
    {
        "admission_reason": "string",
        "events": ["list of strings"],
        "laboratory": ["list of strings"],
        "microbiology": ["list of strings"],
        "vad": ["list of strings"],
        "surgery": ["list of strings"],
        "chemotherapy": ["list of strings"],
        "medications": ["list of strings"],
        "discharge_instructions": ["list of strings"]
    }

    Medical Note:
    """ + document_content + """

    Ensure the response is in valid JSON format that can be parsed.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    try:
        # Try to parse the JSON response
        parsed_data = json.loads(response.content)
        
        # Update scratchpad with parsed data
        return {
            "scratchpad": {
                "admission_reason": parsed_data.get("admission_reason", ""),
                "events": parsed_data.get("events", []),
                "laboratory": parsed_data.get("laboratory", []),
                "microbiology": parsed_data.get("microbiology", []),
                "vad": parsed_data.get("vad", []),
                "surgery": parsed_data.get("surgery", []),
                "chemotherapy": parsed_data.get("chemotherapy", []),
                "medications": parsed_data.get("medications", []),
                "discharge_instructions": parsed_data.get("discharge_instructions", [])
            }
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {str(e)}")
        print("Response was:", response.content)
        return {
            "scratchpad": state["agent_store"].copy()
        }

def check_consistency(state: State) -> dict:
    """Check consistency between scratchpad and documents."""
    prompt = """
    Compare the following information and check for consistency.
    Original AgentStore:
    """ + json.dumps(state['agent_store'], indent=2) + """
    
    Scratchpad:
    """ + json.dumps(state['scratchpad'], indent=2) + """
    
    Are they consistent? Respond with just 'yes' or 'no'.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    is_consistent = response.content.strip().lower() == "yes"
    return {"is_consistent": is_consistent}

def update_agent_store(state: State):
    """Update the agent store with scratchpad data if consistent."""
    if state["is_consistent"]:
        return {"agent_store": state["scratchpad"]}
    return {}

def write_discharge_note(state: State):
    """Generate the final discharge note."""
    prompt = """
    Based on the following structured information, write a comprehensive discharge note:
    """ + json.dumps(state['agent_store'], indent=2) + """
    
    The discharge note should include:
    1. Patient Demographics and Admission Details
    2. Hospital Course
    3. Significant Lab/Test Results
    4. Medications on Discharge
    5. Follow-up Instructions
    6. Warning Signs to Watch For
    
    Format it professionally and include all relevant sections.
    Make sure to include specific details from the provided information.
    If any section lacks information, note that explicitly.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}

# Build the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("retrieve_documents", retrieve_documents)
graph.add_node("create_agent_store", create_agent_store)
graph.add_node("read_document", read_document)
graph.add_node("check_consistency", check_consistency)
graph.add_node("update_agent_store", update_agent_store)
graph.add_node("write_discharge_note", write_discharge_note)

# Add edges
graph.add_edge(START, "retrieve_documents")
graph.add_edge("retrieve_documents", "create_agent_store")
graph.add_edge("create_agent_store", "read_document")
graph.add_edge("read_document", "check_consistency")

# Add conditional edges based on consistency check
def route_consistency(state: State):
    if not state["is_consistent"]:
        # If not consistent, get next document or end
        current_idx = state["documents"].index(state["current_document"])
        if current_idx + 1 < len(state["documents"]):
            return "read_document"
        return END
    return "update_agent_store"

graph.add_conditional_edges(
    "check_consistency",
    route_consistency,
    {
        "read_document": "read_document",
        "update_agent_store": "update_agent_store",
        END: END
    }
)

graph.add_edge("update_agent_store", "write_discharge_note")
graph.add_edge("write_discharge_note", END)

# Initialize memory
memory = MemorySaver()

# Compile the graph
workflow = graph.compile(checkpointer=memory)

def main():
    # Initial state
    initial_state = {
        "messages": [],
        "documents": [],
        "agent_store": {},
        "scratchpad": {},
        "current_document": "",
        "is_consistent": False
    }
    
    # Configuration for the checkpointer
    config = {
        "configurable": {
            "thread_id": "discharge_note_1",
            "checkpoint_ns": "discharge_notes",
            "checkpoint_id": "note_1"
        }
    }
    
    # Run the workflow
    try:
        final_state = workflow.invoke(initial_state, config)
        print("Discharge Note Generated:")
        print("-" * 50)
        if final_state.get("messages"):
            print(final_state["messages"][-1].content)
    except Exception as e:
        print(f"Error generating discharge note: {str(e)}")

if __name__ == "__main__":
    main() 