# agents/coordinator_agent.py

import uuid
import os
import shutil
import time
from typing import List, Dict, Any, Optional

# Import Message Context Protocol (MCP) definitions and message creation utility
from agents.mcp_messages import MCPMessage, create_mcp_message, \
                                DocumentUploadPayload, QueryRequestPayload, FinalResponsePayload, ErrorPayload

# Import the specialized agents
from agents.ingestion_agent import IngestionAgent
from agents.retrieval_agent import RetrievalAgent
from agents.llm_response_agent import LLMResponseAgent

class CoordinatorAgent:
    """
    The central agent responsible for orchestrating communication between other agents
    in the RAG pipeline. It acts as a message broker, dispatching messages between
    Ingestion, Retrieval, and LLM Response agents, and also handles direct UI communication.
    """
    def __init__(self, embedding_dim: int = 384, vector_store_dir: str = "vector_store_data"):
        """
        Initializes the CoordinatorAgent and all its sub-agents.

        Args:
            embedding_dim (int): The dimension for embeddings, passed to RetrievalAgent.
            vector_store_dir (str): Directory for persisting the FAISS vector store.
        """
        self.agents = {} # Dictionary to hold references to all managed agents

        # Initialize sub-agents, passing the Coordinator's dispatch_message method
        # as their message broker.
        self.ingestion_agent = IngestionAgent(self.dispatch_message)
        # The RetrievalAgent is initialized with the embedding dimension and persistence directory.
        self.retrieval_agent = RetrievalAgent(self.dispatch_message, 
                                              embedding_dim=embedding_dim, 
                                              vector_store_dir=vector_store_dir)
        self.llm_response_agent = LLMResponseAgent(self.dispatch_message)

        # Register agents by their names for easy lookup during message dispatch
        self.agents[self.ingestion_agent.name] = self.ingestion_agent
        self.agents[self.retrieval_agent.name] = self.retrieval_agent
        self.agents[self.llm_response_agent.name] = self.llm_response_agent
        
        # A temporary store for responses intended for the UI, keyed by trace_id.
        # This allows the UI (which polls) to retrieve the specific response it's waiting for.
        self.ui_responses: Dict[str, MCPMessage] = {}

        print(f"[{self.__class__.__name__}] Initialized all agents.")

    def dispatch_message(self, message: MCPMessage):
        """
        Receives an MCPMessage from any agent and dispatches it to the intended receiver.
        Messages for the 'UI' are stored for polling by the Streamlit application.

        Args:
            message (MCPMessage): The message to dispatch.
        """
        receiver_name = message["receiver"]
        
        # If the message is intended for the UI, store it
        if receiver_name == "UI":
            self.ui_responses[message["trace_id"]] = message
            return
        
        # Otherwise, dispatch the message to the appropriate agent
        if receiver_name in self.agents:
            self.agents[receiver_name].process_message(message)
        else:
            print(f"[{self.__class__.__name__}] Error: No agent found for receiver '{receiver_name}'. "
                  f"Message from {message['sender']} of type '{message['type']}' dropped.")

    def handle_document_upload(self, document_paths: List[Dict[str, str]]) -> str:
        """
        Initiates the document ingestion process. Called by the UI when files are uploaded
        or default documents are to be processed.

        Args:
            document_paths (List[Dict[str, str]]): A list of dictionaries, each containing
                                                   'path' and 'type' (e.g., 'pdf', 'docx').

        Returns:
            str: The unique trace_id for this ingestion request, used by the UI to track.
        """
        trace_id = str(uuid.uuid4())
        print(f"[{self.__class__.__name__}] Received document upload request. "
              f"Initiating ingestion (Trace ID: {trace_id}).")
        
        payload: DocumentUploadPayload = {"document_paths": document_paths}
        upload_msg = create_mcp_message(
            sender="UI", # Sender is "UI" as it originates from the Streamlit interface
            receiver="IngestionAgent",
            msg_type="DOCUMENT_UPLOAD",
            payload=payload,
            trace_id=trace_id
        )
        self.dispatch_message(upload_msg)
        return trace_id

    def handle_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Initiates a user query. Called by the UI when a user asks a question.
        Includes optional conversation history for multi-turn understanding.

        Args:
            query (str): The user's current query.
            conversation_history (Optional[List[Dict[str, str]]]): A list of previous
                                                                    {"role": "...", "content": "..."} messages.

        Returns:
            str: The unique trace_id for this query request, used by the UI to track.
        """
        trace_id = str(uuid.uuid4())
        print(f"[{self.__class__.__name__}] Received query request: '{query}'. "
              f"Initiating retrieval (Trace ID: {trace_id}).")
        
        # Prepare the payload, including conversation history if provided
        payload: QueryRequestPayload = {
            "query": query,
            "conversation_history": conversation_history if conversation_history is not None else []
        }
        
        query_msg = create_mcp_message(
            sender="UI", # Sender is "UI" as it originates from the Streamlit interface
            receiver="RetrievalAgent", # First, the query goes to RetrievalAgent
            msg_type="QUERY_REQUEST",
            payload=payload,
            trace_id=trace_id
        )
        self.dispatch_message(query_msg)
        return trace_id
    
    def get_ui_response(self, trace_id: str) -> Optional[MCPMessage]:
        """
        Retrieves a completed response message (e.g., FINAL_RESPONSE, ERROR) for a given trace_id.
        This method is polled by the Streamlit UI to get results back from the agent system.

        Args:
            trace_id (str): The unique identifier of the request (query or ingestion).

        Returns:
            Optional[MCPMessage]: The response message if found, otherwise None.
        """
        return self.ui_responses.pop(trace_id, None)

    def clear_vector_store(self):
        """
        Clears the in-memory vector store and deletes its persistent data from disk.
        This provides a "fresh start" for the knowledge base.
        """
        # Instruct the RetrievalAgent to clear its internal vector store
        self.retrieval_agent.vector_store.clear()
        
        # Remove the persistent directory from disk
        if os.path.exists(self.retrieval_agent.vector_store_dir):
            shutil.rmtree(self.retrieval_agent.vector_store_dir)
            # Recreate the directory to ensure it's ready for new data
            os.makedirs(self.retrieval_agent.vector_store_dir, exist_ok=True)
        print(f"[{self.__class__.__name__}] Vector store and its persistent data cleared.")


if __name__ == '__main__':
    # --- Standalone End-to-End Test for CoordinatorAgent with Persistence and Multi-Turn ---

    print("--- Testing CoordinatorAgent (End-to-End Simulation with Persistence and Multi-Turn) ---")

    # Define a temporary directory for vector store persistence for testing purposes
    TEMP_VECTOR_STORE_DIR = "temp_vector_store_data_e2e"
    # Ensure a clean slate for the test by removing any previous test data
    if os.path.exists(TEMP_VECTOR_STORE_DIR):
        shutil.rmtree(TEMP_VECTOR_STORE_DIR)
    os.makedirs(TEMP_VECTOR_STORE_DIR)

    # Initialize the CoordinatorAgent
    coordinator = CoordinatorAgent(vector_store_dir=TEMP_VECTOR_STORE_DIR)

    # --- Setup dummy test_docs directory for ingestion ---
    TEST_DOCS_DIR = "test_e2e_docs"
    if os.path.exists(TEST_DOCS_DIR):
        shutil.rmtree(TEST_DOCS_DIR)
    os.makedirs(TEST_DOCS_DIR)
    
    # Create sample documents for the E2E test
    with open(os.path.join(TEST_DOCS_DIR, "report.txt"), "w") as f:
        f.write("Annual financial report summary: Revenue up by 15%, profit margin at 10%. Key initiatives for next quarter include expanding into new markets and optimizing existing sales channels.")
    with open(os.path.join(TEST_DOCS_DIR, "roadmap.txt"), "w") as f:
        f.write("Project roadmap: Phase 1 (Discovery) completed. Phase 2 (Development) involves implementing agentic architecture and Model Context Protocol for inter-agent communication. Phase 3 (Deployment) focuses on scaling and user feedback.")
    
    print(f"\n--- Simulating Document Upload via Coordinator (from UI) ---")
    doc_paths_for_upload = [
        {"path": os.path.join(TEST_DOCS_DIR, "report.txt"), "type": "txt"},
        {"path": os.path.join(TEST_DOCS_DIR, "roadmap.txt"), "type": "txt"},
    ]
    upload_trace_id = coordinator.handle_document_upload(doc_paths_for_upload)

    # For synchronous testing, give agents time to process and save
    print(f"[{coordinator.__class__.__name__}] Document upload initiated with Trace ID: {upload_trace_id}")
    time.sleep(2) # Allow time for ingestion and saving to disk

    print(f"\n--- Simulating App Restart (re-initializing Coordinator to test persistence) ---")
    # This new coordinator instance should load the previously saved vector store from disk
    coordinator_reloaded = CoordinatorAgent(vector_store_dir=TEMP_VECTOR_STORE_DIR)
    # Basic check to see if the vector store loaded any chunks
    assert len(coordinator_reloaded.retrieval_agent.vector_store.get_all_chunks()) > 0, "Vector store did not load any chunks on restart!"
    print(f"[{coordinator_reloaded.__class__.__name__}] Reloaded Coordinator. Vector store contains "
          f"{len(coordinator_reloaded.retrieval_agent.vector_store.get_all_chunks())} vectors.")


    print(f"\n--- Simulating Multi-Turn User Query via Reloaded Coordinator ---")
    chat_history = [] # This will simulate st.session_state.chat_history

    # Turn 1
    user_query_1 = "What was the revenue increase mentioned in the report?"
    print(f"\nUser: {user_query_1}")
    query_trace_id_1 = coordinator_reloaded.handle_query(user_query_1, conversation_history=chat_history)

    final_response_1 = None
    attempts = 0
    max_attempts = 10 # Shorter timeout for individual turns in test
    while final_response_1 is None and attempts < max_attempts:
        response_msg = coordinator_reloaded.get_ui_response(query_trace_id_1)
        if response_msg and response_msg["type"] == "FINAL_RESPONSE":
            final_response_1 = response_msg["payload"]
            break
        time.sleep(0.5)
        attempts += 1
    
    if final_response_1:
        print(f"Assistant: {final_response_1['answer']}")
        chat_history.append({"role": "user", "content": user_query_1})
        chat_history.append({"role": "assistant", "content": final_response_1['answer']})
    else:
        print("Assistant: No response for Turn 1.")

    # Turn 2 (Follow-up question using context from previous turn)
    user_query_2 = "And what about profit margin?"
    print(f"\nUser: {user_query_2}")
    query_trace_id_2 = coordinator_reloaded.handle_query(user_query_2, conversation_history=chat_history)

    final_response_2 = None
    attempts = 0
    while final_response_2 is None and attempts < max_attempts:
        response_msg = coordinator_reloaded.get_ui_response(query_trace_id_2)
        if response_msg and response_msg["type"] == "FINAL_RESPONSE":
            final_response_2 = response_msg["payload"]
            break
        time.sleep(0.5)
        attempts += 1

    if final_response_2:
        print(f"Assistant: {final_response_2['answer']}")
        chat_history.append({"role": "user", "content": user_query_2})
        chat_history.append({"role": "assistant", "content": final_response_2['answer']})
    else:
        print("Assistant: No response for Turn 2.")

    # Turn 3 (another follow-up, potentially related to roadmap now)
    user_query_3 = "What's involved in Phase 2?"
    print(f"\nUser: {user_query_3}")
    query_trace_id_3 = coordinator_reloaded.handle_query(user_query_3, conversation_history=chat_history)

    final_response_3 = None
    attempts = 0
    while final_response_3 is None and attempts < max_attempts:
        response_msg = coordinator_reloaded.get_ui_response(query_trace_id_3)
        if response_msg and response_msg["type"] == "FINAL_RESPONSE":
            final_response_3 = response_msg["payload"]
            break
        time.sleep(0.5)
        attempts += 1

    if final_response_3:
        print(f"Assistant: {final_response_3['answer']}")
        chat_history.append({"role": "user", "content": user_query_3})
        chat_history.append({"role": "assistant", "content": final_response_3['answer']})
    else:
        print("Assistant: No response for Turn 3.")


    # Clean up dummy test directory and vector store data
    print(f"\n--- Cleaning up {TEST_DOCS_DIR} and {TEMP_VECTOR_STORE_DIR} ---")
    shutil.rmtree(TEST_DOCS_DIR)
    shutil.rmtree(TEMP_VECTOR_STORE_DIR) # Clean up persistent data
    print("End-to-End CoordinatorAgent test complete.")