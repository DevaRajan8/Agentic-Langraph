import streamlit as st
import os
import time
import requests
import logging
from typing import List, Dict, Any, Optional, Union
import json
from pydantic import BaseModel, Field
import traceback
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Import StateGraph from langgraph
from langgraph.graph import StateGraph

# ---------------------
# Logging Configuration
# ---------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

# ---------------------
# API Key Configuration
# ---------------------
GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()

# ---------------------
# Groq API Call Helper
# ---------------------
def call_groq_api(prompt: str, max_retries: int = 3) -> str:
    """
    Call Groq API with retry logic.
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.2
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            logger.info(f"Groq API success: {content[:100]}..." if len(content) > 100 else f"Groq API success: {content}")
            return content
        except requests.exceptions.RequestException as e:
            logger.warning(f"Groq API attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    return "ERROR: API request failed after retries."

# ---------------------
# State Definition
# ---------------------
class WorkflowState(BaseModel):
    query: str = ""
    tasks: List[str] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)
    ok: bool = False
    feedback: str = ""
    task_id: int = 0
    task_description: str = ""
    output: str = ""
    tool_input: str = ""
    tool_output: str = ""
    refine_tasks: Optional[List[str]] = None
    charts: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)

# ---------------------
# Helper Functions
# ---------------------
def extract_json_from_response(response: str) -> Dict:
    """
    Extract JSON object from response text that might contain code blocks or other formatting.
    """
    # Handle the case where the response might be in a code block
    if "```json" in response:
        # Extract the JSON part from within the code block
        parts = response.split("```json", 1)
        if len(parts) > 1:
            json_part = parts[1].split("```", 1)[0].strip()
        else:
            json_part = response
    elif "```" in response:
        # Extract content from any code block
        parts = response.split("```", 1)
        if len(parts) > 1:
            json_part = parts[1].split("```", 1)[0].strip()
        else:
            json_part = response
    else:
        json_part = response.strip()
    
    # Try to parse the extracted part
    try:
        result = json.loads(json_part)
        return result
    except json.JSONDecodeError:
        # If that fails, try to find a JSON object in the text
        for line in response.split('\n'):
            if line.strip().startswith('{') and line.strip().endswith('}'):
                try:
                    result = json.loads(line.strip())
                    return result
                except:
                    pass
                    
        # As a last resort, look for anything between braces
        import re
        json_candidates = re.findall(r'\{[^{}]*\}', response)
        for candidate in json_candidates:
            try:
                result = json.loads(candidate)
                return result
            except:
                pass
                
        # If all else fails
        raise json.JSONDecodeError("Could not extract valid JSON from response", response, 0)

def get_state_dict(state: Union[Dict[str, Any], WorkflowState]) -> Dict[str, Any]:
    """Get dictionary representation of state, handling both dict and Pydantic models."""
    if isinstance(state, WorkflowState):
        return state.model_dump()  # Use model_dump instead of dict for Pydantic v2
    return state

def update_state(original_state: Union[Dict[str, Any], WorkflowState], 
                updates: Dict[str, Any]) -> Union[Dict[str, Any], WorkflowState]:
    """Update state with new values, preserving the original type."""
    if isinstance(original_state, WorkflowState):
        # Create a new WorkflowState with updated values
        updated_dict = original_state.model_dump()
        updated_dict.update(updates)
        return WorkflowState(**updated_dict)
    else:
        # Update the dictionary directly
        original_state.update(updates)
        return original_state

# ---------------------
# Custom Tool Functions with Simulated Data
# ---------------------
def fetch_web_tool(state: Union[Dict[str, Any], WorkflowState]) -> Union[Dict[str, Any], WorkflowState]:
    """
    Enhanced web fetching tool with simulated data for GDP information.
    Now parses the JSON input to pull out the real query string.
    """
    state_dict = get_state_dict(state)
    raw_input = state_dict.get("tool_input", "")
    # Try to parse the JSON-encoded input
    try:
        payload = json.loads(raw_input)
        query = payload.get("query", "")
    except Exception:
        query = raw_input

    # Normalize for matching
    q = query.lower()
    result_data = None

    if "gdp growth" in q and "us" in q:
        result_data = {
            "years": [2020, 2021, 2022, 2023, 2024],
            "gdp_growth": [-3.4, 5.7, 2.1, 2.5, 1.8],
            "source": "Simulated US GDP data"
        }
    elif "gdp growth" in q and "china" in q:
        result_data = {
            "years": [2020, 2021, 2022, 2023, 2024],
            "gdp_growth": [2.2, 8.1, 3.0, 5.2, 4.7],
            "source": "Simulated China GDP data"
        }

    if result_data is not None:
        # store into state.data
        updated = state_dict.copy()
        updated.setdefault("data", {})
        if "us" in q:
            updated["data"]["us_gdp"] = result_data
        else:
            updated["data"]["china_gdp"] = result_data

        updated["tool_output"] = json.dumps(result_data)
        return update_state(state, updated)
    else:
        # fallback to a real web search (or your existing placeholder)
        fallback = f"Web search results for: {query}"
        return update_state(state, {"tool_output": fallback})

def calculator_tool(state: Union[Dict[str, Any], WorkflowState]) -> Union[Dict[str, Any], WorkflowState]:
    """Simulated calculator tool"""
    state_dict = get_state_dict(state)
    
    expression = state_dict.get("tool_input", "")
    try:
        # Safely evaluate the expression
        import ast
        result = ast.literal_eval(expression)
        return update_state(state, {"tool_output": f"Calculated result: {result}"})
    except Exception as e:
        return update_state(state, {"tool_output": f"Could not evaluate the expression: {str(e)}"})

def data_processing_tool(state: Union[Dict[str, Any], WorkflowState]) -> Union[Dict[str, Any], WorkflowState]:
    """Data processing tool for cleaning and transforming data"""
    state_dict = get_state_dict(state)
    tool_input = state_dict.get("tool_input", "")
    data = state_dict.get("data", {})
    
    try:
        input_data = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
        operation = input_data.get("operation", "")
        
        if operation == "clean":
            # Just return the data as is (simulating cleaning)
            result = json.dumps(data)
        elif operation == "compare":
            if "us_gdp" in data and "china_gdp" in data:
                # Create comparison data
                comparison = {
                    "years": data["us_gdp"]["years"],
                    "us_growth": data["us_gdp"]["gdp_growth"],
                    "china_growth": data["china_gdp"]["gdp_growth"],
                    "difference": [round(ch - us, 2) for us, ch in zip(
                        data["us_gdp"]["gdp_growth"], 
                        data["china_gdp"]["gdp_growth"]
                    )]
                }
                result = json.dumps(comparison)
                # Update data with comparison results
                data["comparison"] = comparison
            else:
                result = "Error: Missing required data for comparison"
        else:
            result = f"Unknown operation: {operation}"
        
        return update_state(state, {"tool_output": result, "data": data})
    except Exception as e:
        return update_state(state, {"tool_output": f"Data processing error: {str(e)}"})

def chart_creation_tool(state: Union[Dict[str, Any], WorkflowState]) -> Union[Dict[str, Any], WorkflowState]:
    """Tool to create charts based on provided data"""
    state_dict = get_state_dict(state)
    tool_input = state_dict.get("tool_input", "")
    data = state_dict.get("data", {})
    charts = state_dict.get("charts", [])
    
    try:
        input_data = json.loads(tool_input) if isinstance(tool_input, str) else tool_input
        chart_type = input_data.get("chart_type", "line")
        
        if "comparison" in data:
            comparison = data["comparison"]
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the data
            years = comparison["years"]
            x = range(len(years))
            
            if chart_type == "line":
                ax.plot(x, comparison["us_growth"], marker='o', label='US GDP Growth')
                ax.plot(x, comparison["china_growth"], marker='s', label='China GDP Growth')
                ax.set_title('US vs China GDP Growth Comparison')
                ax.set_ylabel('GDP Growth (%)')
                ax.grid(True, linestyle='--', alpha=0.7)
            elif chart_type == "bar":
                width = 0.35
                ax.bar([i - width/2 for i in x], comparison["us_growth"], width, label='US')
                ax.bar([i + width/2 for i in x], comparison["china_growth"], width, label='China')
                ax.set_title('US vs China GDP Growth Comparison')
                ax.set_ylabel('GDP Growth (%)')
            
            # Add x-axis labels
            ax.set_xticks(x)
            ax.set_xticklabels(years)
            ax.set_xlabel('Year')
            
            # Add legend
            ax.legend()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Convert to base64 string
            chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
            chart_data = f"data:image/png;base64,{chart_b64}"
            
            # Store chart in state
            charts.append(chart_data)
            
            result = f"Chart created successfully: GDP Growth Comparison ({chart_type} chart)"
            plt.close(fig)
        else:
            result = "Error: No comparison data available for chart creation"
        
        return update_state(state, {"tool_output": result, "charts": charts})
    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"Chart creation error: {str(e)}\n{tb_str}")
        return update_state(state, {"tool_output": f"Chart creation error: {str(e)}"})

# ---------------------
# Agent Functions
# ---------------------
def plan_agent(state: Union[Dict[str, Any], WorkflowState]) -> Union[Dict[str, Any], WorkflowState]:
    """Planning agent that breaks down a query into tasks."""
    state_dict = get_state_dict(state)
    
    query = state_dict.get("query", "")
    prompt = f"""
    You are a task planning agent. Given a query, break it down into subtasks.
    
    Query: {query}
    
    Return your response as a valid JSON object with a 'tasks' key containing a list of task descriptions.
    
    Example format:
    {{
        "tasks": [
            "First task description",
            "Second task description",
            "Third task description"
        ]
    }}
    
    Make sure your response is valid JSON that can be parsed with json.loads().
    Do not include any text outside the JSON object.
    """
    
    response = call_groq_api(prompt)
    tasks = []
    
    try:
        # Try to extract JSON from response
        result = extract_json_from_response(response)
        tasks = result.get("tasks", [])
    except Exception as e:
        logger.error(f"JSON parsing error in plan_agent: {str(e)}")
        # Fallback: Try to extract tasks with simple parsing
        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if line and (line.startswith("- ") or line.startswith("* ") or (len(line) > 1 and line[0].isdigit() and line[1] in [".", ")"])):
                tasks.append(line.lstrip("- *1234567890.) "))
    
    # If still no tasks, create a default task
    if not tasks:
        tasks = ["Process the query: " + query]
        
    return update_state(state, {"tasks": tasks})

def tool_agent(state: Union[Dict[str, Any], WorkflowState]) -> Union[Dict[str, Any], WorkflowState]:
    """Enhanced tool agent that executes a single task."""
    state_dict = get_state_dict(state)
    
    task = state_dict.get("task_description", "")
    data = state_dict.get("data", {})
    
    # Provide context about available data
    data_context = ""
    if data:
        data_context = "Available data:\n"
        for key, value in data.items():
            data_context += f"- {key}: {json.dumps(value)[:100]}...\n"
    
    prompt = f"""
    You are a tool-using agent that can use multiple tools to complete tasks.
    
    Task: {task}
    
    {data_context}
    
    Available tools:
    - 'fetch_web': For searching information online
    - 'calculator': For calculating numerical results
    - 'data_processing': For cleaning and comparing data
    - 'chart_creation': For creating charts from data
    
    First, determine which tool is needed to complete this specific task.
    
    Return your response as valid JSON in the following format:
    
    {{
        "tool": "tool_name",  // One of the available tools or null if no tool needed
        "input": {{           // Your input parameters for the tool
            "key1": "value1",
            "key2": "value2"
        }},
        "reasoning": "Your reasoning for choosing this tool and parameters",
        "output": null        // leave null if using a tool, provide direct response if no tool needed
    }}
    
    Make sure your response contains only the JSON object.
    """
    
    response = call_groq_api(prompt)
    output = ""
    
    try:
        # Extract and parse JSON response
        result = extract_json_from_response(response)
        tool = result.get("tool")
        reasoning = result.get("reasoning", "")
        
        logger.info(f"Tool agent selected {tool} for task: {task}")
        logger.info(f"Reasoning: {reasoning}")
        
        # Convert input to JSON string for tool
        tool_input = result.get("input", {})
        tool_input_str = json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
        
        # Update state with tool input
        state = update_state(state, {"tool_input": tool_input_str})
        
        if tool == "fetch_web":
            updated_state = fetch_web_tool(state)
            updated_dict = get_state_dict(updated_state)
            output = updated_dict.get("tool_output", "No result")
            
        elif tool == "calculator":
            updated_state = calculator_tool(state)
            updated_dict = get_state_dict(updated_state)
            output = updated_dict.get("tool_output", "No result")
            
        elif tool == "data_processing":
            updated_state = data_processing_tool(state)
            updated_dict = get_state_dict(updated_state)
            output = updated_dict.get("tool_output", "No result")
            
        elif tool == "chart_creation":
            updated_state = chart_creation_tool(state)
            updated_dict = get_state_dict(updated_state)
            output = updated_dict.get("tool_output", "No result")
            
        else:
            output = result.get("output", "No direct result provided")
            
    except Exception as e:
        logger.error(f"Error in tool_agent: {str(e)}, Response: {response}")
        output = f"Failed to process task: {str(e)}"
    
    return update_state(state, {"output": output})

def reflection_agent(state: Union[Dict[str, Any], WorkflowState]) -> Union[Dict[str, Any], WorkflowState]:
    """Enhanced reflection agent that evaluates task results and suggests refinements."""
    state_dict = get_state_dict(state)
    
    query = state_dict.get("query", "")
    tasks = state_dict.get("tasks", [])
    results = state_dict.get("results", [])
    data = state_dict.get("data", {})
    charts = state_dict.get("charts", [])
    
    # Format tasks and results for LLM prompt
    tasks_results = "\n".join([
        f"Task: {r.get('task')}\nResult: {r.get('result')}\n"
        for r in results
    ])
    
    # Add data context
    data_context = ""
    if data:
        data_context = "Available data:\n"
        for key, value in data.items():
            data_summary = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
            data_context += f"- {key}: {data_summary}\n"
    
    # Add chart context
    chart_context = f"Charts created: {len(charts)}" if charts else "No charts created yet."
    
    prompt = f"""
    You are a reflection agent. Evaluate the results of the executed tasks and determine if they 
    satisfy the original query requirements.
    
    Original Query: {query}
    
    Tasks and Results:
    {tasks_results}
    
    {data_context}
    
    {chart_context}
    
    Return your response strictly in the following JSON format ONLY:
    {{
        "ok": true/false,
        "feedback": "your assessment and reasoning",
        "refine_tasks": ["task1", "task2", ...]
    }}
    
    Set "ok" to true only if ALL requirements from the original query have been met.
    If not ok, provide specific tasks that need to be completed in "refine_tasks".
    
    Make sure your response contains only the JSON with no explanations or text outside the JSON.
    """
    
    response = call_groq_api(prompt)
    
    try:
        # Extract and parse JSON response
        logger.info(f"Reflection raw response: {response}")
        result = extract_json_from_response(response)
        
        ok = result.get("ok", False)
        feedback = result.get("feedback", "")
        
        updates = {
            "ok": ok,
            "feedback": feedback
        }
        
        if "refine_tasks" in result and isinstance(result["refine_tasks"], list) and not ok:
            updates["refine_tasks"] = result["refine_tasks"]
        else:
            # If no refinements provided but not OK, use original tasks
            updates["refine_tasks"] = tasks if not ok else []
            
        return update_state(state, updates)
        
    except Exception as e:
        logger.error(f"Error in reflection_agent: {str(e)}")
        # If parsing fails, set default values
        return update_state(state, {
            "ok": False,
            "feedback": f"Failed to parse reflection results: {str(e)}",
            "refine_tasks": tasks  # Keep original tasks as fallback
        })

# ---------------------
# Initialize Langgraph
# ---------------------
# Initialize StateGraph with the defined state type
graph = StateGraph(WorkflowState)

# Register nodes in graph
graph.add_node("PlanAgent", plan_agent)
graph.add_node("ToolAgent", tool_agent)
graph.add_node("ReflectionAgent", reflection_agent)
graph.add_node("fetch_web", fetch_web_tool)
graph.add_node("calculator", calculator_tool)
graph.add_node("data_processing", data_processing_tool)
graph.add_node("chart_creation", chart_creation_tool)

# Set the entry point
graph.set_entry_point("PlanAgent")

# ---------------------
# Define Graph Edges
# ---------------------
# Define the flow between states
graph.add_edge("PlanAgent", "ToolAgent")
graph.add_edge("ToolAgent", "ReflectionAgent")

# Define conditional transitions
def check_completion(state):
    state_dict = get_state_dict(state)
    if state_dict.get("ok", False):
        return "END"
    else:
        return "PlanAgent"

graph.add_conditional_edges("ReflectionAgent", check_completion)

# Compile the graph
compiled_graph = graph.compile()

# ---------------------
# Workflow Execution
# ---------------------
def run_workflow(query: str) -> Dict[str, Any]:
    """Run the entire workflow from start to finish"""
    initial_state = WorkflowState(query=query)
    
    # Execute the full workflow
    final_state = compiled_graph.invoke(initial_state)
    return get_state_dict(final_state)

# ---------------------
# Utility Functions
# ---------------------
def plan_query(query: str) -> List[str]:
    initial_state = WorkflowState(query=query)
    
    # Execute plan agent step
    state = plan_agent(initial_state)
    state_dict = get_state_dict(state)
    return state_dict.get("tasks", [])

def execute_tasks(tasks: List[str]) -> Dict[str, Any]:
    results = []
    state = WorkflowState()
    
    for idx, task in enumerate(tasks, 1):
        # Update task in state
        state = update_state(state, {"task_id": idx, "task_description": task})
        
        # Execute tool agent step
        state = tool_agent(state)
        state_dict = get_state_dict(state)
        
        # Add result
        results.append({
            "task": task, 
            "result": state_dict.get("output", "No result")
        })
    
    # Return both results and state
    return {
        "results": results,
        "state": state
    }

def reflect(query: str, tasks: List[str], results: List[Dict[str, Any]], 
           data: Dict[str, Any] = None, charts: List[str] = None) -> Dict[str, Any]:
    
    state_updates = {
        "query": query,
        "tasks": tasks,
        "results": results
    }
    
    if data:
        state_updates["data"] = data
    
    if charts:
        state_updates["charts"] = charts
    
    state = WorkflowState(**state_updates)
    
    # Execute reflection agent step
    res_state = reflection_agent(state)
    res_dict = get_state_dict(res_state)
    
    return {
        "ok": res_dict.get("ok", False),
        "refinements": res_dict.get("refine_tasks", []),
        "feedback": res_dict.get("feedback", "")
    }

# ---------------------
# Streamlit App
# ---------------------
def main():
    st.set_page_config(page_title="Agentic Workflow with Langgraph & Groq API", layout="wide")
    
    st.title("Agentic Workflow with Langgraph & Groq API 🚀")
    
    # Show debug info option
    show_debug = st.sidebar.checkbox("Show Debug Info", False)
    
    # Text area for query input
    query = st.text_area("Enter your query:", height=100)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        run_button = st.button("Run Workflow")
    
    if run_button and query:
        try:
            # Create a container for workflow steps
            workflow_container = st.container()
            
            # Save workflow state
            workflow_state = WorkflowState(query=query)
            
            # Planning
            with workflow_container:
                with st.spinner("Planning..."):
                    tasks = plan_query(query)
                    
                if not tasks:
                    st.error("Failed to generate tasks. Please try again.")
                    st.stop()
                    
                st.subheader("Planned Subtasks")
                for i, t in enumerate(tasks, 1):
                    st.write(f"{i}. {t}")
                
                # Initialize data and charts
                data = {}
                charts = []
        
                # Execution loop with reflection
                st.subheader("Workflow Progress")
                progress_complete = False
                
                for iteration in range(1, 4):
                    st.write(f"**Iteration {iteration}**")
                    
                    # Execute tasks
                    with st.spinner(f"Executing tasks..."):
                        execution_result = execute_tasks(tasks)
                        results = execution_result["results"]
                        execution_state = execution_result["state"]
                        
                        # Extract data and charts from execution state
                        execution_state_dict = get_state_dict(execution_state)
                        data = execution_state_dict.get("data", data)
                        iteration_charts = execution_state_dict.get("charts", [])
                        if iteration_charts:
                            charts.extend(iteration_charts)
                    
                    # Display task results
                    st.write("**Task Results:**")
                    for r in results:
                        st.write(f"**{r['task']}** → {r['result']}")
                    
                    # Show any charts generated
                    if iteration_charts:
                        st.write("**Generated Charts:**")
                        for chart in iteration_charts:
                            st.image(chart)
                    
                    # Display data if debug mode is on
                    if show_debug and data:
                        st.sidebar.subheader(f"Debug: Data after iteration {iteration}")
                        st.sidebar.json(data)
                    
                    # Reflection
                    with st.spinner(f"Reflection iteration {iteration}..."):
                        refl = reflect(query, tasks, results, data, charts)
                        
                    if show_debug:
                        st.sidebar.subheader(f"Debug: Reflection {iteration}")
                        st.sidebar.json(refl)
                        
                    if refl["ok"]:
                        st.success(f"✅ Iteration {iteration}: All tasks verified successfully!")
                        st.write("Feedback:", refl.get("feedback", "No feedback provided"))
                        progress_complete = True
                        break
                    else:
                        st.warning(f"⚠️ Iteration {iteration}: refining tasks...")
                        st.write("Feedback:", refl.get("feedback", "No feedback provided"))
                        
                        if "refinements" in refl and refl["refinements"]:
                            tasks = refl["refinements"]
                            st.write("**Refined tasks:**")
                            for i, t in enumerate(tasks, 1):
                                st.write(f"{i}. {t}")
                        else:
                            st.error("No refined tasks provided.")
                            break
                
                if progress_complete:
                    st.success("✅ Workflow completed successfully!")
                    
                    # Display final charts
                    if charts:
                        st.subheader("Generated Charts")
                        for chart in charts:
                            st.image(chart)
                    
                    # Display final text results
                    st.subheader("Task Results")
                    for r in results:
                        st.write(f"**{r['task']}:** {r['result']}")
                    # Display whatever charts and results we have
                    if charts:
                        st.subheader("Generated Charts (partial)")
                        for chart in charts:
                            st.image(chart)
                            
                    st.subheader("Task Results (partial)")
                    for r in results:
                        st.write(f"**{r['task']}:** {r['result']}")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if show_debug:
                st.sidebar.subheader("Error Traceback")
                st.sidebar.code(traceback.format_exc())
            logger.error(f"Workflow error: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()