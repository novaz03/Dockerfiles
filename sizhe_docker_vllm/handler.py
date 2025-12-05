import runpod
import requests
import json
import os
import sys

# The internal vLLM URL (running on localhost inside the container)
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

def handler(job):
    """
    Runpod Handler function.
    Funnels the Runpod 'input' directly to the vLLM OpenAI-compatible API.
    """
    job_input = job.get("input", {})
    
    # 1. Adapt Input
    # Runpod inputs usually come as {"input": {"messages": [...], "model": "..."}}
    # or sometimes just {"messages": ...} depending on how you invoke it.
    # We try to accommodate both.
    
    if "messages" not in job_input and "prompt" not in job_input:
        # Check if the inputs are nested inside an 'input' key (common in some templates)
        if "input" in job_input:
             job_input = job_input["input"]

    # Default payload structure for OpenAI API
    payload = {
        "model": os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B-Instruct"),
        "messages": job_input.get("messages", []),
        "temperature": job_input.get("temperature", 0.7),
        "max_tokens": job_input.get("max_tokens", 2048),
        "stream": False # Serverless requires non-streaming usually, unless using specialized streaming handlers
    }

    # Merge any other keys from input (e.g. top_p, presence_penalty)
    for key, value in job_input.items():
        if key not in payload:
            payload[key] = value

    # 2. Forward to vLLM
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(VLLM_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        # 3. Return Response
        return response.json()

    except requests.exceptions.RequestException as e:
        error_msg = f"Request to vLLM failed: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
             error_msg += f" | Body: {e.response.text}"
        
        return {"error": error_msg}

# Start the Runpod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
