"""
RunPod Handler: Whisper + vLLM
Supports:
- transcribe: transcribe audio only
- summarize: summarize text with LLM
- full: transcribe audio and extract action items with LLM
- passthrough: forward chat completions directly to the local vLLM server
"""

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

import requests
import runpod

print("=== Handler (Whisper + vLLM) starting ===", flush=True)
print(f"Python: {sys.version}", flush=True)

# Whisper configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# vLLM configuration
VLLM_MODEL = os.getenv("MODEL_NAME", os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
VLLM_API_URL = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")

print(f"Config: WHISPER_MODEL={WHISPER_MODEL}, VLLM_MODEL={VLLM_MODEL}", flush=True)

# Lazy-loaded whisper model
_whisper_model = None


def get_whisper_model():
    """Lazy load Whisper model on first use."""
    global _whisper_model
    if _whisper_model is None:
        print("Loading faster-whisper...", flush=True)
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        print("Whisper model loaded", flush=True)
    return _whisper_model


# Action items extraction prompt
ACTION_ITEMS_PROMPT = """You are an expert Meeting Analyst. Extract action items from this meeting transcript.

Output JSON format:
{
  "summary": "3-5 sentences summarizing the meeting",
  "items": [
    {
      "task": "Task description",
      "assignee": "Person responsible or TBD",
      "due_date": "Date or TBD",
      "priority": "HIGH / MEDIUM / LOW"
    }
  ]
}

Output in the SAME LANGUAGE as the transcript.

Transcript:
"""


def transcribe_audio(audio_path: str, language: str = None) -> dict:
    """Transcribe audio using faster-whisper."""
    model = get_whisper_model()

    segments_iter, info = model.transcribe(
        audio_path,
        beam_size=5,
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments = []
    full_text_parts = []

    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text,
            })
            full_text_parts.append(text)

    return {
        "segments": segments,
        "full_text": " ".join(full_text_parts),
        "language": info.language,
    }


def generate_with_vllm(prompt: str, max_tokens: int = 4096, temperature: float = 0.3) -> str:
    """Generate text using the local vLLM OpenAI-compatible server."""
    payload = {
        "model": VLLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    response = requests.post(VLLM_API_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()

    if not data.get("choices"):
        raise ValueError("Empty response from vLLM")

    return data["choices"][0]["message"]["content"].strip()


def extract_action_items(transcript_text: str, custom_prompt: str = None) -> dict:
    """Extract action items using vLLM."""
    prompt = (custom_prompt or ACTION_ITEMS_PROMPT) + transcript_text
    prompt += "\n\n---\nOutput ONLY valid JSON:\n"

    response_text = generate_with_vllm(prompt, max_tokens=4096, temperature=0.3)

    try:
        # Handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```json"):
                    in_json = True
                    continue
                if line.startswith("```"):
                    in_json = False
                    continue
                if in_json:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        return json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "summary": "",
            "items": [],
            "raw_response": response_text,
            "parse_error": True,
        }


def forward_chat(job_input: dict) -> dict:
    """Passthrough mode: forward chat completions to the local vLLM server."""
    payload = {
        "model": job_input.get("model", VLLM_MODEL),
        "messages": job_input.get("messages", []),
        "temperature": job_input.get("temperature", 0.7),
        "max_tokens": job_input.get("max_tokens", 2048),
        "stream": False,
    }

    # Merge any other keys while skipping handler-specific fields
    skip_keys = {"mode", "audio_base64", "language", "prompt_template"}
    for key, value in job_input.items():
        if key in skip_keys or key in payload:
            continue
        payload[key] = value

    response = requests.post(VLLM_API_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def handler(job: dict) -> dict:
    """Main RunPod handler."""
    print("=== Job received ===", flush=True)

    job_input = job.get("input", {})

    mode = job_input.get("mode", "auto")

    # Auto-detect operation mode
    if mode == "auto":
        if job_input.get("audio_base64"):
            mode = "full"
        elif job_input.get("text") or job_input.get("prompt") or job_input.get("prompt_template"):
            mode = "summarize"
        elif job_input.get("messages"):
            mode = "passthrough"
        else:
            return {"error": "Missing required field: audio_base64 or text/prompt/messages"}

    print(f"Processing mode: {mode}", flush=True)

    # Passthrough chat completions
    if mode == "passthrough":
        try:
            return forward_chat(job_input)
        except requests.exceptions.RequestException as exc:
            body = exc.response.text if getattr(exc, "response", None) else ""
            return {"error": f"Request to vLLM failed: {exc} | Body: {body}"}

    result = {}

    # Mode: transcribe or full
    if mode in ["transcribe", "full"]:
        audio_base64 = job_input.get("audio_base64")
        if not audio_base64:
            return {"error": "Missing required field: audio_base64 for transcribe/full mode"}

        language = job_input.get("language")

        # Decode audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_bytes = base64.b64decode(audio_base64)
            f.write(audio_bytes)
            audio_path = f.name

        try:
            print("Starting transcription...", flush=True)
            transcript = transcribe_audio(audio_path, language)
            print(f"Transcription done: {len(transcript['segments'])} segments", flush=True)
            result["transcript"] = transcript

            # Full mode: continue with action item extraction
            if mode == "full" and transcript["full_text"]:
                print("Starting action item extraction...", flush=True)
                custom_prompt = job_input.get("prompt")
                try:
                    action_items = extract_action_items(transcript["full_text"], custom_prompt)
                    result["action_items"] = action_items
                    print(f"Extraction done: {len(action_items.get('items', []))} items", flush=True)
                except (requests.exceptions.RequestException, ValueError) as exc:
                    result["action_items_error"] = f"vLLM request failed: {exc}"

        finally:
            Path(audio_path).unlink(missing_ok=True)

    # Mode: summarize - LLM only
    elif mode == "summarize":
        text = job_input.get("text") or job_input.get("prompt")
        if not text:
            return {"error": "Missing required field: text or prompt for summarize mode"}

        prompt_template = job_input.get("prompt_template")
        if prompt_template:
            prompt = prompt_template + text
        else:
            prompt = ACTION_ITEMS_PROMPT + text

        prompt += "\n\n---\nOutput ONLY valid JSON:\n"

        max_tokens = job_input.get("max_tokens", 4096)
        temperature = job_input.get("temperature", 0.3)

        print("Starting LLM generation...", flush=True)
        try:
            response_text = generate_with_vllm(prompt, max_tokens, temperature)
            try:
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                result["action_items"] = json.loads(response_text)
            except json.JSONDecodeError:
                result["raw_response"] = response_text
                result["parse_error"] = True
        except (requests.exceptions.RequestException, ValueError) as exc:
            result["error"] = f"vLLM request failed: {exc}"

        print("LLM generation done", flush=True)

    else:
        return {"error": f"Unsupported mode: {mode}"}

    return result


print("=== Starting RunPod serverless ===", flush=True)
runpod.serverless.start({"handler": handler})
