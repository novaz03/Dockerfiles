"""
RunPod Handler: Whisper + vLLM (基于官方 vLLM worker)
支持三种操作模式：
1. transcribe: 仅转录音频
2. summarize: 仅 LLM 总结文本
3. full: 转录 + LLM 提取 action items
"""

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

print("=== Handler (Whisper + vLLM) starting ===", flush=True)
print(f"Python: {sys.version}", flush=True)

import runpod

print("RunPod imported successfully", flush=True)

# Whisper 配置
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# vLLM 配置 (使用官方 worker 的环境变量)
VLLM_MODEL = os.getenv("MODEL_NAME", os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"))

print(f"Config: WHISPER_MODEL={WHISPER_MODEL}, VLLM_MODEL={VLLM_MODEL}", flush=True)

# Lazy-loaded models
_whisper_model = None
_vllm_engine = None
_sampling_params_class = None


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


def get_vllm_engine():
    """Lazy load vLLM engine on first use."""
    global _vllm_engine, _sampling_params_class
    if _vllm_engine is None:
        print("Loading vLLM...", flush=True)
        from vllm import LLM, SamplingParams
        _sampling_params_class = SamplingParams
        _vllm_engine = LLM(
            model=VLLM_MODEL,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=8192,
        )
        print("vLLM model loaded", flush=True)
    return _vllm_engine, _sampling_params_class


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
    """Generate text using vLLM."""
    engine, SamplingParams = get_vllm_engine()

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )

    outputs = engine.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text.strip()


def extract_action_items(transcript_text: str, custom_prompt: str = None) -> dict:
    """Extract action items using vLLM."""
    prompt = (custom_prompt or ACTION_ITEMS_PROMPT) + transcript_text

    # 添加输出格式提示
    prompt += "\n\n---\nOutput ONLY valid JSON:\n"

    response_text = generate_with_vllm(prompt, max_tokens=4096, temperature=0.3)

    # Parse JSON response
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


def handler(job: dict) -> dict:
    """Main RunPod handler."""
    print(f"=== Job received ===", flush=True)

    job_input = job.get("input", {})

    # 确定操作模式
    mode = job_input.get("mode", "auto")  # auto, transcribe, summarize, full

    # 如果有 audio_base64，默认 full 模式；否则默认 summarize 模式
    if mode == "auto":
        if job_input.get("audio_base64"):
            mode = "full"
        elif job_input.get("text") or job_input.get("prompt"):
            mode = "summarize"
        else:
            return {"error": "Missing required field: audio_base64 or text/prompt"}

    print(f"Processing mode: {mode}", flush=True)

    result = {}

    # Mode: transcribe or full - 需要音频
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

            # Full mode: 继续提取 action items
            if mode == "full" and transcript["full_text"]:
                print("Starting action item extraction...", flush=True)
                custom_prompt = job_input.get("prompt")
                action_items = extract_action_items(transcript["full_text"], custom_prompt)
                result["action_items"] = action_items
                print(f"Extraction done: {len(action_items.get('items', []))} items", flush=True)

        finally:
            Path(audio_path).unlink(missing_ok=True)

    # Mode: summarize - 仅 LLM
    elif mode == "summarize":
        text = job_input.get("text") or job_input.get("prompt")
        if not text:
            return {"error": "Missing required field: text or prompt for summarize mode"}

        # 支持自定义 prompt 模板
        prompt_template = job_input.get("prompt_template")
        if prompt_template:
            prompt = prompt_template + text
        else:
            prompt = ACTION_ITEMS_PROMPT + text

        prompt += "\n\n---\nOutput ONLY valid JSON:\n"

        max_tokens = job_input.get("max_tokens", 4096)
        temperature = job_input.get("temperature", 0.3)

        print("Starting LLM generation...", flush=True)
        response_text = generate_with_vllm(prompt, max_tokens, temperature)

        # 尝试解析 JSON
        try:
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            result["action_items"] = json.loads(response_text)
        except json.JSONDecodeError:
            result["raw_response"] = response_text
            result["parse_error"] = True

        print("LLM generation done", flush=True)

    return result


# Start RunPod serverless
print("=== Starting RunPod serverless ===", flush=True)
runpod.serverless.start({"handler": handler})
