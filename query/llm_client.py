import httpx
# We import Groq to interact with the cloud-hosted Groq LLM API.
from groq import Groq
from typing import Optional

# We import settings to know the current active mode, model names, and API keys.
from config.settings import settings

# CONFIGURATION DESCRIPTION:
# To toggle between local inference and the Groq API:
# 1. Open the `.env` file in the project root.
# 2. Update the `LLM_MODE` variable:
#    - Set `LLM_MODE=ollama` to run the model locally on your Ryzen/RTX hardware.
#    - Set `LLM_MODE=groq` to run inference via the ultra-fast Groq Cloud API.
# 3. If using Groq, ensure `GROQ_API_KEY` is populated in your `.env`.
#
# A PROMPT is the template of instructions, background context (retrieved chunks), 
# and the user's question sent to the LLM. It guides the model to answer precisely
# using ONLY the facts provided, preventing hallucinations.
#
# LOCAL vs. CLOUD API Tradeoffs:
# - Local (Ollama): Fully private, 100% free, runs offline. However, response speed 
#   depends on local hardware, and model size is limited by VRAM.
# - Cloud API (Groq): Blazing-fast inference (hundreds of tokens per second), runs 
#   larger models (like Llama 3.3 70B), but requires internet access and exposes data 
#   to an external API (data privacy trade-off).

class LLMClient:
    """
    Unified LLM Client that manages routing requests to either a local Ollama instance 
    or the cloud-based Groq API, with automatic fallback handling.
    """
    def __init__(self, llm_mode: Optional[str] = None):
        # Determine the active mode, defaulting to settings if not provided in constructor.
        self.llm_mode = llm_mode or settings.llm_mode
        
        # Initialize the Groq client if the API key is present.
        self.groq_client = None
        if settings.groq_api_key:
            self.groq_client = Groq(api_key=settings.groq_api_key)

    def generate(self, prompt: str) -> str:
        """
        Generates text completion for the given prompt, routing based on active mode.
        If local Ollama fails (offline), it automatically falls back to Groq if configured.
        
        Parameters:
            prompt (str): The combined instructions, retrieved context, and question.
            
        Returns:
            str: The LLM generated response text.
        """
        # (1) Handle local Ollama generation mode
        if self.llm_mode.lower() == "ollama":
            try:
                return self._generate_ollama(prompt)
            except Exception as e:
                print(f"\n[Warning] Local Ollama failed or is offline: {e}")
                # (2) Fallback logic: check if Groq can be used instead
                if self.groq_client:
                    print("Attempting automatic fallback to Groq Cloud API...")
                    return self._generate_groq(prompt)
                else:
                    print("No Groq API key configured. Fallback unavailable.")
                    raise RuntimeError("Local Ollama failed, and no fallback API key is configured.") from e
        
        # (3) Handle cloud Groq generation mode
        elif self.llm_mode.lower() == "groq":
            if not self.groq_client:
                raise ValueError("Groq mode selected but GROQ_API_KEY is not configured in environment.")
            return self._generate_groq(prompt)
            
        else:
            raise ValueError(f"Invalid LLM mode: '{self.llm_mode}'. Select 'ollama' or 'groq'.")

    def _generate_ollama(self, prompt: str) -> str:
        """
        Invokes local Ollama API via HTTP request.
        """
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {
            "model": settings.local_llm_model,
            "prompt": prompt,
            "stream": False  # Disable streaming for easier response synthesis in backend
        }
        
        print(f"Sending prompt to local Ollama (model: {settings.local_llm_model})...")
        # Send HTTP POST request to local Ollama API
        response = httpx.post(url, json=payload, timeout=60.0)
        
        # Raise exception for bad HTTP status codes
        response.raise_for_status()
        
        # Extract response text from JSON structure
        data = response.json()
        return data.get("response", "").strip()

    def _generate_groq(self, prompt: str) -> str:
        """
        Invokes Groq API using the groq Python package.
        """
        if not self.groq_client:
            raise ValueError("Groq client not initialized. Check GROQ_API_KEY.")
            
        print("Sending prompt to Groq API (model: llama-3.1-8b-instant)...")
        # Request completion from Groq API
        chat_completion = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.1-8b-instant",  # Standard active fast model on Groq
            temperature=0.2,          # Keep temp low for deterministic factual responses
            max_tokens=1024
        )
        
        # Extract and return response text
        return chat_completion.choices[0].message.content.strip()
