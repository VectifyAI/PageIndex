import tiktoken
import openai
import logging
import os
import time
import json
import asyncio
from typing import Optional, List, Dict, Any, Union, Tuple
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")

def count_tokens(text: Optional[str], model: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text string using the specified model's encoding.

    Args:
        text (Optional[str]): The text to encode. If None, returns 0.
        model (str): The model name to use for encoding. Defaults to "gpt-4o".

    Returns:
        int: The number of tokens.
    """
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
         # Fallback for newer or unknown models
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def ChatGPT_API_with_finish_reason(
    model: str, 
    prompt: str, 
    api_key: Optional[str] = OPENAI_API_KEY, 
    chat_history: Optional[List[Dict[str, str]]] = None
) -> Tuple[str, str]:
    """
    Call OpenAI Chat Completion API and return content along with finish reason.
    
    Args:
        model (str): The model name (e.g., "gpt-4o").
        prompt (str): The user prompt.
        api_key (Optional[str]): OpenAI API key. Defaults to env var.
        chat_history (Optional[List[Dict[str, str]]]): Previous messages for context.

    Returns:
        Tuple[str, str]: A tuple containing (content, finish_reason).
                         Returns ("Error", "error") if max retries reached.
    """
    max_retries = 10
    if not api_key:
        logging.error("No API key provided.")
        return "Error", "missing_api_key"

    client = openai.OpenAI(api_key=api_key)
    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history.copy() # Avoid modifying original list if passed by ref (shallow copy enough for append)
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            
            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason == "length":
                return content, "max_output_reached"
            else:
                return content, "finished"

        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1) 
            else:
                logging.error('Max retries reached for prompt: ' + prompt[:50] + '...')
                return "Error", "error"
    return "Error", "max_retries"

def ChatGPT_API(
    model: str, 
    prompt: str, 
    api_key: Optional[str] = OPENAI_API_KEY, 
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Call OpenAI Chat Completion API and return the content string.
    
    Args:
        model (str): The model name.
        prompt (str): The user prompt.
        api_key (Optional[str]): OpenAI API key.
        chat_history (Optional[List[Dict[str, str]]]): Previous messages.

    Returns:
        str: The response content, or "Error" if failed.
    """
    max_retries = 10
    if not api_key:
        logging.error("No API key provided.")
        return "Error"

    client = openai.OpenAI(api_key=api_key)
    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history.copy()
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
   
            return response.choices[0].message.content or ""
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt[:50] + '...')
                return "Error"
    return "Error"

async def ChatGPT_API_async(
    model: str, 
    prompt: str, 
    api_key: Optional[str] = OPENAI_API_KEY
) -> str:
    """
    Asynchronously call OpenAI Chat Completion API.

    Args:
        model (str): The model name.
        prompt (str): The user prompt.
        api_key (Optional[str]): OpenAI API key.

    Returns:
        str: The response content, or "Error" if failed.
    """
    max_retries = 10
    if not api_key:
        logging.error("No API key provided.")
        return "Error"

    messages = [{"role": "user", "content": prompt}]
    for i in range(max_retries):
        try:
            async with openai.AsyncOpenAI(api_key=api_key) as client:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt[:50] + '...')
                return "Error"
    return "Error"

def get_json_content(response: str) -> str:
    """
    Extract content inside markdown JSON code blocks.

    Args:
        response (str): The full raw response string.

    Returns:
        str: The extracted JSON string stripped of markers.
    """
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content

def extract_json(content: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Robustly extract and parse JSON from a string, handling common LLM formatting issues.

    Args:
        content (str): The text containing JSON.

    Returns:
        Union[Dict, List]: The parsed JSON object or empty dict/list on failure.
    """
    try:
        # First, try to extract JSON enclosed within ```json and ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # Adjust index to start after the delimiter
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # If no delimiters, assume entire content could be JSON
            json_content = content.strip()

        # Clean up common issues that might cause parsing errors
        json_content = json_content.replace('None', 'null')  # Replace Python None with JSON null
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
        json_content = ' '.join(json_content.split())  # Normalize whitespace

        # Attempt to parse and return the JSON object
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        # Try to clean up the content further if initial parsing fails
        try:
            # Remove any trailing commas before closing brackets/braces
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except:
            logging.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}
