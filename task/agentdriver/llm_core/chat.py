import json
from typing import List, Dict
from task.agentdriver.llm_core.timeout import timeout

from openai import OpenAI
from task.agentdriver.llm_core.api_keys import OPENAI_ORG, OPENAI_API_KEY, OPENAI_BASE_URL

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    client = OpenAI(api_key=OPENAI_API_KEY)
    # print("completion_with_backoff kwargs:", kwargs)
    # input()
    # return openai.ChatCompletion.create(**kwargs)
    return client.chat.completions.create(**kwargs)

@timeout(15)
def run_one_round_conversation(
        full_messages: List[Dict], 
        system_message: str, 
        user_message: str,
        temperature: float = 0.0,
        model_name: str = "gpt-3.5-turbo-0613" # "gpt-3.5-turbo-16k-0613"
        # model_name: str = "gpt-3.5-turbo" # "gpt-3.5-turbo-16k-0613"
    ):
    """
    Perform one round of conversation using OpenAI API
    """
    message_for_this_round = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ] if system_message else [{"role": "user", "content": user_message}]
    
    full_messages.extend(message_for_this_round)
    
    response = completion_with_backoff(
        model=model_name,
        messages=full_messages,
        temperature=temperature,
    )

    # response_message = response["choices"][0]["message"]
    response_message = response.choices[0].message
    
    # Append assistant's reply to conversation
    full_messages.append(response_message)

    return full_messages, response_message

def run_one_round_conversation_with_functional_call(
        full_messages: List[Dict], 
        system_message: str, 
        user_message: str, 
        functional_calls_info: List[Dict],
        temperature: float = 0.0,
        model_name: str = "gpt-3.5-turbo-0613" # "gpt-3.5-turbo-16k-0613"
    ):
    """
    Perform one round of conversation with functional call using OpenAI API
    """
    message_for_this_round = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ] if system_message else [{"role": "user", "content": user_message}]
    
    full_messages.extend(message_for_this_round)
    
    response = completion_with_backoff(
        model=model_name,
        messages=full_messages,
        temperature=temperature,
        functions=functional_calls_info,
        function_call="auto",
    )

    response_message = response["choices"][0]["message"]
    
    # Append assistant's reply to conversation
    full_messages.append(response_message)
    
    return full_messages, response_message