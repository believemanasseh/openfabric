import logging
from datetime import datetime
from typing import Dict, List, Optional

from ollama import ChatResponse, ResponseError, chat


def format_context(similar_generations: Optional[List[Dict]]) -> str:
    """Format previous image generations into a context string for the LLM.

    Takes a list of previous similar generations and formats them into a readable
    string that includes timestamps and both original and expanded prompts.

    Args:
        similar_generations: Optional[List[Dict]]: List of previous generation records,
            each containing timestamp, original_prompt, and optionally expanded_prompt.

    Returns:
        str: Formatted string containing previous generations context, or empty string
            if no similar generations provided.
    """
    if not similar_generations:
        return ""

    context_lines = ["Previous relevant generations:"]
    for gen in similar_generations:
        timestamp = datetime.fromisoformat(gen["timestamp"]).strftime("%Y-%m-%d %H:%M")
        context_lines.append(f"- {timestamp}: {gen['original_prompt']}")
        if gen.get("expanded_prompt"):
            context_lines.append(f"Expanded as: {gen['expanded_prompt']}")

    return "\n".join(context_lines)


def query_llm(prompt: str, similar_generations: Optional[List[Dict]]) -> str:
    """Query the LLM to expand a user's image generation prompt.

    Uses Ollama to interact with a local LLM, providing context from similar
    previous generations to help create a more detailed and consistent image prompt.

    Args:
        prompt (str): The user's original image generation prompt
        similar_generations (Optional[List[Dict]]): List of previous similar generations
            to use as context

    Returns:
        str: The expanded prompt from the LLM

    Raises:
        Exception: If the LLM query fails for any reason
    """
    try:
        context = format_context(similar_generations)

        system_prompt = f"""You are a creative AI assistant. Expand this prompt to create a detailed image description. Include specific details about lighting, mood, style, and composition.
        
        Context: {context}

        User Prompt: {prompt}
        
        Based on the context above and any similar past generations, I want you to expand the user's prompt in detail that it can be fed to another llm to get a response. Not very detailed but enough to create a stunning image.

        If the user refers to past generations, incorporate relevant elements while adding the requested modifications. DO NOT add additional notes or ask follow-up questions in your response.
        
        Examples of how to expand the prompt: 
        User Prompt: Create a red honda for me
        Expanded Prompt: A red Honda Civic parked in a sunlit driveway, with a clear blue sky and green trees in the background. The car is shiny and well-maintained, reflecting the sunlight. The scene conveys a sense of calm and serenity.

        User Prompt: Create a red honda with a blue background
        Expanded Prompt: A red Honda Civic parked against a vibrant blue background, with the car's glossy surface reflecting the blue hue. The scene is bright and cheerful, evoking a sense of fun and excitement.
        """
        logging.info(f"System prompt: {system_prompt}")

        response: ChatResponse = chat(
            model="mistral:7b",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response["message"]["content"]
    except ResponseError as e:
        raise Exception(f"LLM query failed: {e}")
