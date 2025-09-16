import os
from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('gemini_api_key')

def add_citations(response):
    """
    Insert citation links into a generated response text using the grounding metadata.

    Args:
        response (google.genai.types.GenerateContentResponse): 
            The response object returned by the Gemini model, which includes 
            candidates, text, and grounding metadata.

    Returns:
        str: The response text with inline citations appended in Markdown format.
    """
    text = response.text
    supports = response.candidates[0].grounding_metadata.grounding_supports
    chunks = response.candidates[0].grounding_metadata.grounding_chunks    

    # Sort supports by end_index in descending order to avoid shifting issues when inserting.
    sorted_supports = sorted(supports, key=lambda s: s.segment.end_index, reverse=True)

    for support in sorted_supports:
        end_index = support.segment.end_index
        if support.grounding_chunk_indices:
            # Create citation string like [1](link1)[2](link2)
            citation_links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks):
                    uri = chunks[i].web.uri
                    citation_links.append(f"[{i + 1}]({uri})")

            citation_string = ", ".join(citation_links)
            text = text[:end_index] + citation_string + text[end_index:]

    return text

def chat_completion(prompt:str, img_path:str):
    """
    Generate a grounded response from Gemini using a text prompt and an image file.

    This function sets up a Gemini client, uploads an image, and performs a content 
    generation request with Google Search grounding enabled. It returns the raw 
    response object from the API, which can later be processed with `add_citations()`.

    Args:
        prompt (str): 
            The user query or instruction to send to the model.
        img_path (str): 
            Path to the image file to include in the request.

    Returns:
        google.genai.types.GenerateContentResponse: 
            The raw response object containing generated text, candidates, 
            and grounding metadata.
    """
    # Configure the client
    client = genai.Client()

    # Define the grounding tool
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    # Configure generation settings
    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    my_file = client.files.upload(file=f"{img_path}")

    # Make the request
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[my_file, prompt],
        config=config,
    )

    return response
