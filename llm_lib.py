from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import tiktoken

load_dotenv()
client = OpenAI()


class Dimension(BaseModel):
    dimension: str
    description: str


class Dimensions(BaseModel):
    dimensions: list[Dimension]


class CommonDimension(BaseModel):
    dimension: str


class CommonDimensions(BaseModel):
    dimensions: list[CommonDimension]


class ExtractDimension(BaseModel):
    dimension: str
    details: str


class ExtractDimensions(BaseModel):
    dimensions: list[ExtractDimension]


def count_tokens(text, model="gpt-4o"):
    # Get the encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)

    # Encode the text and count tokens
    num_tokens = len(encoding.encode(text))

    return num_tokens


def call_llm_for_zkp(readme):
    prompt = f""" You are given a readme content of a project. Your job is to identify if this project is a 
                zero-knowledge proof project (ZKP) or not. If it is a zero-knowledge proof project then you should return 
                'yes', otherwise 'no' in the response. readme_content: {readme}
                """
    token_count = count_tokens(prompt)
    if (token_count > 128000):
        raw_label = "na"
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
        )
        raw_label = completion.choices[0].message.content
    except Exception as e:
        raw_label = "error"
    return raw_label


def call_llm_for_dimension(readme, dim_count=6):
    prompt = f""" You are given a readme content of a project. Your job is to extract {dim_count} comparison 
                dimensions from this readme content with other project readme content.
                readme_content: {readme}
                """
    token_count = count_tokens(prompt)
    if (token_count > 128000):
        dimensions = "na"
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            response_format=Dimensions,
        )
        dimensions = completion.choices[0].message.content
    except Exception as e:
        dimensions = "error"
    return dimensions


def call_llm_for_common_dimensions(combined_dim, dim_count=6):
    prompt = f"""From this list of dimensions, identify {dim_count} common dimensions which can be used for the 
    comparisons of various projects.
    list_of_dimensions: {combined_dim}
    """
    token_count = count_tokens(prompt)
    if (token_count > 128000):
        # In this case, split the data and call llm multiple times.
        dimensions = "token_limit_exceeded"
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            response_format=CommonDimensions,
        )
        dimensions = completion.choices[0].message.content
    except Exception as e:
        dimensions = "error"
    return dimensions


def call_llm_for_extract_dimensions(readme, dim_len, dimensions):
    prompt = f"""Extract these {dim_len} dimensions and summarise from the below readme content. 
    dimensions: {dimensions} 
    readme_content: {readme}
    """
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            response_format=ExtractDimensions,
        )
        dimensions = completion.choices[0].message.content
    except Exception as e:
        dimensions = "error"
    return dimensions
