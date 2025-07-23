# Use the API to send a text message to DeepSeek-R1.

import boto3
import json
from botocore.exceptions import ClientError
from credentials import get_bedrock_client
import re

def deepseek(comment):
    # Create a Bedrock Runtime client in the AWS Region of your choice.
    client = get_bedrock_client()

    # Set the cross Region inference profile ID for DeepSeek-R1
    model_id = "arn:aws:bedrock:us-east-1:043309345392:inference-profile/us.deepseek.r1-v1:0"

    # Define the prompt for the model.
    prompt = f"""
    Each comment must be scored on four dimensions as follows:
    ### Comment Evaluation & Scoring
        
    For each comment, assign a **Moderation Score out of 100** according to these parameters:
        
    1. **Bad Word Focus (0–30)**
        - + Reward respectful or helpful language that supports civil discourse.
        - – Penalize for profanity, slurs, violence, racism, terrorism, illegal activities, or spam (e.g., “click here to win”).
           
    2. **Intent Focus (0–30)**
        - + Reward comments that clarify positions or raise questions in good faith.
        - – Penalize for threats, harmful intent, trolling, or manipulation.
        
    3. **Sentiment Analysis (0–20)**
        - – Penalize for toxicity, aggression, rudeness, or insults, even without explicit bad words.
        
    4. **Semantic Analysis (0–20)**
        - – Penalize for hate speech, misinformation (esp. elections, healthcare, science), or conspiracy theories.
        
        
    | Category    | Label | Definition                                                                                                                                            |
    |-------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | sexual      | S     | Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness). |
    | hate        | H     | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste. Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.|
    | violence    | V     | Content that describe, promotes or glorifies violence or celebrates the suffering or humiliation of others.                                                           |
    | harassment  | HR    | Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.                                              |
    | self-harm   | SH    | Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders. Content that contains a graphic description or shows support for self-harm.|
    | sexual/minors | S3 | Sexual content that includes an individual who is under 18 years old. |
    | hate/threatening | H2 | Hateful content that also includes violence or serious harm towards the targeted group. |
    | violence/graphic | V2 | Violent content that describe, depicts death, violence, or serious physical injury in extreme graphic detail. |
    
    ### Output Format:
    ```markdown
    ### Comment
    > “The comment text…”
    - **Bad Word Focus (0–30):** {{score}} → *brief rationale*
    - **Intent Focus (0–30):** {{score}} → *brief rationale*
    - **Sentiment Analysis (0–20):** {{score}} → *brief rationale*
    - **Semantic Analysis (0–20):** {{score}} → *brief rationale*
    - **Total (0–100):** **{{sum}}** -> Label
        
    COMMENT
    {comment}
    Your thinking process **always** give it between </think> tags
    """

    system = f"""You are a professional terminologist with fluent knowledge of greek and english. You have been assigned a task to prepare translations from the following text.
            You have extensive knowledge in eGovernance and are able to supplement the terms with correct translations into english."""

    # Embed the prompt in DeepSeek-R1's instruction format.
    formatted_prompt = f"""
    <｜begin▁of▁sentence｜><|System|>{system}<｜User｜>{prompt}<｜Assistant｜>\n
    """

    body = json.dumps({
        "prompt": formatted_prompt,
        "max_tokens": 5000,
        "temperature": 0.5,
        "top_p": 0.9,
    })

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=body)

        # Read the response body.
        model_response = json.loads(response["body"].read())

        # Extract choices.
        choices = model_response["choices"]
        # print(choices)
        # Get the raw response text
        response_text = choices[0]['text']

        final_response = re.sub(r'</think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        if '</thinking>' in final_response:
            final_response = re.sub(r'.*?</thinking>', '', final_response, flags=re.DOTALL).strip()

        if '</think>' in final_response:
            final_response = final_response.split('</think>', 1)[1].strip()

        # print(final_response)
        return final_response

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

