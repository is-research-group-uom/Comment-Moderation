import json
from botocore.exceptions import ClientError
from credentials import get_bedrock_client


def claude3_5(comment):
    # Create an Amazon Bedrock Runtime client.
    brt = get_bedrock_client()

    # Set the model ID
    model_id = "arn:aws:bedrock:us-east-1:043309345392:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0"

    # Define the prompt for the model
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
    | hate        | H     | Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.   |
    | violence    | V     | Content that describe, promotes or glorifies violence or celebrates the suffering or humiliation of others.                                                           |
    | harassment  | HR    | Content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.                                              |
    | self-harm   | SH    | Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.                                             |
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
    """

    # Format the request payload (back to simple text)
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 131072,
        "top_k": 250,
        "stop_sequences": [],
        "temperature": 0.7,
        "top_p": 0.999,
        "system": '''
            You are a moderation assistant affiliated with the AI4Deliberation group at the University of Macedonia. Your primary role is to evaluate online comments in forums and 
            debates. You must analyze user comments for **deliberation quality and moderation concerns**. You must be unbiased, transparent, and consistent. Your score should reflect both civility 
            and contribution to meaningful deliberation.
        ''',
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }

    # Convert the native request to JSON
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request
        response = brt.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)

    # Decode the response body
    model_response = json.loads(response["body"].read())

    # Extract and print the response text
    response_text = model_response['content'][0]['text']

    return response_text