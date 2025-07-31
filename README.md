# Comment-Moderation
A basic template for moderating comments. For each comment, we generate a moderation score and assign labels using prompts sent to LLMs hosted on AWS Bedrock.

#### LLMs used
1. Claude 3.5
2. Llama 3.3 70b
3. Deepseek R1

#### Pipeline
1. We fetch data from the [mmathys/openai-moderation-api-evaluation](https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation)
2. Send each comment to the LLM for moderation analysis.
3. For each comment, generate a dictionary containing:
   - `Model Logic`: Explanation of why the model assigned specific scores and labels.
   - `Ai Labels`: Labels assigned by the model.
   - `Human Labels`: Labels annotated in the dataset.
4. Pass the resulting dictionary to the evaluation step.

#### Evaluation
We evaluate the model output by constructing a confusion matrix, defined as follows
 1. `True Positive`: The model correctly identified labels that match the annotated data.
 2. `True Negative`: The model correctly did not flag the comment, matching the annotation.
 3. `False Positive`: The model assigned a label that is not present in the annotated data.
 4. `False Negative`: The model failed to assign a label that should have been applied, according to the annotation.

#### Important Note
Some comments in the dataset are highly toxic or sensitive. As a result, certain LLMs may refuse to analyze them due to safety constraints.
For more comprehensive testing, consider using uncensored or less restrictive models.
