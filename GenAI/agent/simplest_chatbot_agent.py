
from huggingface_hub import InferenceClient
import os

client = InferenceClient(
	provider="hf-inference",
	api_key=os.environ["HUGGING_FACE_KEY"]
)

cust_details=input("Enter the customer details: ")

messages = [
    {   "role": "system", 
     "content": '''Act as a Customer Relation Manager Agent and determine the following:
                        1. Churn Probability?
                        2. Customer lifetime value ?
                        3. Customer Segmentation (with Gold, Silver and Bronze)?'''
    },
	{
		"role": "user",
		"content": cust_details
	},
    {
		"role": "assistant",
		"content": '''content": '{"churnProbality": int, "customerLifetimeValue": int, "customerSegmentation": str}'''
	}
]

stream = client.chat.completions.create(
	model="mistralai/Mistral-7B-Instruct-v0.3", 
	messages=messages, 
	temperature=0.5,
	max_tokens=2048,
	top_p=0.7,
	stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")