from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

client = OpenAI()

# Chain Of Thought: The model is encouraged to break down reasoning step by step before arriving at an answer.

SYSTEM_PROMPT = """
You are a helpful AI assistant that always explains your reasoning step by step before giving the final answer.
This is called the Chain of Thought (CoT) technique.

Instructions:
1. When the user asks a question, first break it down into small steps.
2. Write out your thought process clearly (like solving a puzzle step by step).
3. After reasoning, provide the final answer in a simple and clear way.
4. Always include both "reasoning" and "final_answer" in the JSON response.

Output Format:
{
  "reasoning": "string",
  "final_answer": "string"
}

Example 1:
Input: What is 12 * 3 + 6?
Output: {
  "reasoning": "First multiply 12 * 3 = 36. Then add 6. 36 + 6 = 42.",
  "final_answer": "42"
}

Example 2:
Input: Tom has 10 pencils. He gives 3 to Mary and then buys 5 more. How many pencils does he have?
Output: {
  "reasoning": "Start with 10 pencils. Give away 3, so 10 - 3 = 7. Then add 5 new pencils: 7 + 5 = 12.",
  "final_answer": "12"
}
"""


# response = client.chat.completions.create(
#     model="gpt-4.1-mini",
#     response_format={"type": "json_object"},
#     messages=[
#         { "role": "system", "content": SYSTEM_PROMPT },
#         { "role": "user", "content": "What is 5 / 2 * 3 to the power 4" },
#         { "role": "assistant", "content": json.dumps({ "step": "analyse", "content": "The user is asking to calculate the value of the expression 5 divided by 2, multiplied by 3 raised to the power of 4." })  },
#         { "role": "assistant", "content": json.dumps({"step": "think", "content": "According to the order of operations (PEMDAS/BODMAS), I need to calculate the exponent first: 3 to the power 4. Then I perform the division 5/2. Finally, I multiply the results."})  },
#         { "role": "assistant", "content": json.dumps({"step": "output", "content": "3 to the power 4 equals 81, 5 divided by 2 equals 2.5, and 2.5 multiplied by 81 equals 202.5"})  },
#         { "role": "assistant", "content": json.dumps({"step": "validate", "content": "Double-checking the calculations: 3^4 = 81 is correct, 5/2 = 2.5 is correct, and 2.5 * 81 = 202.5 is also correct."})  },
#         { "role": "assistant", "content": json.dumps({"step": "result", "content": "The value of the expression 5 / 2 * 3^4 is 202.5, computed by first calculating 3^4 = 81, then dividing 5 by 2 to get 2.5, and multiplying 2.5 by 81."})  },
        
#     ]
# )

# print("\n\n🤖:", response.choices[0].message.content, "\n\n")


messages = [
    { "role": "system", "content": SYSTEM_PROMPT }
]

query = input("> ")
messages.append({ "role": "user", "content": query })
while True:
    response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_object"},
        messages=messages
    )

    messages.append({ "role": "assistant", "content": response.choices[0].message.content })
    parsed_response = json.loads(response.choices[0].message.content)

    # Handle Chain of Thought reasoning
    if parsed_response.get("reasoning"):
        print("          🧠 Reasoning:", parsed_response.get("reasoning"))

    if parsed_response.get("final_answer"):
        print("🤖 Final Answer:", parsed_response.get("final_answer"))
        break
