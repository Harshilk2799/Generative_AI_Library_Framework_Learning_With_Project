import random

# Custom LLM
class NakliLLM:
    def __init__(self):
        print("LLM Created!")

    def predict(self, prompt):
        response_list = [
            "Delhi is the capital of India.",
            "IPL is a cricket league.",
            "AI stands for Artificial Intelligence."
        ]

        return {"response": random.choice(response_list)}
    

# llm = NakliLLM()
# response = llm.predict("What is the capital of India?")
# print(response)


# Custom PromptTemplate
class NakliPromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)
    
# template = NakliPromptTemplate(
#     template="Write a {length} poem about {topic}",
#     input_variables=["topic", "length"]
# )
# prompt = template.format({"topic": "India", "length": "short"})
# print("Prompt: ", prompt)

# llm = NakliLLM()
# response = llm.predict(prompt)

# print("Response: ",response)




# Custom Chain
class NakliLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm 
        self.prompt = prompt

    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        print("Final Prompt: ", final_prompt)
        result = self.llm.predict(final_prompt)
        print("Result: ", result)
        return result["response"]
    
template = NakliPromptTemplate(
    template="Write a {length} poem about {topic}",
    input_variables=["topic", "length"]
)
llm = NakliLLM()

chain = NakliLLMChain(llm, template)
response = chain.run({"length": "short", "topic": "India"})
print(response)