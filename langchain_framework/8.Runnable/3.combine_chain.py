import random
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass

# Custom LLM
class NakliLLM(Runnable):
    def __init__(self):
        print("LLM Created!")

    def invoke(self, prompt):
        response_list = [
            "Delhi is the capital of India.",
            "IPL is a cricket league.",
            "AI stands for Artificial Intelligence."
        ]

        return {"response": random.choice(response_list)}
        

    def predict(self, prompt):
        response_list = [
            "Delhi is the capital of India.",
            "IPL is a cricket league.",
            "AI stands for Artificial Intelligence."
        ]

        return {"response": random.choice(response_list)}
    
# Custom PromptTemplate
class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)

class NakliStrOutputParser(Runnable):
    def __init__(self):
        pass 

    def invoke(self, input_data):
        return input_data["response"]


class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data
    

template1 = NakliPromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

template2 = NakliPromptTemplate(
    template="Explain the following joke {response}",
    input_variables=["response"]
)

llm = NakliLLM()

parser = NakliStrOutputParser()

chain1 = RunnableConnector([template1, llm])

chain2 = RunnableConnector([template2, llm, parser])

# print(chain1.invoke({"topic": "AI"}))
# print(chain2.invoke({"response": "this is joke"}))

final_chain = RunnableConnector([chain1, chain2])
print(final_chain)
print(final_chain.invoke({"topic": "AI"}))