from langchain_experimental.utilities import PythonREPL

python_tool = PythonREPL()

print(python_tool.run("""

for i in range(1, 11):
    print(i)
"""))