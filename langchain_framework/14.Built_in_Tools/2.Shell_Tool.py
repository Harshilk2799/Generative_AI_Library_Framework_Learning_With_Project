from langchain_community.tools import ShellTool

shell_tool = ShellTool()

# result = shell_tool.invoke("whoami")
result = shell_tool.run({"commands": ["ls", "whoami"]})

print(result)