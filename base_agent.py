from typing import List, Dict, Any
from abc import ABC, abstractmethod
import openai
from pydantic import BaseModel

class Tool(BaseModel):
    name: str
    description: str
    function: callable

class BaseAgent(ABC):
    def __init__(self, api_key: str):
        """Initialize the base agent with OpenAI API key."""
        self.api_key = api_key
        openai.api_key = api_key
        self.tools: List[Tool] = []
        self.conversation_history = []

    def add_tool(self, tool: Tool):
        """Add a tool to the agent's toolkit."""
        self.tools.append(tool)

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools and their descriptions."""
        return [{"name": tool.name, "description": tool.description} 
                for tool in self.tools]

    @abstractmethod
    def plan(self, task: str) -> List[str]:
        """Create a plan to solve the given task."""
        pass

    @abstractmethod
    def execute(self, task: str) -> Dict[str, Any]:
        """Execute the task using available tools."""
        pass

    def reflect(self, task: str, result: Dict[str, Any]) -> str:
        """Reflect on the execution results and provide insights."""
        reflection_prompt = f"""
        Task: {task}
        Result: {result}
        
        Please analyze the execution results and provide insights on:
        1. What worked well
        2. What could be improved
        3. Alternative approaches that could have been taken
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": reflection_prompt}]
        )
        
        return response.choices[0].message.content
