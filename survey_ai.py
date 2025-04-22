import argparse
import dotenv
import getpass
import os

from langchain_openai.chat_models import ChatOpenAI

from langchain_core.messages import SystemMessage

from langgraph.types import Command
from langgraph.graph import START
from langgraph.graph import END
from langgraph.graph import StateGraph

from agentc.catalog import Catalog

from agentc_langgraph.agent import ReActAgent
from agentc_langgraph.agent import State
from agentc_langgraph.graph import GraphRunnable

# ------------------------------------------

def get_next_node(last_message, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent has decided that the work is done.
        return END
    return goto


# ------------------------------------------

class SurveyGeneratorAgent(ReActAgent):
    def __init__(self, catalog, span, llm):
        super().__init__(catalog=catalog,
                         span=span,
                         prompt_name="survey_generator_agent",
                         chat_model=llm)

    def _invoke(self, span, state, config):
        agent = self.create_react_agent(span)
        
        result = agent.invoke(state)
        
        goto = get_next_node(result["messages"][-1], "chart_generator")
        
        result["messages"][-1] = SystemMessage(
            content=result["messages"][-1].content, name="survey_generator")
        
        return Command(update={"messages": result["messages"]}, goto=goto)

# ------------------------------------------

class ChartGeneratorAgent(ReActAgent):
    def __init__(self, catalog, span, llm):
        super().__init__(catalog=catalog,
                         span=span,
                         prompt_name="chart_generator_agent",
                         chat_model=llm)

    def _invoke(self, span, state, config):
        agent = self.create_react_agent(span)
        
        result = agent.invoke(state)
        
        goto = get_next_node(result["messages"][-1], "survey_generator")
        
        result["messages"][-1] = SystemMessage(
            content=result["messages"][-1].content, name="chart_generator")
        
        return Command(update={"messages": result["messages"]}, goto=goto)

# ------------------------------------------

class CustomerSurveyAgentGraph(GraphRunnable):
    def __init__(self, catalog, span):
        super().__init__(catalog=catalog,
                         span=span)

    def compile(self):
        workflow = StateGraph(State)
        
        workflow.add_node("survey_generator",
                          SurveyGeneratorAgent(catalog=self.catalog, span=self.span))
        
        workflow.add_node("chart_generator",
                          ChartGeneratorAgent(catalog=self.catalog, span=self.span))

        workflow.add_edge(START, "survey_generator")
        
        return workflow.compile()

# ------------------------------------------

def run(catalog, span, llm, user_input):
    graph = CustomerSurveyAgentGraph(catalog, span)

    span.log(content={"kind": "user", "value": user_input})

    events = graph.stream(
        {"messages": [("user", user_input)],
         "is_last_step": False,
         "previous_node": None},
        {"recursion_limit": 10})

    msgs = []

    for event in events:
        for key in event:
            for msg in event[key]["messages"]:
                if isinstance(msg, SystemMessage):
                    msgs.append(msg.content)
                    
                    print(msg.content)
                    print("----")

    return msgs

# ------------------------------------------

def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"please provide env var: {var}")

# ------------------------------------------

if __name__ == "main":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default="""
        First, get the customer surveys created over the last 6 months,
        then give a brief summary of them.

        Use pie charts as needed.
    """)
    
    parser.add_argument("--model", type=str, default="gpt-4o")
    
    parser.add_argument("--temperature", type=int, default=0)

    args = parser.parse_args()
   
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

    top_catalog = Catalog()

    top_span = catalog.Span(name="survey_ai.py")

    _set_if_undefined("OPENAI_API_KEY")

    top_llm = ChatOpenAI(model=args.model, temperature=args.temperature / 100.0)

    run(top_catalog, top_span, top_llm, args.input)
)

