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

dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

def _set_if_undefined(var: str):
    if os.environ.get(var) is None:
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ------------------------------------------

catalog = Catalog()

app_span = catalog.Span(
    name="customer survey expert AI",
)


# ------------------------------------------

def get_next_node(last_message, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent has decided that the work is done.
        return END
    return goto


# ------------------------------------------

class ResearchAgent(ReActAgent):
    def __init__(self, span):
        super().__init__(catalog=catalog,
                         prompt_name="researcher_agent",
                         span=span,
                         chat_model=llm)

    def _invoke(self, span, state, config):
        agent = self.create_react_agent(span)
        
        result = agent.invoke(state)
        
        goto = get_next_node(result["messages"][-1], "chart_generator")
        
        result["messages"][-1] = SystemMessage(
            content=result["messages"][-1].content, name="researcher")
        
        return Command(
            update={
                # share internal message history of research agent with other agents
                "messages": result["messages"],
            },
            goto=goto,
        )

# ------------------------------------------

class CharterAgent(ReActAgent):
    def __init__(self, span):
        super().__init__(catalog=catalog,
                         prompt_name="charter_agent",
                         span=span,
                         chat_model=llm)

    def _invoke(self, span, state, config):
        agent = self.create_react_agent(span)
        
        result = agent.invoke(state)
        
        goto = get_next_node(result["messages"][-1], "researcher")
        
        result["messages"][-1] = SystemMessage(
            content=result["messages"][-1].content, name="researcher")
        
        return Command(
            update={
                # share internal message history of research agent with other agents
                "messages": result["messages"],
            },
            goto=goto,
        )

# ------------------------------------------

class CustomerSurveyAgenticGraph(GraphRunnable):
    def __init__(self):
        super().__init__(catalog=catalog,
                         span=application_span)

    def compile(self):
        workflow = StateGraph(State)
        
        workflow.add_node("researcher", ResearchAgent(span=self.span))
        workflow.add_node("chart_generator", CharterAgent(span=self.span))

        workflow.add_edge(START, "researcher")
        
        return workflow.compile()

# ------------------------------------------

def run(user_input):
    graph = CustomerSurveyAgenticGraph()

    application_span.log(content={"kind": "user", "value": user_input})

    events = graph.stream(
        {"messages": [("user", user_input)], "is_last_step": False, "previous_node": None},

        # Maximum number of steps to take in the graph
    
        {"recursion_limit": 10},
    )

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

if __name__ == "main":
    run("""
            First, get the customer surveys created over the last 6 months,
            then give a brief summary of them.

            Use pie charts as needed.
        """)

