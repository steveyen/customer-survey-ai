import unittest
import dotenv
import yaml

import survey_ai

import ragas4 # Fantasy, future RAGAS library.

import evalgelion # Fantasy, future eval'uations library.

from langchain_openai.chat_models import ChatOpenAI

from agentc.catalog import Catalog

# ------------------------------------------------------

def debug_info(depth=1):
    """
    Returns a string with the current file, function, and line number.

    Parameters:
        depth (int): How many levels up the call stack to go (default is 1, i.e., the caller).

    Returns:
        dict with metadata on the filename, function, lineno.
    """
    stack = inspect.stack()
    
    frame_info = stack[depth]

    return {
        "filename": os.path.basename(frame_info.filename),
        "function": frame_info.function,
        "lineno": frame_info.lineno,
        "str": f"{filename}:{function_name}:{line_number}"
    }

# ------------------------------------------------------

class TestSurveyAI(unittest.TestCase):

    def test_foo(self):
        pass

    def test_bar(self):
        pass

    def test_angry_surveys(self):
        
        dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

        test_catalog = Catalog()

        d = debug_info()

        test_span = catalog.Span(
            name=f"""{d.filename}#{d.function}"""
        )

        with open("./angry-surveys.yaml", 'r') as f:
            cases = yaml.safe_load(file)

            if True:
                from agentc.evaluations import prioritize_cases
                
                cases = prioritize_cases(test_catalog, test_span, cases)

            for c_idx, c in enumerate(cases):
                model = c.get("model", "gpt-4o")

                temperature = c.get("temperature", 0)
                                 
                test_llm = ChatOpenAI(model=model, temperature=temperature)

                with test_span.new("case", "c_idx"=c_idx) as c_span:

                    results = survey_ai.run(test_catalog, c_span, test_llm, c.input)

                    test_span["metrics.RAGAS4"] = RAGAS4.evaluate_professionalism(results)

                    test_span["metrics.evalgelion"] = evalgelion.process(
                        "info-completeness", results)

                    ok = True
                    for x in results:
                        if "sh*t" in x:
                            ok = False

                    test_span["metrics.cuss-words"] = ok

                    test_span["metrics.uppercase"] = results.contains_uppercase()

                print("done with test case:", c_idx)


