import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random, wait_random_exponential, wait_exponential, after_log, before_sleep_log

logger = logging.getLogger(__name__)

#logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass

class OpenAISummarizationModel(BaseSummarizationModel):
    def __init__(self, modelString, client):

        self.model = modelString
        self.client = client
        
    @retry(wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(10), 
        before_sleep=before_sleep_log(logger, logging.INFO), 
        after=after_log(logger, logging.INFO), 
        reraise=True)
    def summarize(self, context, max_tokens=500):

        #logging.info(f"Summarizing \"{context[:20]}...\"")

        #raise Exception("Test exception")

        

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                },
            ],
            max_tokens=max_tokens,
        )

        result =  response.choices[0].message.content

        if not isinstance(result, str):
            import traceback
            logger.error("[BAD TYPE ERROR] Summarization returned non-str!")
            logger.error(f"Type: {type(result)} — Value: {result}")
            
            # Capture and log the current stack trace
            stack_trace = "".join(traceback.format_stack())
            logger.error("Stack trace:\n%s", stack_trace)
            
        return result

class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="gpt-3.5-turbo"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e


class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = OpenAI()

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"Write a summary of the following, including as many key details as possible: {context}:",
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(e)
            return e
