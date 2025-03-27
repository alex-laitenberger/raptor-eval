import logging
import os

from openai import OpenAI


import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential, wait_exponential, after_log, before_sleep_log
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .utils import (buildMultipleChoiceQuestionText)

logger = logging.getLogger(__name__)

class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass

class OpenAI_QAModel_MultipleChoice(BaseQAModel):
    def __init__(self, modelString, client):
        """
        Initializes the OpenAI model with the model name set in the modelString

        Args:
            modelName (str): The OpenAI model.
        """
        self.modelString = modelString
        self.client = client

    @retry(wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(10), 
        before_sleep=before_sleep_log(logger, logging.INFO), 
        after=after_log(logger, logging.INFO), 
        reraise=True)
    def answer_question(
        self, context, question, options, tokenizer
    ):
        """
        Generates Answers to specified multiple choice questions and options optimized for QuALITY benchmark.
        """
        questionAndOptions = buildMultipleChoiceQuestionText(question, options)

        prompt = f'''
[Start of Context]:

{context}

[End of Context]

[Start of Question]:

{questionAndOptions}

[End of Question]

[Instructions:]
Based on the context provided, select the most accurate answer to the question from the given options.
Start with a short explanation and then provide your answer as [[1]] or [[2]] or [[3]] or [[4]]. 
For example, if you think the most accurate answer is the first option, respond with [[1]].
'''
        promptLog = f"\n\n#### Prompting {self.modelString}: ####\n\n{prompt}\n\n#### End of Prompt ####\n\n"
        logging.debug(promptLog)

        used_input_tokens = len(tokenizer.encode(prompt))

        response = self.client.chat.completions.create(
            model=self.modelString,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            seed = 42,

        )

        answerString = response.choices[0].message.content.strip()
        
        answerLog = f"\n\n#### {self.modelString} Response: ####\n\n{answerString}\n\n#### End of Response ####\n\n"
        logging.debug(answerLog)
        
        return answerString, used_input_tokens

class OpenAI_QAModel_Generation(BaseQAModel):
    def __init__(self, modelString, client):
        """
        Initializes the OpenAI model with the model name set in the modelString

        Args:
            modelName (str): The OpenAI model.
        """
        self.modelString = modelString
        self.client = client

    @retry(wait=wait_exponential(multiplier=1, max=60), 
        stop=stop_after_attempt(10), 
        before_sleep=before_sleep_log(logger, logging.INFO), 
        after=after_log(logger, logging.INFO), 
        reraise=True)
    def answer_question(
        self, context, question, options, tokenizer, max_tokens=150
    ):
        """
        Generates Answers to specified multiple choice questions and options optimized for QuALITY benchmark.
        """

        prompt = f'''
[Start of Context]:

{context}

[End of Context]

[Start of Question]:

{question}

[End of Question]

[Instructions:]
- Answer the question **only** based on the provided context.
- Keep the answer **short and factual** (preferably between 1-20 words).
- Do **not** provide explanations or additional details beyond what is necessary.
- If the answer is **not explicitly stated** in the context, respond with: "Not found in context."

'''
        promptLog = f"\n\n#### Prompting {self.modelString}: ####\n\n{prompt}\n\n#### End of Prompt ####\n\n"
        logging.debug(promptLog)
        #print(promptLog)

        used_input_tokens = len(tokenizer.encode(prompt))

        response = self.client.chat.completions.create(
            model=self.modelString,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            seed = 42,

        )

        answerString = response.choices[0].message.content.strip()
        
        answerLog = f"\n\n#### {self.modelString} Response: ####\n\n{answerString}\n\n#### End of Response ####\n\n"
        logging.debug(answerLog)
        #print(answerLog)
        
        return answerString, used_input_tokens



class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.completions.create(
                prompt=f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()

        except Exception as e:
            print(e)
            return ""


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4"):
        """
        Initializes the GPT-3 model with the specified model version.

        Args:
            model (str, optional): The GPT-3 model version to use for generating summaries. Defaults to "text-davinci-003".
        """
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]
