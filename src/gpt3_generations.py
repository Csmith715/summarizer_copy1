import openai
import config
import numpy as np
import re

openai.api_key = config.OPENAI_API_KEY


def write_blog(topic: str, keywords: str) -> str:
    prompt = f'Write a long detailed blog about:\n{topic}\nKeywords:\n{keywords}\n\n\n'
    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt=prompt,
        temperature=0.7,
        max_tokens=650,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    blog = response['choices'][0]['text'], response['usage']['total_tokens']
    blog = str(blog).strip('\n')
    return blog


class GPT3Creations:
    def __init__(self):
        self.ignore_list = ['\n', 'tp_tokens', 'Q', ':']
        self.function_dict = {
            "BLOG_TEXT": write_blog,
            "BLOG_TOPICS": self.generate_blog_topics,
            "EMAIL_SUBJECT_LINES": self.write_email_subject_lines
        }

    def select_and_write(self, content_type, topic, keywords):
        text, tokens = self.function_dict[content_type](topic, keywords)
        return text, tokens

    def generate_blog_topics(self, topic, keywords):
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f'Create an interesting blog title about:\n{topic}\nKeywords:\n{keywords}\n\n',
            temperature=1,
            max_tokens=25,
            top_p=1,
            frequency_penalty=1.0,
            presence_penalty=1.0,
            logprobs=1,
            n=5
        )
        first_tokens = response['usage']['total_tokens']
        blog_topic = self.select_text(response)
        blog_topics, expanded_tokens = expand_blog_topics(blog_topic, topic, keywords)
        blog_topics = string_to_array(blog_topics)
        total_tokens = first_tokens + expanded_tokens
        return blog_topics, total_tokens

    def write_email_subject_lines(self, email_body, keywords=None):
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=f"Create an email subject line for the following email:\n\n{email_body}\n\n",
            temperature=1,
            max_tokens=25,
            top_p=1,
            frequency_penalty=1.0,
            presence_penalty=1.0,
            logprobs=1,
            n=5
        )
        first_tokens = response['usage']['total_tokens']
        esl = self.select_text(response)
        esl_list, expanded_tokens = expand_email_sl_topics(esl, email_body)
        esl_text = string_to_array(esl_list)
        total_tokens = first_tokens + expanded_tokens
        return esl_text, total_tokens

    def gpt3_creation_probability(self, individual_response):
        tps = individual_response['logprobs']['token_logprobs']
        probs = np.exp(tps)
        tp_tokens = individual_response['logprobs']['tokens']
        avg_array = [y for x, y in zip(tp_tokens, probs) if x not in self.ignore_list]
        gpt3_avg = 0
        if avg_array:
            gpt3_avg = np.average(avg_array)
        return gpt3_avg

    def select_text(self, response):
        probs = [self.gpt3_creation_probability(g) for g in response['choices']]
        texts = [g['text'] for g in response['choices']]
        selected_text = texts[np.argmax(probs)]
        selected_text = [t for t in selected_text.split('\n') if t][0]
        return selected_text


def expand_blog_topics(selected_topic, topic, keywords, number_of_topics=10):
    num = str(number_of_topics+1)
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Create an interesting blog title about:\n{topic}\nKeywords:\n{keywords}\n\n1. ",
        suffix=f"\n{num}. {selected_topic}\n\n\n",
        temperature=0.78,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0.9,
        presence_penalty=1.63
    )
    return response['choices'][0]['text'], response['usage']['total_tokens']

def expand_email_sl_topics(selected_topic, email_body, number_of_topics=10):
    num = str(number_of_topics+1)
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Create a list of email subject lines for the following email:\n\n{email_body}\n\n1.",
        suffix=f"\n{num}. {selected_topic}\n\n\n",
        temperature=0.78,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0.9,
        presence_penalty=1.63
    )
    return response['choices'][0]['text'], response['usage']['total_tokens']

def string_to_array(text: str) -> list:
    split_text = text.split('\n')
    new_list = [re.sub('^\d+\.', '', s) for s in split_text]
    new_list = [n.strip() for n in new_list]
    return new_list
