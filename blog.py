import openai
import config
import numpy as np

openai.api_key = config.OPENAI_API_KEY

def write_blog(prompt1):
    response = openai.Completion.create(
      engine="davinci-instruct-beta-v3",
      prompt=prompt1,
      temperature=0.7,
      max_tokens=650,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']

def continue_blog(prompt1):
    response = openai.Completion.create(
      engine="davinci-instruct-beta-v3",
      prompt=f"Continue writing this blog:\n\n{prompt1}\n\n",
      temperature=0.6,
      max_tokens=200,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']


def conclude_blog(prompt1):
    response = openai.Completion.create(
      engine="davinci-instruct-beta-v3",
      prompt=f"Write a concluding section for this blog:\n\n{prompt1}\n\n",
      temperature=0.7,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    return response['choices'][0]['text']

# def write_email_subject_lines(email_body):
#     response = openai.Completion.create(
#         model="text-davinci-002",
#         prompt=f"Create a list of email subject lines for the following email:\n\n{email_body}\n\n1.",
#         temperature=0.7,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#
#     return response['choices'][0]['text']

def write_sms_campaigns(sms_body):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f'Write an SMS promotion for the following event:\n\n{sms_body}\n\n',
        temperature=0.5,
        max_tokens=60,
        top_p=1,
        n=5,
        frequency_penalty=0,
        presence_penalty=0
    )
    out_array = [r['text'].strip('\n') for r in response['choices']]
    sms_text = '\n\n--------------\n\n'.join(out_array) if out_array else ''

    return sms_text


class MultilineGenerations:
    def __init__(self):
        self.ignore_list = ['\n', 'tp_tokens', 'Q', ':']

    def generate_blog_topics(self, topic, keywords, num_topics):
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
        blog_topic = self.select_text(response)
        blog_topics = expand_blog_topics(blog_topic, topic, keywords, num_topics)
        blog_topics = f'1. {blog_topics}'
        return blog_topics

    def write_email_subject_lines(self, email_body, num_topics):
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
        esl = self.select_text(response)
        esl_list = expand_email_sl_topics(esl, email_body, num_topics)
        esl_list = f'1. {esl_list}'
        return esl_list

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


def expand_blog_topics(selected_topic, topic, keywords, number_of_topics=5):
    num = str(number_of_topics+1)
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Create an interesting blog title about:\n{topic}\nKeywords:\n{keywords}\n\n1. ",
        suffix=f"\n{num}. {selected_topic}\n\n\n",
        temperature=0.78,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.9,
        presence_penalty=1.63
    )
    return response['choices'][0]['text']

def expand_email_sl_topics(selected_topic, email_body, number_of_topics=5):
    num = str(number_of_topics+1)
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Create a list of email subject lines for the following email:\n\n{email_body}\n\n1.",
        suffix=f"\n{num}. {selected_topic}\n\n\n",
        temperature=0.78,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.9,
        presence_penalty=1.63
    )
    return response['choices'][0]['text']
