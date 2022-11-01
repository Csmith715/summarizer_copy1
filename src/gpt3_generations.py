import openai
import config
import numpy as np
import re
# import requests
import logging

logger = logging.getLogger()
openai.api_key = config.OPENAI_API_KEY
vowels = ['a', 'e', 'i', 'o', 'u']


def write_blog(topic: str, keywords: str):
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

    blog = response['choices'][0]['text']
    tokens = response['usage']['total_tokens']
    blog = str(blog).strip('\n')
    return blog, tokens


# def create_bullet_list(title: str, introduction: str) -> str:
#     post_data = {
#         "introduction": [introduction],
#         "title": title,
#         "bullet_points": {"Email Bullet List": []},
#         "more_info": [],
#         "keywords": [],
#         "accountId": "QWNjb3VudHwx",
#         "topics": []
#     }
#     result = requests.post('https://ai.dev.contentware.com/find-content/gpt3-creations', json=post_data)
#     bpoint_list = result.json()['bullet_points']
#     if bpoint_list:
#         final_list = bpoint_list[0]
#     else:
#         final_list = ''
#     return final_list

def create_bullet_list(title: str, introduction: str) -> str:
    # n is set to 5 to create 5 outputs
    text = f'{title}\n{introduction}'
    response = openai.Completion.create(
        model='curie:ft-contentware-2022-10-04-15-51-24',
        prompt=f"{text}\n\nA concise learning objective:\n\n",
        n=5,
        temperature=0.7,
        max_tokens=25,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.2,
        stop=["\n", ". ", "?", "!"]
    )
    out_array = [r['text'].strip('\n') for r in response['choices']]
    final_list = '\n'.join(out_array)
    final_list = f'Key Takeaways:\n{final_list}'

    return final_list

# def prompt_writer(promotion_text: str, promotion_type: str) -> str:
#     ptype = promotion_type.lower()
#     if ptype[0] in vowels:
#         prompt = f"Write a LinkedIn post for an {ptype} based on the following content:\n\n{promotion_text}\n\n"
#     else:
#         prompt = f"Write a LinkedIn post for a {ptype} based on the following content:\n\n{promotion_text}\n\n"
#     return prompt

def social_media_prompt(channel: str, title: str, keywords: str) -> list:
    # n is set to 5 to create 5 outputs
    text = f'{title}\nKeywords: {keywords}'
    response = openai.Completion.create(
        model='curie:ft-contentware-2022-06-01-19-09-59',
        prompt=f"Write a {channel} post based on the following content:\n\n{text}\n\n",
        n=5,
        temperature=0.5,
        max_tokens=120,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1
    )
    out_array = [r['text'].strip('\n') for r in response['choices']]
    return out_array

def write_introduction(event_type: str, title: str, keywords: str):
    prompt = f'Write an introduction summary about this {event_type}:\nTitle: {title}\nKeywords:\n{keywords}\n\n\n'
    response = openai.Completion.create(
        engine="davinci-instruct-beta-v3",
        prompt=prompt,
        temperature=0.7,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    introduction = response['choices'][0]['text']
    tokens = response['usage']['total_tokens']
    introduction = str(introduction).strip('\n')
    return introduction, tokens


def create_landing_page(event_type: str, title: str, keywords: str):
    logger.info('Writing introduction')
    intro, tokes = write_introduction(event_type, title, keywords)
    logger.info('Introduction complete, creating bullets')
    bullet_list = create_bullet_list(title, intro)
    logger.info('Bullets Complete')
    landing_page = f'{intro}\n\n{bullet_list}'
    return landing_page, tokes


class GPT3Creations:
    def __init__(self):
        self.ignore_list = ['\n', 'tp_tokens', 'Q', ':']
        # self.topic = ''
        # self.keywords = ''
        # self.event_type = ''
        self.function_dict = {
            "BLOG_TEXT": write_blog,
            "BLOG_TOPICS": self.generate_blog_topics,
            "EMAIL_SUBJECT_LINES": self.write_email_subject_lines,
            "LANDING_PAGE": create_landing_page,
            "SOCIAL_MEDIA_POST": social_media_prompt
        }

    def select_and_write(self, content_type, topic, keywords, event_type):
        # self.topic = topic
        # self.keywords = keywords
        # self.event_type = event_type
        if event_type:
            text, tokens = self.function_dict[content_type](event_type, topic, keywords)
        else:
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
    new_list = [re.sub('^\d+.\.', '', s) for s in new_list]
    new_list = [n.strip() for n in new_list if len(n) > 10]
    new_list = [n.strip('"') for n in new_list]
    return new_list
