import openai
import config
import numpy as np
import re
import logging
from nltk.tokenize import sent_tokenize

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
    out_array = [f'- {x}' for x in out_array]
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

def social_media_prompt(channel: str, title: str, keywords: str):
    # n is set to 10 to create 10 outputs
    text = f'{title}\nKeywords: {keywords}'
    response = openai.Completion.create(
        # model='curie:ft-contentware-2022-06-01-19-09-59',
        engine="davinci-instruct-beta-v3",
        prompt=f"Write a {channel} post based on the following content:\n\n{text}\n\n",
        n=10,
        temperature=0.5,
        max_tokens=120,
        top_p=1,
        frequency_penalty=1,
        presence_penalty=1,
        # stop=["\n"]
    )
    out_array = [r['text'].strip('\n') for r in response['choices']]
    lr = [len(r) for r in out_array]
    sorted_array = [x for _, x in sorted(zip(lr, out_array))]
    final_array = sorted_array[:2] + [sorted_array[4]] + sorted_array[8:]
    tokens = response['usage']['total_tokens']
    return final_array, tokens

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
        self.function_dict = {
            "BLOG_TEXT": write_blog,
            "BLOG_TOPICS": self.generate_blog_topics,
            "EMAIL_SUBJECT_LINES": self.write_email_subject_lines,
            "LANDING_PAGE": create_landing_page,
            "SOCIAL_MEDIA_POST": social_media_prompt
        }

    def select_and_write(self, content_type, topic, keywords, event_type):
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

class NewGPT3Content:
    def __init__(self, data: dict):
        self.title = data['title']  # string
        self.summary = data['summary']  # string
        # self.focus = data['snippet']
        self.bullets = data['snippets']  # array
        self.date = data['date']
        # self.tone = data['tone']
        self.promotion_type = data['promotion type']
        self.sm_type = data['social media type']
        self.final_prompt = ''

    def generate_social_media_content(self):
        self.create_social_media_prompt()
        posts = self.write_social_media()
        if self.bullets:
            posts = self.scrub_output(posts)
        return posts

    def create_social_media_prompt(self):
        prompt_a = self.craft_prompt()
        if self.sm_type:
            prompt_b = f'Create a varied series of long {self.sm_type} posts from this content:\n'
        # if self.tone and self.sm_type:
        #     prompt_b = f'Create a {self.tone} {self.sm_type} post from this content:\n'
        # elif self.tone and not self.sm_type:
        #     prompt_b = f'Create a {self.tone} social media post from this content:\n'
        # elif self.sm_type and not self.tone:
        #     prompt_b = f'Create a {self.sm_type} post from this content:\n'
        # prompt = f"Title: {t}\nSummary: {s}\nPromotion Type: Webinar\n{foci}\nCreate a varied series of long Facebook posts from this content:\n\nFocus 1:"
        else:
            prompt_b = 'Create a varied series of long social media posts from this content\n'
        if self.bullets:
            prompt_b = f'{prompt_b}\nFocus 1:'
        self.final_prompt = f'{prompt_a}\n{prompt_b}'

    def craft_prompt(self):
        prompt = ''
        if self.title:
            prompt = f'Title: {self.title}\n'
        if self.summary:
            prompt = f'{prompt}Summary: {self.summary}\n'
        if self.date:
            prompt = f'{prompt}Date: {self.date}\n'
        # if self.focus:
        #     prompt = f'{prompt}Focus: {self.focus}\n'
        if self.promotion_type:
            prompt = f'{prompt}Promotion Type: {self.promotion_type}\n'
        if self.bullets:
            foci = self.arrange_focal_points()
            prompt = f'{prompt}{foci}\n'
        return prompt

    def write_social_media(self) -> list:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.final_prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0.25,
            presence_penalty=0.25,
            n=5
        )
        out_array = [r['text'].strip('\n') for r in response['choices']]
        out_array = clean_gpt3_output(out_array)
        cleaned_array = [remove_sm_hashtags(s_post) for s_post in out_array]

        return cleaned_array

    def scrub_output(self, out_posts: list) -> list:
        new_clean_posts = []
        for post in out_posts:
            first_sentence = sent_tokenize(post)[0]
            first_sentence = first_sentence.split('\n')[0]
            temp_list = [f.strip('.').strip('!').strip('?').lower() for f in self.bullets]
            new_text = ''
            for t in temp_list:
                ss = f'^{t}'
                search_result = re.search(ss, first_sentence.lower())
                if search_result and len(first_sentence[search_result.end():]) < 3:
                    new_text = post[search_result.end():]
                    new_text = new_text[1:] if new_text[0] == '.' else new_text
                    new_text = new_text.strip().strip('\n')
                    new_text = re.sub('^\[Post \d\] ', '', new_text)
            if new_text:
                new_clean_posts.append(new_text)
            else:
                new_clean_posts.append(post)
        return new_clean_posts

    def arrange_focal_points(self):
        focal_list = [f'Focus {i + 1}: {f}\n' for i, f in enumerate(self.bullets)]
        return ''.join(focal_list)

def remove_sm_hashtags(post: str):
    clean_text = f'{post} '
    fi = re.finditer('#(.*?) ', clean_text)
    if fi:
        for f in fi:
            clean_text = clean_text.replace(f.group(), '')
    return clean_text.strip()

def clean_gpt3_output(output_array: list) -> list:
    cleaned = []
    for out in output_array:
        rs = re.split('\nFocus \d+:', out)
        crs = [r.strip('\n') for r in rs]
        frs = [remove_sm_hashtags(c.strip()) for c in crs if c]
        cleaned.extend(frs)
    cleaned = list(set(cleaned))
    return cleaned
