import openai
import logging
import concurrent.futures
import re
from config import url_string, emoji_pattern, OPENAI_API_KEY
import math
from itertools import chain
# import time
import random

logger = logging.getLogger()
openai.api_key = OPENAI_API_KEY


class SocialContentCreation:
    def __init__(self, title: str, description: str, keywords: str, containers: list):
        self.title = title
        self.summary = description
        self.input_prompts = []
        self.keywords = keywords
        self.containers = containers
        self.social_result_dict = {}
        self.input_prompts = []
        self.social_keys = ['Facebook', 'Twitter', 'Instagram']
        # self.social_keys = ['Facebook', 'Twitter']
        # self.social_keys = ['Facebook', 'Instagram']
        # self.social_keys = ['Twitter', 'Instagram']
        self.map_dict = {}
        self.data = {}

    def make_social_creations(self):
        self.create_prompts()
        self.make_social_posts()
        self.map_containers()
        return self.data

    def create_prompts(self):
        post_text = '\n\nYou are a helpful assistant that specializes in creating social media content.'
        system_text = f'Title: {self.title}\nDescription: {self.summary}\nKeywords: {self.keywords}{post_text}'
        for key in self.social_keys:
            self.input_prompts.append(
                (
                    system_text,
                    f'Create 5 {key} text only posts from the content provided. Number each post.\n\n',
                    key
                )
            )

    def make_social_posts(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_chat_text, sys_text, u_text, soc_key) for sys_text, u_text, soc_key in self.input_prompts]
        results = [future.result() for future in futures]
        for res in results:
            split_posts = res[0].split('\n')
            split_posts = [clean_gpt_text(s) for s in split_posts if s]
            self.social_result_dict[res[1]] = split_posts
        self.map_dict["Fb-LI"] = self.social_result_dict.get('Facebook', [])
        self.map_dict["IG"] = self.social_result_dict.get('Instagram', [])
        self.map_dict["TW"] = self.social_result_dict.get('Twitter', [])

    def map_containers(self):
        for container in self.containers:
            container_key = container['content']
            if '#image asset=' in container_key:
                self.data[container_key] = [container_key]*len(container['positions'])
            else:
                self.data[container_key] = {
                    "content": [],
                    "substitutions": []
                }
            fc_value = re.search('{#fullcontent value="(.*?)"', container_key)
            if fc_value:
                base_val = fc_value.group()
                fcv = fc_value.group(1)
                content_vals = self.map_dict[fcv][:len(container['positions'])]
                content_vals = [re.sub(base_val, f'{base_val} content="{cv}"', container_key) for cv in content_vals]
                self.data[container_key]['content'] = content_vals
                self.data[container_key]['substitutions'] = [re.sub(base_val, f'{base_val} content="{m}"', container_key) for m in self.map_dict[fcv]]
            elif '#image asset=' not in container_key:
                self.data[container_key]['content'] = [container_key]
                self.data[container_key]['substitutions'] = []

class SocialGenerations:
    def __init__(
            self,
            snippets: list,
            job_title: str,
            introduction: str,
            promotion_type: str,
            action_verb: str,
            promo_val: str
    ):
        self.chunked_snippets = list(divide_chunks(snippets, 5))
        self.snippets = snippets
        self.title = job_title
        self.summary = introduction
        self.promotion = promotion_type
        self.action_verb = action_verb
        self.input_prompts = []
        self.result_dict = {
            "davinci:ft-contentware:esl-generation-2023-04-21-16-37-03": [],            # 'email subject lines'
            "davinci:ft-contentware:instagram-generation-v2-2023-04-17-01-40-04": [],   # 'Instagram'
            "gpt-4-fb": [],      # Facebook Ads
            "gpt-4-li": [],      # LinkedIn Ads
            "gpt-4-eh": [],       # Email Headlines
            "gpt-4-buttons": [],   # CTA Buttons
            "gpt3.5-scta": []  # shortcta
        }
        # self.question_model = seq2seq_model
        self.ad_n_count = math.ceil(10 / len(self.chunked_snippets))
        self.promo_val = promo_val

    def create_socials(self):
        self.make_input_prompts()
        self.make_gpt()
        return self.result_dict

    def make_input_prompts(self):
        if self.action_verb:
            form1 = f"This is content for an upcoming {self.promotion}:\n\nTitle: {self.title}\nSummary: {self.summary}\nAction Verb: {self.action_verb}\nObjectives:\n\n"
            # form2 = f"This is content for an upcoming {self.promotion}:\n\nTitle: {self.title}\nSummary: {self.summary}\nAction Verb: {self.action_verb}\n"
        else:
            form1 = f"This is content for an upcoming {self.promotion}:\n\nTitle: {self.title}\nSummary: {self.summary}\nObjectives:\n\n"
            # form2 = f"This is content for an upcoming {self.promotion}:\n\nTitle: {self.title}\nSummary: {self.summary}\n"
        igram_suffix = f'\n\nCreate a varied series of short Instagram posts that promotes this {self.promotion}.'
        esl_suffix = f'\n\nCreate a varied series of email subject lines that promotes this {self.promotion}. Number each subject line.'
        head_suffix = f'\n\nCreate ten email headlines that will promote this {self.promotion}. '
        short_cta_suffix = f'\n\nWrite ten short Call to Action sentences for an email communication about this {self.promotion} that will encourage a response. '
        # button1_suffix = f'Create fifteen single word action button phrases that would encourage a user to click on an ad that promotes this {self.promotion}. '
        # button2_suffix = f'Create fifteen two word action button phrases that would encourage a user to click on an ad that promotes this {self.promotion}. '
        button1_suffix = f'Create 15 single word email call to action buttons that would encourage the reader to engage in a {self.promotion}. '
        button2_suffix = f'Create 15 two word email call to action buttons that would encourage the reader to engage in a {self.promotion}. '
        unfocused_bullets = []
        for chunk in self.chunked_snippets:
            unfocused_blist = [f'- {bul}' for bul in chunk]
            unfocused_bullets.append('\n'.join(unfocused_blist))
        random_ufb = random.choice(unfocused_bullets)
        for ufb in unfocused_bullets:
            # if self.promo_val == 'SINGLE_SOCIAL_POST': # this is the value Mitch and David agreed on mid-May
            if self.promo_val == 'ANY_VALUE_not listed':  # An unnecessary change in case SINGLE_SOCIAL_POST is passed accidentally
                self.input_prompts.append(
                    (
                        f'{form1}{ufb}\n\nYou are an expert marketing campaign writer.',
                        "gpt-4-fb",
                        75,
                        self.ad_n_count
                    )
                )
                self.input_prompts.append(
                    (
                        f'{form1}{ufb}\n\nYou are an expert marketing campaign writer.',
                        "gpt-4-li",
                        75,
                        self.ad_n_count
                    )
                )
            self.input_prompts.append(
                (
                    f"{form1}{ufb}{igram_suffix}Each post should be less than 180 characters. Number each post.\n\n",
                    "davinci:ft-contentware:instagram-generation-v2-2023-04-17-01-40-04",
                    100,
                    3
                )
            )
            self.input_prompts.append(
                (
                    f'{form1}{ufb}{esl_suffix}\n\n',
                    "davinci:ft-contentware:esl-generation-2023-04-21-16-37-03",
                    75,
                    3
                )
            )
            # self.input_prompts.append(
            #     (
            #         f'{form1}{ufb}\n\nWrite a short Call to Action sentence for an email communication about this {self.promotion} that will encourage a response.\n\n',
            #         "davinci:ft-contentware:email-cta-v2-2023-05-04-23-04-53",
            #         6,
            #         10
            #     )
            # )
        self.input_prompts.append(
            (
                f'{form1}{random_ufb}{short_cta_suffix}Each Call to Action should be at most 8 words long. Number each Call to Action.',
                "gpt3.5-scta",
                80,
                3
            )
        )
        self.input_prompts.append(
            (
                f'{form1}{random_ufb}{head_suffix}At least one should be in the form of a question with a response to that question. Separate the headlines by "\n" and do not '
                f'number them.',
                "gpt-4-eh",
                120,
                2
            )
        )
        self.input_prompts.append(
            (
                f'{button1_suffix}Number each button phrase.\n\n',
                "gpt-4-buttons1",
                35,
                1
            )
        )
        self.input_prompts.append(
            (
                f'{button2_suffix}Number each button phrase.\n\n',
                "gpt-4-buttons2",
                35,
                1
            )
        )

    def make_gpt(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.generate_text, prompt, model, mt, nval) for prompt, model, mt, nval in self.input_prompts]
        results = [future.result() for future in futures]
        for res in results:
            model = res['model']
            if model == 'gpt-4-fb':
                self.result_dict['gpt-4-fb'] = res['result']
            elif model == 'gpt-4-li':
                self.result_dict['gpt-4-li'] = res['result']
            elif model == 'gpt-4-eh':
                self.result_dict['gpt-4-eh'] = clean_headlines(res['result'])
            elif model == 'gpt-4-buttons1' or model == 'gpt-4-buttons2':
                cleaned_buttons = clean_buttons(res['result'])
                self.result_dict['gpt-4-buttons'].extend(cleaned_buttons)
            elif model == 'gpt3.5-scta':
                self.result_dict['gpt3.5-scta'] = clean_short_ctas(res['result'])
            else:
                texts = [c['text'] for c in res.choices]
                clean_texts = clean_gpt_list(texts, model)
                self.result_dict[model].extend(clean_texts)

    def generate_chat_text(self, user_text, system_text, repetitions, model_id):
        prompt_message = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ]
        try:
            chat_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=prompt_message,
                n=repetitions,
            )
            final_response = [g.message.content for g in chat_response.choices]
        except Exception as e:
            logger.info(e)
            final_response = [self.snippets]
        return {
            'model': model_id,
            'result': final_response
        }

    def generate_text(self, prompt, model, max_tokes, n_value):
        if model == 'gpt-4-fb':
            response = self.generate_chat_text(
                'For each objective listed, write the main body for a Facebook Ad. Number each post, do not use emojis, and do not exceed 125 characters for each post.',
                prompt,
                n_value,
                model
            )
        elif model == 'gpt-4-li':
            response = self.generate_chat_text(
                'For each objective listed, write the main body for a LinkedIn Ad. Number each post, do not use emojis, and do not exceed 150 characters for each post.',
                prompt,
                n_value,
                model
            )
        elif model == 'gpt-4-eh':
            response = self.generate_chat_text(
                prompt,
                'You are an expert at writing promotional material.',
                n_value,
                model
            )
        elif model == 'gpt-4-buttons1':
            response = self.generate_chat_text(
                prompt,
                'You are an expert at writing promotional material.',
                n_value,
                model
            )
        elif model == 'gpt-4-buttons2':
            response = self.generate_chat_text(
                prompt,
                'You are an expert at writing promotional material.',
                n_value,
                model
            )
        elif model == 'gpt3.5-scta':
            response = self.generate_chat_text(
                prompt,
                'You are an expert at writing promotional material.',
                n_value,
                model
            )
        else:
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=max_tokes,
                frequency_penalty=0.25,
                presence_penalty=0.25,
                n=n_value
            )
        # te = time.time()
        # tdiff = te-ts
        # print(f'{model}: {tdiff}')
        return response

def divide_chunks(list_to_chunk, n):
    for i in range(0, len(list_to_chunk), n):
        yield list_to_chunk[i:i + n]

def remove_sm_hashtags(post: str):
    clean_text = f'{post} '
    fi = re.finditer('#(.*?) ', clean_text)
    if fi:
        for f in fi:
            clean_text = clean_text.replace(f.group(), '')
    return clean_text.strip()

def clean_gpt_text(post: str) -> str:
    clean_text = f'{post} '
    fi = re.finditer('#(.*?) ', clean_text)
    if fi:
        for f in fi:
            clean_text = clean_text.replace(f.group(), '')
    clean_text = re.sub(emoji_pattern, ' ', clean_text)
    clean_text = re.sub(r'^\d+\. ', '', clean_text)
    clean_text = clean_text.replace('[link]', '').replace('[Link]', '').replace('[Link to article]', '').replace('[LINK]', '').replace('\u200d', '').replace('  ', ' ')
    clean_text = clean_text.replace('  ', ' ').replace('"', '')
    clean_text = remove_sm_hashtags(clean_text)
    return clean_text.strip()

def clean_gpt_list(output_array: list, model_name: str) -> list:
    cleaned = []
    for out in output_array:
        rs = out.split('\n')
        crs = [r.strip('\n') for r in rs if r]
        frs = [clean_gpt_text(c.strip()) for c in crs if c]
        frs = [f.split('\n')[0] for f in frs]
        if model_name == 'davinci:ft-contentware:instagram-generation-v2-2023-04-17-01-40-04"':
            frs = [fr for fr in frs if re.search('[?.!]$', fr)]
        else:
            frs = frs[:-1]
        ufs = [remove_url(f) for f in frs]
        cleaned.extend(ufs)
    if model_name == 'davinci:ft-contentware:email-cta-v2-2023-05-04-23-04-53':
        cleaned = output_array
    cleaned = list(set(cleaned))
    return cleaned

def remove_url(social_post: str) -> str:
    url_free_string = re.sub(url_string, '', social_post, )
    return url_free_string

def drop_last(esl_post: str) -> str:
    sesl = esl_post.split('\n')
    fixed_esl = '\n'.join(sesl[:-1])
    return fixed_esl

def generate_chat_text(user_text, system_text='You are a helpful assistant', social_key=None):
    # logging.info(f'{social_key} Started')
    prompt_message = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text}
    ]
    try:
        collected_messages = []
        chat_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt_message,
            n=1,
            stream=True
        )
        for chunk in chat_response:
            chunk_message = chunk['choices'][0]['delta']  # extract the message
            collected_messages.append(chunk_message)  # save the message
        # final_response = chat_response.choices[0]['message']['content']
        final_response = ''.join([m.get('content', '') for m in collected_messages])
    except Exception as e:
        final_response = f"ChatGPT isn't working right now. Here is the error they sent us:\n{e}"
    # logging.info(f'{social_key} Complete')
    return final_response, social_key

def clean_headlines(email_headlines: list):
    cleaned = []
    for e in email_headlines:
        headlines = e.split('\n')
        headlines = [h.strip().strip('"') for h in headlines if h]
        cleaned.append(headlines)
    return list(chain.from_iterable(cleaned))

def clean_buttons(cta_buttons: list) -> list:
    cleaned = []
    if cta_buttons:
        # print(cta_buttons)
        button_text = cta_buttons[0]
        split_buttons = button_text.split('\n')
        for s in split_buttons:
            clean_text = re.sub(r'^\d+\. ', '', s)
            clean_text = clean_text.strip().strip('"')
            cleaned.append(clean_text)
    return cleaned

def clean_short_ctas(short_ctas: list) -> list:
    cleaned = []
    for response in short_ctas:
        split_response = response.split('\n')
        for s in split_response:
            clean_text = re.sub(r'^\d+\. ', '', s)
            clean_text = clean_text.strip().strip('"')
            cleaned.append(clean_text)
    return cleaned
