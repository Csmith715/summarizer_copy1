import openai
import logging
import concurrent.futures
import re
from config import url_string, emoji_pattern, OPENAI_API_KEY

logger = logging.getLogger()
openai.api_key = OPENAI_API_KEY


class SocialGenerations:
    def __init__(
            self,
            n_questions: list,
            snippets: list,
            job_title: str,
            introduction: str,
            promotion_type: str,
            seq2seq_model
    ):
        self.chunked_snippets = list(divide_chunks(snippets, 5))
        self.non_questions = n_questions
        self.snippets = snippets
        self.title = job_title
        self.summary = introduction
        self.promotion = promotion_type
        self.input_prompts = []
        self.result_dict = {
            "davinci:ft-contentware:esl-generation-2023-04-21-16-37-03": [],            # 'email subject lines'
            "davinci:ft-contentware:instagram-generation-v2-2023-04-17-01-40-04": [],   # 'Instagram'
            "summarizer": []    # Summarizer
        }
        self.question_model = seq2seq_model

    def create_socials(self):
        self.make_input_prompts()
        self.make_gpt()
        return self.result_dict

    def make_input_prompts(self):
        form1 = f"This is content for an upcoming {self.promotion}:\n\nTitle: {self.title}\nSummary: {self.summary}\nObjectives:\n\n"
        igram_suffix = f'\n\nCreate a varied series of short Instagram posts that promotes this {self.promotion}.'
        esl_suffix = f'\n\nCreate a varied series of email subject lines that promotes this {self.promotion}. Number each subject line.'
        unfocused_bullets = []
        for chunk in self.chunked_snippets:
            unfocused_blist = [f'- {bul}' for bul in chunk]
            unfocused_bullets.append('\n'.join(unfocused_blist))
        # random_ufb = random.choice(unfocused_bullets)
        for ufb in unfocused_bullets:
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
        self.input_prompts.append(
            (
                None,
                "summarizer",
                None,
                None,
            )
        )

    def make_gpt(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.generate_text, prompt, model, mt, nval) for prompt, model, mt, nval in self.input_prompts]
        results = [future.result() for future in futures]
        for res in results:
            model = res['model']
            if model == 'summarizer':
                self.result_dict['summarizer'] = res['result']
            else:
                texts = [c['text'] for c in res.choices]
                clean_texts = clean_gpt_list(texts, model)
                self.result_dict[model].extend(clean_texts)

    def question_generator(self) -> dict:
        logger.info('Generating Questions')
        try:
            questions = self.question_model.predict(self.non_questions)
            logger.info('Questions Completed')
        except Exception as e:
            logger.info(e)
            questions = []
        return {
            'model': 'summarizer',
            'result': questions
        }

    def generate_text(self, prompt, model, max_tokes, n_value):
        if model == 'summarizer':
            response = self.question_generator()
        else:
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=max_tokes,
                frequency_penalty=0.25,
                presence_penalty=0.25,
                n=n_value
            )
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
    clean_text = clean_text.replace('  ', ' ')
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
    cleaned = list(set(cleaned))
    return cleaned

def remove_url(social_post: str) -> str:
    url_free_string = re.sub(url_string, '', social_post, )
    return url_free_string

def drop_last(esl_post: str) -> str:
    sesl = esl_post.split('\n')
    fixed_esl = '\n'.join(sesl[:-1])
    return fixed_esl

def generate_chat_text(user_text, system_text='You are a helpful assistant'):
    prompt_message = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text}
    ]
    try:
        chat_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=prompt_message,
            n=1
        )
        final_response = chat_response.choices[0]['message']['content']
    except Exception as e:
        final_response = f"ChatGPT isn't working right now. Here is the error they sent us:\n{e}"
    return final_response
