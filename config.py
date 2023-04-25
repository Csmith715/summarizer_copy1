import openai
import re

# OPEN API STUFF
OPENAI_API_KEY = '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'

url_string = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
emoji_pattern = re.compile("["
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                           u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                           u"\U00002702-\U000027B0"  # Dingbats
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

# FLASK STUFF
class Config(object):
    DEBUG = True
    TESTING = False

def email_cta_creations(input_prompt: str):
    response = openai.Completion.create(
        model="curie:ft-contentware-2023-03-21-11-24-35",
        prompt=input_prompt,
        temperature=0.7,
        max_tokens=30,
        top_p=1,
        frequency_penalty=0.25,
        presence_penalty=0.25,
        n=40,
        stop=['\n']
    )
    raw_array = [res['text'].strip('\n') for res in response['choices']]

    return raw_array

# form2 = f"This is content for an upcoming {self.promotion}:\n\nTitle: {self.title}\nSummary: {self.summary}\nObjectives:\n"
# prompt = f"{form2}{snippets}\n\nWrite a Call to Action sentence for an email communication about this {self.promotion} that will encourage a response.\n\n"
