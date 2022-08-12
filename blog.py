import openai
import config

openai.api_key = config.OPENAI_API_KEY

def write_blog(prompt1):
    response = openai.Completion.create(
      engine="davinci-instruct-beta-v3",
      prompt=prompt1,
      temperature=0.7,
      max_tokens=250,
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

def write_email_subject_lines(email_body):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Create a list of email subject lines for the following email:\n\n{email_body}\n\n1.",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response['choices'][0]['text']

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
