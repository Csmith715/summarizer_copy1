import flask
import transformers
# from transformers import pipeline, BartTokenizerFast
import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import pickle
from itertools import product
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs
import torch
import boto3
from concurrent.futures import ThreadPoolExecutor

import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 


cta_root_path = os.getenv('CTA_ROOT_PATH', '/tmp')
qg_root_path = os.getenv('QG_ROOT_PATH', '/tmp')
bucket_name = os.getenv('BUCKET_NAME', 'contentware-nlp')
s3 = boto3.client('s3')

def download_s3_folder(bucket_name, s3_folder):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    local_dir = f'{qg_root_path}/{qg_path}'
    # if not os.path.exists(local_dir):
    #     os.makedirs(target_dir, exist_ok=True)

    #bucket = s3.Bucket(bucket_name)
    #bucket = bucket_name
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)

# def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
#     s3_resource = boto3.resource('s3')
#     bucket = s3_resource.Bucket(bucketName) 
#     for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
#         if not os.path.exists(os.path.dirname(obj.key)):
#             os.makedirs(os.path.dirname(obj.key))
#         bucket.download_file(obj.key, obj.key) # save to same path


def download_file(path, file):
    target_dir = f'{cta_root_path}/{path}'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    return s3.download_file(bucket_name, f'{path}/{file}', f'{target_dir}/{file}')


cta_path = os.getenv('CTA_PATH', 'CTA_Bullets')
qg_path = os.getenv('QG_PATH', 'question-generation')

download_s3_folder(bucket_name, 'question-generation')
# print(torch.cuda.is_available())
# seq_model = Seq2SeqModel(
#     encoder_decoder_type="bart",
#     encoder_decoder_name=os.path.join(qg_root_path, qg_path),
#     use_cuda=torch.cuda.is_available()
# )

application = flask.Flask(__name__)
# @application.before_first_request
# def before_first_request():
#     load_summarizer()

# summarizer = None
# tokenizer = None

# def load_summarizer():
#     # global summarizer
#     if summarizer and tokenizer:
#         return
#     else:
#         summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', tokenizer='sshleifer/distilbart-cnn-12-6')
#         tokenizer = BartTokenizerFast.from_pretrained('sshleifer/distilbart-cnn-12-6')

def purge_extra(rule_list):
    outlist = []
    for phrase in rule_list:
        nlist = [j.lower() for j in phrase.split(' ') if j not in stop_words]
        if len(nlist) == len(set(nlist)):
            outlist.append(phrase)
    return outlist

def get_max_token_length(text, character_length):
    sum_len = 0
    num_tokens = 0
    tokens = tokenizer.tokenize(text)
    for i,t in enumerate(tokens):
        if sum_len < character_length:
            num_tokens = i
            sum_len = sum_len + len(t)
        else:
            break
    return num_tokens

# model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def replace_tokens(dod, event_type, model):
    embedding_dict = {}
    for d in dod:
        subbed_phraseology_list = []
        for x in dod[d]['name']:
            subbed_text = x.replace('{%Token%}', event_type.capitalize())
            subbed_text = subbed_text.replace('{%token%}', event_type.lower())
            subbed_text = subbed_text.replace('{%TOKEN%}', event_type)
            subbed_phraseology_list.append(subbed_text)
        
        embedding = [(t, model.encode(t, convert_to_tensor=False)) for t in subbed_phraseology_list]
        embedding_dict[d] = embedding
    return embedding_dict

def create_questions(text):
    gq = seq_model.predict([text])
    return gq
#     gq = fix_caps(text, gq[0])

@application.route('/summarizer', methods=['POST'])
def summarizer():
    data = {}

    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()

    intext = req_data['text']
    max_character_length = req_data['max length']
    maxlen = get_max_token_length(intext, int(max_character_length))

    # sumtext = summarizer(intext, min_length=10, max_length=maxlen, clean_up_tokenization_spaces = True)
    # data['summarized text'] = sumtext[0]['summary_text']
    data['summarized text'] = intext

    return flask.jsonify(data)

@application.route('/summarizer/generatequestions', methods=['POST'])
def generatequestions():
    #download_s3_folder(bucket_name, 'question-generation')
    #downloadDirectoryFroms3(bucket_name, 'question-generation')
    data = {}

    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()

    inputtext = req_data['text']

    seq_model = Seq2SeqModel(
        encoder_decoder_type="bart",
        encoder_decoder_name=os.path.join(qg_root_path, qg_path),
        use_cuda=torch.cuda.is_available()
    )

    #data['generated question'] = seq_model.predict([inputtext])
    data['generated question'] = seq_model.predict(inputtext)

    return flask.jsonify(data)

@application.route('/summarizer/generatequestions_test', methods=['POST'])
def generatequestions_test():
    data = {}

    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()

    inputbullets = req_data['text']
    with ThreadPoolExecutor() as exe:
        exe.submit(create_questions)
        result = exe.map(create_questions,inputbullets)

    data['generated questions'] = list(result)

    return flask.jsonify(data)


# for b in bstrings:
#     if b[-1] != '?':
#         post_data = {"text": str(b)}
#         r = requests.post('https://gateway.dev.contentware.com/ai/summarizer/generatequestions', json = post_data)
#         gq = r.json()['generated question']
#         gq = fix_caps(b, gq[0])
#         gquestions.append(gq if gq else '')
#     else:
#         gquestions.append('')

@application.route('/summarizer/updateCTA', methods=['GET'])
def updateCTA():
    #s3 = boto3.resource('s3')
    #s3.Bucket(BUCKET_NAME).download_file(KEY, 'campaign-metadata.json')
    download_file(cta_path, 'campaign-metadata.json')

    # Load JSON CTA/Wrapper Info
    
    #with open('campaign-metadata.json') as f:
    with open(os.path.join(cta_root_path, cta_path, 'campaign-metadata.json')) as f:
        rules = json.load(f)

    # Create a dictionary of dataframes for each CTA
    dict_of_dfs = dict()
    for x in rules['ctas']:
        temp_df = pd.DataFrame(x['phrases'])
        if not temp_df.empty:
            temp_df = temp_df.drop(['ctaId'], axis=1)
            dict_of_dfs[x['categoryName']] = temp_df

    # Create a dictonary of bullet point wrapper rules
    new_rule_dict = {}
    for r in rules['bulletPoints']:
        bp = r['beforePositions']
        full_list = list(product(*bp))
        joined_list = [' '.join(f)+' ' for f in full_list]
        final_list = purge_extra(joined_list)
        new_rule_dict[r['name']] = final_list

    new_rule_dict['HWW_Rules'] = new_rule_dict['How, What, Why, When, Which, Where Rule #1'] + new_rule_dict['How, What, Why, When, Which, Where Rule #2'] + new_rule_dict['How, What, Why, When, Which, Where Rule #3']
    new_rule_dict['Noun_Rules'] = new_rule_dict['Noun Rule #1'] + new_rule_dict['Noun Rule #2']
    del new_rule_dict['How, What, Why, When, Which, Where Rule #1']
    del new_rule_dict['How, What, Why, When, Which, Where Rule #2']
    del new_rule_dict['How, What, Why, When, Which, Where Rule #3']
    del new_rule_dict['Noun Rule #1']
    del new_rule_dict['Noun Rule #2']

    # Create embedding dictionary for CTA's

    # embedding_dict = {}
    # for d in dict_of_dfs:
    #     embedding = [(t, model.encode(t, convert_to_tensor=False)) for t in dict_of_dfs[d]['name']]
    #     embedding_dict[d] = embedding
    ptl = ['WEBINAR','EVENT','ONLINE_EVENT','VIRTUAL_EVENT','ONLINE_SESSION','CONFERENCE','SEMINAR','LECTURE']
    sbert_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    event_dict = {}
    for p in ptl:
        ed = replace_tokens(dict_of_dfs, p.replace('_',' '), sbert_model)
        event_dict[p] = ed   

    # with open(os.path.join(cta_root_path, cta_path, 'phraseology_embeddings.pkl'), 'wb') as fp:
    #     pickle.dump(embedding_dict, fp)
    with open(os.path.join(cta_root_path, cta_path, 'phraseology_embeddings.pkl'), 'wb') as fp:
        pickle.dump(event_dict, fp)

    with open(os.path.join(cta_root_path, cta_path, 'bullet_rules.json'), 'w') as fp:
        json.dump(new_rule_dict, fp)

    s3.upload_file(os.path.join(cta_root_path, cta_path, 'phraseology_embeddings.pkl'), bucket_name, 'CTA_Bullets/phraseology_embeddings.pkl')
    s3.upload_file(os.path.join(cta_root_path, cta_path, 'bullet_rules.json'), bucket_name, 'CTA_Bullets/bullet_rules.json')
    return flask.Response(response='done', status=200, mimetype='text/plain')


@application.route('/healthz', methods=['GET'])
def healthz():
    return flask.Response(response='ok', status=200, mimetype='text/plain')


if __name__ == "__main__":
    port = os.getenv('FLASK_PORT', 5000)
    host = os.getenv('FLASK_HOST', None)
    debug = not os.getenv('LIVE', False)
    # load_summarizer()
    application.run(host=host, port=port, debug=debug)
