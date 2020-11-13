import flask
from transformers import pipeline
import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import pickle
from itertools import product

import boto3
import botocore

import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english')) 

#BUCKET_NAME = 'contentware-nlp' 
#KEY = 'CTA_Bullets/campaign-metadata.json' 

application = flask.Flask(__name__)


cta_root_path = os.getenv('CTA_ROOT_PATH', '/tmp')
bucket_name = os.getenv('BUCKET_NAME', 'contentware-nlp')
s3 = boto3.client('s3')

cta_path = os.getenv('CTA_PATH', 'CTA_Bullets')

def download_file(path, file):
    target_dir = f'{cta_root_path}/{path}'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    return s3.download_file(bucket_name, f'{path}/{file}', f'{target_dir}/{file}')

def purge_extra(rule_list):
    outlist = []
    for phrase in rule_list:
        nlist = [j.lower() for j in phrase.split(' ') if j not in stop_words]
        if len(nlist) == len(set(nlist)):
            outlist.append(phrase)
    return outlist

def load_summarizer():
    global summarizer
    summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6', tokenizer='sshleifer/distilbart-cnn-12-6')

@application.route('/summarizer', methods=['POST'])
def summarizer():
    data = {}

    if flask.request.content_type == 'application/json':
        req_data= flask.request.get_json()

    intext = req_data['text']
    maxlen = req_data['max length']

    sumtext = summarizer(intext, min_length=5, max_length=maxlen, clean_up_tokenization_spaces = True)
    data['summarized text'] = sumtext[0]['summary_text']

    return flask.jsonify(data)

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

    new_rule_dict['HWW_Rules'] = new_rule_dict['How, What, Why Rule #1'] + new_rule_dict['How, What, Why Rule #2'] + new_rule_dict['How, What, Why Rule #3']
    new_rule_dict['Noun_Rules'] = new_rule_dict['Noun Rule #1'] + new_rule_dict['Noun Rule #2']
    del new_rule_dict['How, What, Why Rule #1']
    del new_rule_dict['How, What, Why Rule #2']
    del new_rule_dict['How, What, Why Rule #3']
    del new_rule_dict['Noun Rule #1']
    del new_rule_dict['Noun Rule #2']

    # Create embedding dictionary for CTA's
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    embedding_dict = {}
    for d in dict_of_dfs:
        embedding = [(t, model.encode(t, convert_to_tensor=False)) for t in dict_of_dfs[d]['name']]
        embedding_dict[d] = embedding

    with open(os.path.join(cta_root_path, cta_path, 'phraseology_embeddings.pkl'), 'wb') as fp:
        pickle.dump(embedding_dict, fp)

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
    load_summarizer()
    application.run(host=host, port=port, debug=debug)
