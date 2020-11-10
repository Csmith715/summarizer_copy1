import flask
from transformers import pipeline
import os
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import pickle

import boto3
import botocore

BUCKET_NAME = 'contentware-nlp' 
KEY = 'CTA_Bullets/campaign-metadata.json' 

application = flask.Flask(__name__)

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
def updateCTA:
    s3 = boto3.resource('s3')
    s3.Bucket(BUCKET_NAME).download_file(KEY, 'campaign-metadata.json')

    # Load JSON CTA/Wrapper Info
    with open('campaign-metadata.json') as f:
        rules = json.load(f)

    # Create a dictionary of dataframes for each CTA
    dict_of_dfs = dict()
    for x in rules['ctas']:
        temp_df = pd.DataFrame(x['phrases'])
        if not temp_df.empty:
            temp_df = temp_df.drop(['ctaId'], axis=1)
            dict_of_dfs[x['categoryName']] = temp_df

    # Create embedding dictionary for CTA's
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    embedding_dict = {}
    for d in dict_of_dfs:
        embedding = [(t, model.encode(t, convert_to_tensor=False)) for t in dict_of_dfs[d]['name']]
        embedding_dict[d] = embedding

    with open('phraseology_embeddings.pkl', 'wb') as fp:
        pickle.dump(embedding_dict, fp)

    s3.upload_file('phraseology_embeddings.pkl', BUCKET_NAME, 'CTA_Bullets/phraseology_embeddings.pkl')
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
