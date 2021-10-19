import flask
import os
from models import ModelFuncs, bucket_name
from utils import cta_path, cta_root_path
import logging
from logging.config import fileConfig
fileConfig('logging.conf')
logger = logging.getLogger('root')

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
@application.route('/summarizer', methods=['POST'])
def summarizer():
    data = {}
    req_data=None
    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()
    intext = req_data['text']
    data['summarized text'] = intext
    return flask.jsonify(data)

@application.route('/summarizer/generatequestions', methods=['POST'])
def generatequestions():
    data = {}
    req_data = None
    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()
    inputtext = req_data['text']
    logger.info('Generating Quesitons')
    data['generated question'] = ModelFuncs.create_questions(inputtext)

    return flask.jsonify(data)

@application.route('/summarizer/updateCTA', methods=['GET'])
def updatecta():
    ModelFuncs(post_data=None).CongigureCTA(cta_path, cta_root_path, bucket_name)
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
