import flask
import os
from models import ModelFuncs, bucket_name
from utils import cta_path, cta_root_path, qg_path, qg_root_path
import logging
from simpletransformers.seq2seq import Seq2SeqModel
import torch
from logging.config import fileConfig
fileConfig('logging.conf')
logger = logging.getLogger('root')

application = flask.Flask(__name__)

model = None
def load_model():
    global model
    if not model:
        model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name=os.path.join(qg_root_path, qg_path),
            use_cuda=torch.cuda.is_available()
        )

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
    data['generated question'] = model.predict(inputtext)
    logger.info('Questions Created')

    return flask.jsonify(data)

@application.route('/summarizer/updateCTA', methods=['GET'])
def updatecta():
    mfuncs = ModelFuncs(cta_path, cta_root_path, bucket_name)
    mfuncs.CongigureCTA()
    return flask.Response(response='done', status=200, mimetype='text/plain')

@application.route('/healthz', methods=['GET'])
def healthz():
    return flask.Response(response='ok', status=200, mimetype='text/plain')

if __name__ == "__main__":
    port = os.getenv('FLASK_PORT', 5000)
    host = os.getenv('FLASK_HOST', None)
    debug = not os.getenv('LIVE', False)
    load_model()
    logger.info('Seq2Seq Model Loaded')
    application.run(host=host, port=port, debug=debug)
