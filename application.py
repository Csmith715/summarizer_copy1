import flask
from transformers import pipeline
import os


application = flask.Flask(__name__)

def load_summarizer():
    global summarizer
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn', tokenizer='facebook/bart-large-cnn')

@application.route('/summarizer', methods=['POST'])
def summarizer():
    data = {}

    if flask.request.content_type == 'application/json':
        req_data= flask.request.get_json()

    intext = req_data['text']
    maxlen = req_data['max length']

    sumtext = summarizer(intext, min_length=5, max_length=maxlen)
    data['summarized text'] = sumtext[0]['summary_text']

    return flask.jsonify(data)

@application.route('/healthz', methods=['GET'])
def healthz():
    return flask.Response(response='ok', status=200, mimetype='text/plain')


if __name__ == "__main__":
    port = os.getenv('FLASK_PORT', 5000)
    host = os.getenv('FLASK_HOST', None)
    debug = not os.getenv('LIVE', False)
    load_summarizer()
    application.run(host=host, port=port, debug=debug)
