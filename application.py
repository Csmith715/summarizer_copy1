import flask
from flask import render_template, request
import os
from models import ModelFuncs, bucket_name
from utils import cta_path, cta_root_path, qg_path, qg_root_path
import logging
from simpletransformers.seq2seq import Seq2SeqModel
import torch
from logging.config import fileConfig
import blog
from blog import MultilineGenerations
from src.gpt3_generations import GPT3Creations

fileConfig('logging.conf')
logger = logging.getLogger('root')

def page_not_found(e):
    return render_template('404.html'), 404


application = flask.Flask(__name__)

application.register_error_handler(404, page_not_found)
model = None
def load_model():
    global model
    if not model:
        try:
            model = Seq2SeqModel(
                encoder_decoder_type="bart",
                encoder_decoder_name=os.path.join(qg_root_path, qg_path),
                use_cuda=torch.cuda.is_available()
            )
        except Exception as e:
            # for local testing
            print(e)
            model = Seq2SeqModel(
                encoder_decoder_type="bart",
                encoder_decoder_name='/Users/micksmith/Contentware_Local_Models/question_generation',
                use_cuda=torch.cuda.is_available()
            )
        logger.info('Model Loaded')

@application.route('/summarizer', methods=['POST'])
def summarizer():
    data = {}
    req_data = None
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
    logger.info('Generating Questions')
    data['generated question'] = model.predict(inputtext)
    logger.info('Questions Created')

    return flask.jsonify(data)

@application.route('/summarizer/updateCTA', methods=['GET'])
def updatecta():
    mfuncs = ModelFuncs(cta_path, cta_root_path, bucket_name)
    mfuncs.CongigureCTA()
    return flask.Response(response='done', status=200, mimetype='text/plain')

@application.route('/summarizer/generate_blogs', methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        if 'form1' in request.form:
            topic = request.form['content']
            keywords = request.form['blogKeywords']
            prompt = f'Write a long detailed blog about:\n{topic}\nKeywords:\n{keywords}\n\n\n'
            blog_text = blog.write_blog(prompt)
            written_blog = blog_text.replace('\n', '<br>')

        if 'form2' in request.form:
            prompt = request.form['gcontent']
            blog_text = ''
            if request.form['submit_button'] == 'continue':
                blog_text = blog.continue_blog(prompt)
            elif request.form['submit_button'] == 'conclude':
                blog_text = blog.conclude_blog(prompt)
            blog_text = blog_text.strip('\n').strip()
            blog_text = f'{prompt}\n\n{blog_text}'

    return render_template('index.html', **locals())

@application.route('/summarizer/generate_email_subject_lines', methods=["GET", "POST"])
def index2():
    if request.method == 'POST':
        if 'form1' in request.form:
            email_body = request.form['email_content']
            number_topics = request.form['eslTopicNumber']
            if number_topics == '' or number_topics == 0:
                number_topics = 5
            else:
                number_topics = int(number_topics)
            email_text = MultilineGenerations().write_email_subject_lines(email_body, number_topics)
            email_text = f'1. {email_text}'
            written_subjectlines = email_text.replace('\n', '<br>')
    return render_template('index2.html', **locals())

@application.route('/summarizer/generate_sms_campaigns', methods=["GET", "POST"])
def index3():
    if request.method == 'POST':
        if 'form1' in request.form:
            sms_body = request.form['sms_content']
            sms_text = blog.write_sms_campaigns(sms_body)
            written_sms = sms_text.replace('\n', '<br>')
    return render_template('index3.html', **locals())

@application.route('/summarizer/generate_blog_topics', methods=["GET", "POST"])
def index4():
    if request.method == 'POST':
        if 'form1' in request.form:
            topic = request.form['blog_topic_content']
            keywords = request.form['blogTopicKeywords']
            number_topics = request.form['blogTopicNumber']
            if number_topics == '' or number_topics == 0:
                number_topics = 5
            else:
                number_topics = int(number_topics)
            topic_text = MultilineGenerations().generate_blog_topics(topic, keywords, number_topics)
            written_topics = topic_text.replace('\n', '<br>')
    return render_template('index4.html', **locals())

@application.route('/gpt3_creations/generate_blogs', methods=["POST"])
def generate_blogs():
    data = {}
    req_data = None
    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()
    g_type = req_data['type']
    topic = req_data['description']
    keywords = req_data['keywords']
    data['content'], data['tokens'] = GPT3Creations().select_and_write(g_type, topic, keywords)
    logger.info('GPT3 Content Created')

    return flask.jsonify(data)

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
