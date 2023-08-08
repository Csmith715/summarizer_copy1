import flask
from flask import render_template, request
import os
from models import ModelFuncs, bucket_name
from utils import cta_path, cta_root_path  # qg_path, qg_root_path
import logging
# from simpletransformers.seq2seq import Seq2SeqModel
# import torch
from logging.config import fileConfig
import blog
from blog import MultilineGenerations
from src.gpt3_generations import GPT3Creations, NewGPT3Content
from src.openai_creations import SocialGenerations, generate_chat_text, SocialContentCreation

fileConfig('logging.conf')
logger = logging.getLogger('root')

def page_not_found():
    return render_template('404.html'), 404


application = flask.Flask(__name__)

application.register_error_handler(404, page_not_found)
# question_model = None
# def load_model():
#     global question_model
#     if not question_model:
#         try:
#             logger.info("Loading Seq2Seq model")
#             question_model = Seq2SeqModel(
#                 encoder_decoder_type="bart",
#                 encoder_decoder_name=os.path.join(qg_root_path, qg_path),
#                 use_cuda=torch.cuda.is_available()
#             )
#         except Exception as e:
#             # for local testing
#             print(e)
#             question_model = Seq2SeqModel(
#                 encoder_decoder_type="bart",
#                 encoder_decoder_name='/Users/micksmith/Contentware_Local_Models/question_generation',
#                 use_cuda=torch.cuda.is_available()
#             )
#         logger.info('Question Model Loaded')

@application.route('/summarizer', methods=['POST'])
def summarizer():
    data = {}
    req_data = None
    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()
    intext = req_data['text']
    data['summarized text'] = intext
    return flask.jsonify(data)

@application.route('/summarizer/social-creations', methods=['POST'])
def social_creations():
    # data = {}
    req_data = None
    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()
    form_data = req_data.get('form_data', {})
    title = form_data.get('socialPostTitle', '')
    description = form_data.get('socialPostDescription', '')
    keywords = form_data.get('keywords', '')
    containers = req_data.get('containers', [])
    logger.info('Creating Social Media content')
    social_content = SocialContentCreation(title, description, keywords, containers)
    # social_posts = social_content.make_social_creations()
    output = social_content.make_social_creations()
    logger.info('Social Content Created')

    return flask.jsonify(output)

@application.route('/summarizer/question_gpt_creations', methods=['POST'])
def question_gpt_creations():
    """
    Parameters: {
            snippets (list): An array of snippets for question, email subject line, and Instagram generation
            title (str): Title of the job passed
            introduction (str): The introduction paragraph passed
            promotion_type (str): The event label or type
            action_verb (str): A specific campaign action that is passed to guide the text generations
            promo_val (str): Promotion label string. Used to filter ads on/off
            }
    Returns: {
        # generated_questions (list): An array of questions created using a Simple Transformer model,
        email_subject_lines (list): An array of email subject lines
        instagram_posts (list): An array of generated Instagram posts
        }
    """
    data = {}
    req_data = None
    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()
    snips = req_data.get('snippets', [])
    # non_questions = req_data.get('non_questions', [])
    title = req_data.get('title', '')
    intro = req_data.get('introduction', '')
    promo = req_data.get('promotion_type', '')
    action_verb = req_data.get('action_verb', '')
    promo_val = req_data.get('promo_val', '')
    logger.info('Creating OpenAI content')
    so_gen = SocialGenerations(snips, title, intro, promo, action_verb, promo_val)
    sog_results = so_gen.create_socials()
    # data['generated_question'] = sog_results['summarizer']
    data['email_subject_lines'] = sog_results['davinci:ft-contentware:esl-generation-2023-04-21-16-37-03']
    data['instagram_posts'] = sog_results['davinci:ft-contentware:instagram-generation-v2-2023-04-17-01-40-04']
    data['facebook_ads'] = sog_results['gpt-4-fb']
    data['linkedin_ads'] = sog_results['gpt-4-li']
    data['email_headlines'] = sog_results['gpt-4-eh']
    data['cta_buttons'] = sog_results['gpt-4-buttons']
    data['shortcta'] = sog_results['davinci:ft-contentware:email-cta-v2-2023-05-04-23-04-53']
    logger.info('Buttons, ESL, Instagram Posts, & Headlines Created')

    return flask.jsonify(data)


# @application.route('/summarizer/generatequestions', methods=['POST'])
# def generatequestions():
#     """
#     Parameters: {
#             text (list): An array of snippets for question generation,
#             email_cta_prompt (str): A string to be passed to an OpenAI call
#             }
#     Returns: {
#         generated question (list): An array of questions created using a Simple Transformer model,
#         email_ctas (list): An array of email call to action texts
#         }
#     """
#     data = {}
#     req_data = None
#     if flask.request.content_type == 'application/json':
#         req_data = flask.request.get_json()
#     inputtext = req_data.get('text', '')
#     logger.info('Generating Questions')
#     data['generated question'] = question_model.predict(inputtext)
#     logger.info('Questions Created')
#
#     return flask.jsonify(data)

@application.route('/summarizer/updateCTA', methods=['GET'])
def updatecta():
    mfuncs = ModelFuncs(cta_path, cta_root_path, bucket_name)
    mfuncs.configure_cta()
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

@application.route('/summarizer/gpt3_creations/generate_blogs', methods=["POST"])
def generate_blogs():
    data = {}
    req_data = None
    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()
    g_type = req_data.get('type', '')
    topic = req_data.get('description', '')
    keywords = req_data.get('keywords', '')
    event_type = req_data.get('channel', '')
    question_text = req_data.get('question', '')
    logger.info('Creating GPT3/GPT4 content')
    if question_text:
        data['content'] = generate_chat_text(question_text)
    else:
        data['content'], data['tokens'] = GPT3Creations().select_and_write(g_type, topic, keywords, event_type)
    logger.info('GPT3 Content Created')

    return flask.jsonify(data)

@application.route('/summarizer/social_media_posts', methods=["POST"])
def create_social_media():
    data = {}
    req_data = None
    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()
    if req_data:
        new_gpt3 = NewGPT3Content(req_data)
        data['content'] = new_gpt3.generate_social_media_content()
    return flask.jsonify(data)

@application.route('/summarizer/chat', methods=["POST"])
def create_chat_response():
    data = {}
    req_data = None
    if flask.request.content_type == 'application/json':
        req_data = flask.request.get_json()
    if req_data:
        user_text = req_data.get('user_question', '')
        data['content'], _ = generate_chat_text(user_text)
    return flask.jsonify(data)

@application.route('/healthz', methods=['GET'])
def healthz():
    return flask.Response(response='ok', status=200, mimetype='text/plain')


if __name__ == "__main__":
    port = os.getenv('FLASK_PORT', 5000)
    host = os.getenv('FLASK_HOST', None)
    debug = not os.getenv('LIVE', False)
    # load_model()
    # logger.info('Seq2Seq Model Loaded')
    application.run(host=host, port=port, debug=debug)
