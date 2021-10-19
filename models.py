from simpletransformers.seq2seq import Seq2SeqModel
from utils import download_s3_folder, download_file, s3
import torch
from nltk.corpus import stopwords
import os
from itertools import product
import json
import pickle
import pandas as pd
import logging

bucket_name = os.getenv('BUCKET_NAME', 'contentware-nlp')
logger = logging.getLogger()
download_s3_folder(bucket_name, 'question-generation')

def replace_tokens(dod, event_type):
    embedding_dict = {}
    for d in dod:
        subbed_phraseology_list = []
        for x in dod[d]['name']:
            subbed_text = x.replace('{%Token%}', event_type.capitalize())
            subbed_text = subbed_text.replace('{%token%}', event_type.lower())
            subbed_text = subbed_text.replace('{%TOKEN%}', event_type)
            subbed_phraseology_list.append(subbed_text)
        embedding = [(t, []) for t in subbed_phraseology_list]
        embedding_dict[d] = embedding
    return embedding_dict

class ModelFuncs:
    def __init__(self, model_dir):
        self.dir = model_dir
        self.model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name=self.dir,
            use_cuda=torch.cuda.is_available()
        )
        logger.info('Seq2Seq Model Loaded')
        self.stop_words = set(stopwords.words('english'))

    def create_questions(self, in_text):
        gqs = self.model.predict(in_text)
        logger.info('Questions Created')
        return gqs

    def purge_extra(self, rule_list):
        outlist = []
        for phrase in rule_list:
            nlist = [j.lower() for j in phrase.split(' ') if j not in self.stop_words]
            if len(nlist) == len(set(nlist)):
                outlist.append(phrase)
        return outlist

    def CongigureCTA(self, cta_path,cta_root_path, bname):
        download_file(cta_path, 'campaign-metadata.json', bname)
        # Load JSON CTA/Wrapper Info
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
            joined_list = [' '.join(f) + ' ' for f in full_list]
            final_list = self.purge_extra(joined_list)
            new_rule_dict[r['name']] = final_list

        new_rule_dict['HWW_Rules'] = new_rule_dict['How, What, Why, When, Which, Where Rule #1'] + new_rule_dict[
            'How, What, Why, When, Which, Where Rule #2'] + new_rule_dict['How, What, Why, When, Which, Where Rule #3']
        new_rule_dict['Noun_Rules'] = new_rule_dict['Noun Rule #1'] + new_rule_dict['Noun Rule #2']
        del new_rule_dict['How, What, Why, When, Which, Where Rule #1']
        del new_rule_dict['How, What, Why, When, Which, Where Rule #2']
        del new_rule_dict['How, What, Why, When, Which, Where Rule #3']
        del new_rule_dict['Noun Rule #1']
        del new_rule_dict['Noun Rule #2']

        ptl = ['WEBINAR', 'EVENT', 'ONLINE_EVENT', 'VIRTUAL_EVENT', 'ONLINE_SESSION', 'CONFERENCE', 'SEMINAR',
               'LECTURE']
        event_dict = {}
        for p in ptl:
            ed = replace_tokens(dict_of_dfs, p.replace('_', ' '))
            event_dict[p] = ed
        with open(os.path.join(cta_root_path, cta_path, 'phraseology_embeddings.pkl'), 'wb') as fp:
            pickle.dump(event_dict, fp)
        with open(os.path.join(cta_root_path, cta_path, 'bullet_rules.json'), 'w') as fp:
            json.dump(new_rule_dict, fp)

        s3.upload_file(os.path.join(cta_root_path, cta_path, 'phraseology_embeddings.pkl'), bucket_name,
                       'CTA_Bullets/phraseology_embeddings.pkl')
        s3.upload_file(os.path.join(cta_root_path, cta_path, 'bullet_rules.json'), bucket_name,
                       'CTA_Bullets/bullet_rules.json')






