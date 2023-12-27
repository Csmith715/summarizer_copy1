
from utils import download_s3_folder, download_file, s3
from nltk.corpus import stopwords
import os
from itertools import product
import json
import pickle
import logging

bucket_name = os.getenv('BUCKET_NAME', 'contentware-nlp')
logger = logging.getLogger()
download_s3_folder(bucket_name, 'question-generation')

rule_cats = [
    'CTA_Take Action (Long)',
    'CTA _Question',
    'CTA_Urgent (Long)',
    'CTA_Day of Reminder (Long)',
    'CTA_Day Before Reminder (Long)',
    'CTA_Day Before Reminder (Short)',
    'CTA_Day of Reminder (Short)',
    'CTA_Take Action (Short)',
    'CTA_Urgent (Short)',
    'Single Bullet Question Follow Up',
    'CTA_Take Action Question Follow Up',
    'CTA Take Action  Long (Social Card)',
    'CTA Drive Urgency Long (Social Card)',
    'CTA Day Before Reminder Long  (Social Card)',
    'CTA Day of  Reminder Long (Social Card)'
]

def replace_tokens(dod: dict, event_type: str) -> dict:
    embedding_dict = {}
    for d in dod:
        if event_type == 'FREEFORM':
            subbed_phraseology_list = dod[d]['name']
        else:
            subbed_phraseology_list = []
            for x in dod[d]['name']:
                subbed_text = x.replace('{%Token%}', event_type.capitalize())
                subbed_text = subbed_text.replace('{%token%}', event_type.lower())
                subbed_text = subbed_text.replace('{%TOKEN%}', event_type)
                subbed_phraseology_list.append(subbed_text)
        embedding = [(t, []) for t in subbed_phraseology_list]
        embedding_dict[d] = embedding
    return embedding_dict

def replace_rule_tokens(base_rule_dict: dict, event_key: str) -> dict:
    new_rule_dict = {}
    for k, v in base_rule_dict.items():
        replaced_rule_values = []
        if not v:
            new_rule_dict[k] = []
        else:
            for value in v:
                subbed_text = value.replace('{%Token%}', event_key.capitalize())
                subbed_text = subbed_text.replace('{%token%}', event_key.lower())
                subbed_text = subbed_text.replace('{%TOKEN%}', event_key)
                replaced_rule_values.append(subbed_text)
            new_rule_dict[k] = replaced_rule_values
    return new_rule_dict

class ModelFuncs:
    def __init__(self, cta_path, cta_root_path, bname):
        self.cpath = cta_path
        self.crpath = cta_root_path
        self.buck_name = bname
        self.stop_words = set(stopwords.words('english'))

    def purge_extra(self, rule_list):
        outlist = []
        for phrase in rule_list:
            nlist = [j.lower() for j in phrase.split(' ') if j not in self.stop_words]
            if len(nlist) == len(set(nlist)):
                outlist.append(phrase)
        return outlist

    def make_rule_dict(self, promotion_dict: dict) -> dict:
        new_rule_dict = {}
        for r in promotion_dict:
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
        return new_rule_dict

    def configure_cta(self):
        download_file(self.cpath, 'campaign-metadata.json', self.buck_name)
        # Load JSON CTA/Wrapper Info
        with open(os.path.join(self.crpath, self.cpath, 'campaign-metadata.json')) as f:
            rules = json.load(f)
        new_ph_dict = {}
        for r in rules['ctas']:
            for k, v in r['phrases'].items():
                new_ph_dict[k] = {}
        for r in rules['ctas']:
            for k, v in r['phrases'].items():
                new_ph_dict[k][r['categoryName']] = [(val['name'], []) for val in v]

        # Create a dictionary of bullet point wrapper rules
        categorized_rule_dict = {}
        for k, v in rules['bulletPoints'].items():
            categorized_rule_dict[k] = self.make_rule_dict(v)

        with open(os.path.join(self.crpath, self.cpath, 'phraseology_embeddings.pkl'), 'wb') as fp:
            pickle.dump(new_ph_dict, fp)
        with open(os.path.join(self.crpath, self.cpath, 'bullet_rules.json'), 'w') as fp:
            json.dump(categorized_rule_dict, fp)

        s3.upload_file(os.path.join(self.crpath, self.cpath, 'phraseology_embeddings.pkl'), self.buck_name,
                       'CTA_Bullets/phraseology_embeddings.pkl')
        s3.upload_file(os.path.join(self.crpath, self.cpath, 'bullet_rules.json'), self.buck_name,
                       'CTA_Bullets/bullet_rules.json')
