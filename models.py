
from utils import download_s3_folder, download_file, s3
from nltk.corpus import stopwords
import os
from itertools import product
import json
import pickle
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer

bucket_name = os.getenv('BUCKET_NAME', 'contentware-nlp')
logger = logging.getLogger()
download_s3_folder(bucket_name, 'question-generation')
st_model = SentenceTransformer('all-MiniLM-L6-v2')

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

def make_cta_embeddings(rules: dict):
    cta_phraseologies = []
    for r in rule_cats:
        for k in rules.keys():
            values = [x[0] for x in rules[k][r]]
            cta_phraseologies.extend(values)
    cta_emb = st_model.encode(cta_phraseologies)
    return cta_emb

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

def replace_rule_tokens(base_rule_dict, event_key) -> dict:
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

    def CongigureCTA(self):
        download_file(self.cpath, 'campaign-metadata.json', self.buck_name)
        # Load JSON CTA/Wrapper Info
        with open(os.path.join(self.crpath, self.cpath, 'campaign-metadata.json')) as f:
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

        categorized_rule_dict = {}
        for p in ptl:
            categorized_rule_dict[p] = replace_rule_tokens(new_rule_dict, p.replace('_', ' '))

        event_dict = {}
        for p in ptl:
            ed = replace_tokens(dict_of_dfs, p.replace('_', ' '))
            event_dict[p] = ed
        cta_embeddings = make_cta_embeddings(event_dict)
        with open(os.path.join(self.crpath, self.cpath, 'cta_embeddings.pkl'), 'wb') as fp:
            pickle.dump(cta_embeddings, fp)
        with open(os.path.join(self.crpath, self.cpath, 'phraseology_embeddings.pkl'), 'wb') as fp:
            pickle.dump(event_dict, fp)
        with open(os.path.join(self.crpath, self.cpath, 'bullet_rules.json'), 'w') as fp:
            json.dump(categorized_rule_dict, fp)

        s3.upload_file(os.path.join(self.crpath, self.cpath, 'phraseology_embeddings.pkl'), self.buck_name,
                       'CTA_Bullets/phraseology_embeddings.pkl')
        s3.upload_file(os.path.join(self.crpath, self.cpath, 'bullet_rules.json'), self.buck_name,
                       'CTA_Bullets/bullet_rules.json')




