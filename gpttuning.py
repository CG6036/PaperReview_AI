import json
import argparse
import random
import re
import pprint
import os
import fitz
import openai

import numpy as np

from pathlib import Path
from dotenv import load_dotenv, set_key
from unicodedata import normalize
from time import sleep
from random import randrange

def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def split_data(data):
    start, end = 0, 0
    li = []
    while True:
        end = data.find("Here are the", end + 1)
        if end == -1:
            li.append(data[start:])
            break
        li.append(data[start:end])
        start = end
    
    return li

def is_uppercase(char):
    if 'A' <= char <= 'Z':
        return True
    else:
        return False

def pdf_to_dict(doc, file_name) :

    paper = {}

    title = ""
    abst = ""
    intro = ""
    body = ""
    concle = ""
    title_prg = False
    abst_prg = False
    intro_prg = False
    body_prg = False
    concle_prg = False

    for page in doc:
        text = page.get_text()
        text = text.split('\n')
        
        for l in text:
            if l == "Published as a conference paper at ICLR 2023" :
                continue
            if len(l) > 2 and (is_uppercase(l[1]) and is_uppercase(l[2])) :
                if title == "" or title_prg == True :
                    if l[len(l)-1] == '-':
                        title += l[:-1]
                    else :
                        title += l
                    title_prg = True

                if len(l.split(" ")) == 1 and is_uppercase(l[-1]):
                    if l == "ABSTRACT" :
                        abst_prg = True
                        continue
                    if l == "INTRODUCTION" :
                        intro_prg = True
                        abst_prg = False
                        abst = abst[:-1]
                        continue
                    if intro_prg == True :
                        body_prg = True
                        intro_prg = False
                        intro = intro[:-1]
                        continue
                    if l == "CONCLUSION" or  l == "CONCLUSIONS AND FUTURE WORK" :
                        concle_prg = True
                        body_prg = False
                        body = body[:-1]
                        continue
                    if l == "REFERENCES" :
                        concle_prg = False
                        concle = concle[:-1]
                        continue

            else :
                if title != ""  and title_prg == True :
                    title_prg = False
                elif abst_prg == True :
                    abst += l
                elif intro_prg == True :
                    intro += l
                elif body_prg == True :
                    body += l
                elif concle_prg == True :
                    concle += l

    paper['title'] = file_name.split('_')[1][:-4]
    paper['abstract'] = abst
    paper['intro'] = intro
    paper['body'] = body
    #paper['conclusion'] = concle
    return paper

def rename_pdf(args):
    path = args.pdf_data_path
    file_list = os.listdir(path)

    # Define the range end
    end_range = 2000
    # Generate the zero-padded index strings
    index_strings = [f'{i:04}' for i in range(end_range + 1)]

    for idx, file_name in enumerate(file_list):
        splited_file_name = file_name.split('_')[1]
        os.rename(path+file_name, path+index_strings[idx]+'_'+splited_file_name)

def write_pdf(args):
    file_list = os.listdir(args.pdf_data_path)
    
    file_lists = []
    for file_name in file_list:
        pdf_file = fitz.open(args.pdf_data_path+file_name)
        file_contents = pdf_to_dict(pdf_file, file_name)
        file_lists.append(file_contents)
    
    with open("./data/pdf_train_data.json", "w") as json_file:
        json.dump(file_lists, json_file, indent=4)

def pdf_preprocessing_data(pdf, reviews):
    processed_train_data = []
    processed_label_data = {}
    
    system_state = {"role": "system", "content": "A prediction model that predicts review results based on the paper"}
    user_input = {"role": "user", "content": "Please predict the outcome of a paper review based on its title, abstract, introduction, main content.\n Your ultimate goal is to inform us whether the paper will be accepted or not.\n\n Here are the steps:\nFirst, you analyze the strengths and weaknesses based on the paper's title, abstract, introduction, main content.\n Second, you assign a grade on the technical novelty. The technical novelty should be rated on a scale of 1 to 4, and the score should be written in the format \'<score>\'. \n Third, you decide whether the submitted paper should be accepted or rejected based on the grades. The final grade should be rated on a scale of 1 to 10, and the score should be written in the format '<score>'. \n\n Here are the title, abstract, introduction, main content\n"}
    
    title = pdf['title'].encode('ascii', errors='ignore').strip().decode('ascii').lower()
    abstract = reviews['abstract'].encode('ascii', errors='ignore').strip().decode('ascii').lower()
    introduction = pdf['intro'].encode('ascii', errors='ignore').strip().decode('ascii').lower()
    main_body = pdf['body'].encode('ascii', errors='ignore').strip().decode('ascii').lower()

    processed_label_data['title'] = title

    title = 'The title of paper is \'' + title +  '\'. '
    abstract = 'The abstract of paper is \'' + abstract +  '\'. '
    introduction = 'The introdction of paper is \'' + introduction +  '\'. '
    main_body = 'The main content of paper is \'' + main_body +  '\'. '

    user_input['content'] = user_input['content'] + title + '\n ' + abstract
    user_input['content'] = user_input['content'] + introduction + '\n ' + main_body
    
    decision = 'reject' if reviews['deicision'][0]['decision'] == 'Reject' else 'accept'
    technical_novelty_score_list = []
    recommendation_score_list = []
    for review in reviews['reviews']:
        gpt_format = {}
        gpt_output = {"role": "assistant", "content": ""}
        output = ''

        recommendation = review['recommendation']
        recommendation = recommendation.split(':')[0]
        recommendation_score_list.append(int(recommendation))

        technical_novelty = review['technical_novelty_and_significance']
        technical_novelty_score_list.append(int(technical_novelty[0]))

        if decision == 'reject' and int(recommendation) >= 6:
            continue
        elif decision == 'accept' and int(recommendation) < 6:
            continue

        recommendation_grade = 'Here are the predicted grade \n'
        recommendation_grade = recommendation_grade + 'grade(1~10) : <' + recommendation + '> \n '
        recommendation = recommendation_grade

        sandw = review['strength_and_weaknesses'].encode('ascii', errors='ignore').strip().decode('ascii').lower()
        sandw = re.sub('[\n]', ' ', sandw)
        sandw = re.sub('[*#\n$]', ' ', sandw)
        
        sandw = 'Here are the strengths and weakness of the paper. \n' + sandw + '\n\n '
        
        technical_novelty_grade = 'Here are the grade for the technical novelty of the paper. \n' 
        technical_novelty_grade = technical_novelty_grade + 'the grade of technical novelty(1~4) : <' + technical_novelty[0] + '> \n '
        technical_novelty = technical_novelty_grade
        
        cqnr = review['clarity,_quality,_novelty_and_reproducibility'].encode('ascii', errors='ignore').strip().decode('ascii').lower()
        cqnr = re.sub('[\n]', ' ', cqnr)
        cqnr = re.sub('[*#\n$]', ' ', cqnr)

        cqnr = 'Here are the clarity, quality, novelty and reproducibility of the paper. \n'  + cqnr + '\n\n '

        output = sandw + cqnr + technical_novelty + recommendation
        
        gpt_output['content'] = output
        gpt_format['messages'] = [system_state, user_input, gpt_output]
        
        processed_train_data.append(gpt_format)
    recommendation_score = int(np.around(np.mean(recommendation_score_list)))
    technical_novelty_score = int(np.around(np.mean(technical_novelty_score_list)))
    
    processed_label_data['technical_novelty'] = technical_novelty_score
    processed_label_data['recommendation'] = recommendation_score
    processed_label_data['recommendation_list'] = recommendation_score_list
    processed_label_data['decision'] = decision

    return processed_train_data, processed_label_data

def preprocess_pdf_data(args):
    pdf_data = read_json(args.pdf_training_data_path)
    review_data = read_json(args.review_data_path)

    processed_train_data = []
    processed_label_data = []
    for pdf in pdf_data:
        pdf_title = pdf['title'].lower()
        for review in review_data:
            review_title = review['title'].lower()
            if pdf_title == review_title:
                train_data, label_data = pdf_preprocessing_data(pdf, review)
                processed_train_data.append(train_data)
                processed_label_data.append(label_data)
    
    with open("./data/processed_pdf_train_data.json", "w") as json_file:
        json.dump(processed_train_data, json_file, indent=4)
    
    with open("./data/processed_pdf_label_data.json", "w") as json_file:
        json.dump(processed_label_data, json_file, indent=4)

def convert_to_gpt_format(args):
    training_data = read_json(args.processed_pdf_training_data_path)
    label_data = read_json(args.processed_pdf_training_data_path)

    random.seed(args.random_seed)
    random_indexes = random.sample(range(len(training_data)-100), 250)
    train_index = random_indexes[:200]
    validation_index = random_indexes[200:]

    with open("./data/pdf_gptformat_training_data.jsonl", "w") as json_file:
        for tr_idx in train_index:
            train = training_data[tr_idx]
            train = random.choice(train)
            json_line = json.dumps(train)
            json_file.write(json_line+'\n')
            
    with open("./data/pdf_gptformat_validation_data.json", "w") as json_file:
        final_data = []
        for valid_idx in validation_index:
            validation = training_data[valid_idx]
            validation = validation[0]
            final_data.append([validation['messages'][1], validation['messages'][2]])
        json.dump(final_data, json_file, indent=4)
    
    with open("./data/pdf_gptformat_label_data.json", "w") as json_file:
        final_data = []
        for label_idx in validation_index:
            label = label_data[label_idx]
            final_data.append(label)
        json.dump(final_data, json_file, indent=4)

def prepare_openai(path):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    model = openai.File.create(
        file=open(path, "r"),
        purpose='fine-tune'
    )
    print(model)

    return model

def finetune_gpt(model):
    job = openai.FineTuningJob.create(
        model="gpt-3.5-turbo",
        training_file=model['id']
    )
    
    job_id = job['id']
    print('before training')
    print(job)
    set_key(dotenv_path=args.env_path, key_to_set="JOB_ID", value_to_set=job_id)

    res = None
    while True:
        res = openai.FineTuningJob.retrieve(job_id)
        if res["finished_at"] != None:
            break
        else:
            print("training continue")
            sleep(100)
    
    ft_model = res
    print('after training')
    print(ft_model)
    set_key(dotenv_path=args.env_path, key_to_set="FT_MODEL", value_to_set=ft_model["fine_tuned_model"])

    return ft_model["fine_tuned_model"]

def save_validation_results(path):
    results = []
    json_data = read_json(path)

    openai.api_key = os.environ["OPENAI_API_KEY"]
    ft_model = os.environ["FT_MODEL"]
    for data in json_data:
        print(data[0])
        response = openai.ChatCompletion.create(
            model = ft_model,
            messages = [
                {"role": "system", "content": "A prediction model that predicts review results based on the abstract"},
                data[0]
            ]
        )
        results.append({
            "input": data,
            "output": response.choices[0].message.content
        })

    with open("./data/pdf_gpt_validation_results.json", "w") as json_file:
        json.dump(results, json_file, indent=4)

def show_validation_test(args):
    validation_results_data = read_json(args.gpt_validation_results_path)
    processed_label_data = read_json(args.processed_pdf_label_data_path)

    novelty_grade, recommendation_grade, accept_reject_grade = [], [], []
    novelty_grade_score, recommendation_grade_score = 0, 0
    accept_th_score = 0
    novelty_th_score = 0
    accept_reject_score = 0
    accept_cnt= 0
    predicted_accept_cnt = 0
    for valid in validation_results_data:
        title = valid['input'][0]['content'].split('content\nThe title of paper is')[1]
        title = title[:title.find('The abstract of paper is')]
        title = title[2:-5]
        
        for label in processed_label_data:
            if label['title'] == title:
                output = valid['output']
                splited_output = split_data(output)

                if len(splited_output) < 3:
                    continue
                if len(splited_output) == 5:
                    splited_output[1] = splited_output[-2]
                    splited_output[2] = splited_output[-1]

                novelty_grade.append([label['technical_novelty'], int(re.findall(r"<(\d+)>", splited_output[1])[0])])
                recommendation_grade.append([label['recommendation'], int(re.findall(r"<(\d+)>", splited_output[2])[0])])
                accept_reject_grade.append([label['decision'], 'accept' if int(re.findall(r"<(\d+)>", splited_output[2])[0]) >= 6 else 'reject'])
                if label['decision'] == 'accept':
                    accept_cnt += 1
                if accept_reject_grade[-1][1] == 'accept':
                    predicted_accept_cnt += 1

    for novelty, recommendation, accept_reject in zip(novelty_grade, recommendation_grade, accept_reject_grade):
        if novelty[0] == novelty[1]:
            novelty_grade_score += 1
        if novelty[0] < 3 and novelty[1] < 3:
            novelty_th_score += 1
        elif novelty[0] >= 3 and novelty[1] >= 3:
            novelty_th_score += 1
        
        if recommendation[0] == recommendation[1]:
            recommendation_grade_score += 1
        if recommendation[0] < 6 and recommendation[1] < 6:
            accept_th_score += 1
        elif recommendation[0] >= 6 and recommendation[1] >= 6:
            accept_th_score += 1
        
        if accept_reject[0] == accept_reject[1]:
            accept_reject_score += 1

    novelty_grade_score /= len(novelty_grade)
    novelty_th_score /= len(novelty_grade)
    recommendation_grade_score /= len(novelty_grade)
    accept_th_score /= len(novelty_grade)
    accept_reject_score /= len(novelty_grade)
    print(f'Number of Validation sets : {len(novelty_grade)}')
    print(f'Number of Accepted paper(label, predicted) : {accept_cnt}, {predicted_accept_cnt}')
    print(f'Number of Rejected paper(label, predicted) : {len(novelty_grade) - accept_cnt}, {len(novelty_grade) - predicted_accept_cnt}')
    print(f'Novelty score : {novelty_grade_score*100}%')
    print(f'Novelty th score : {novelty_th_score*100}%')
    print(f'Recommendation score : {recommendation_grade_score*100}%')
    print(f'Recommendation th score : {accept_th_score*100}%')
    print(f'Accept-Reject score : {accept_reject_score*100}%')

def run(args):
    pdf_train_path = Path(args.pdf_training_data_path)
    if not pdf_train_path.is_file():
        write_pdf(args)

    processed_pdf_train_path = Path(args.processed_pdf_training_data_path)
    processed_pdf_label_path = Path(args.processed_pdf_label_data_path)
    if not processed_pdf_train_path.is_file() or not processed_pdf_label_path.is_file():
        preprocess_pdf_data(args)
    
    gptformat_pdf_train_path = Path(args.gptformat_pdf_train_data_path)
    gptformat_pdf_validation_path = Path(args.gptformat_pdf_validation_data_path)
    gptformat_pdf_label_path = Path(args.gptformat_pdf_label_data_path)
    if not gptformat_pdf_train_path.is_file() or not gptformat_pdf_validation_path.is_file() or not gptformat_pdf_label_path.is_file():
        convert_to_gpt_format(args)
    
    if args.finetuning_gpt:
        model = prepare_openai(args.gptformat_pdf_train_data_path)
        ft_model = finetune_gpt(model)
        print(ft_model)
    
    gpt_validation_results_path = Path(args.gpt_validation_results_path)
    if not gpt_validation_results_path.is_file():
        save_validation_results(args.gptformat_pdf_validation_data_path)
    show_validation_test(args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-data-path", default="./data/iclr2023_papers_reviews.json", type=str)

    parser.add_argument("--pdf-data-path", default="./data/pdf_data/", type=str)
    parser.add_argument("--pdf-training-data-path", default="./data/pdf_train_data.json", type=str)
    parser.add_argument("--processed-pdf-training-data-path", default="./data/processed_pdf_train_data.json", type=str)
    parser.add_argument("--processed-pdf-label-data-path", default="./data/processed_pdf_label_data.json", type=str)
    parser.add_argument("--gptformat-pdf-train-data-path", default="./data/pdf_gptformat_training_data.jsonl", type=str)
    parser.add_argument("--gptformat-pdf-validation-data-path", default="./data/pdf_gptformat_validation_data.json", type=str)
    parser.add_argument("--gptformat-pdf-label-data-path", default="./data/pdf_gptformat_label_data.json", type=str)
    parser.add_argument("--gptformat-pdf-validation-results-path", default="./data/pdf_validation_results_data.json", type=str)
    parser.add_argument("--gpt-validation-results-path", default="./data/pdf_gpt_validation_results.json", type=str)

    parser.add_argument("--env-path", default=".env", type=str)
    parser.add_argument("--random-seed", default=20241, type=int)
    parser.add_argument("--finetuning-gpt", default=False, type=bool)

    args = parser.parse_args()
    load_dotenv()
    # rename_pdf(args)

    run(args)