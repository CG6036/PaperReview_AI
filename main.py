import streamlit as st
import pandas as pd
import openai
import os
import re
from dotenv import load_dotenv
import time  
import fitz

st.set_page_config(page_title="Paper Review System for ICLR", page_icon="📄", layout='centered')

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
st.markdown("""
        <style>
        .stMarkdown {
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        mark {
            background-color: yellow;
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)



def process_with_langchain(abstract, title, intro, body):
    user_input = {"role": "user", "content": "Please predict the outcome of a paper review based on its title, abstract, introduction, main content.\n Your ultimate goal is to inform us whether the paper will be accepted or not.\n\n Here are the steps:\nFirst, you analyze the strengths and weaknesses based on the paper's title, abstract, introduction, main content.\n Second, you assign a grade on the technical novelty. The technical novelty should be rated on a scale of 1 to 4, and the score should be written in the format \'<score>\'. \n Third, you decide whether the submitted paper should be accepted or rejected based on the grades. The final grade should be rated on a scale of 1 to 10, and the score should be written in the format \'<score>\'. \n\n Here are the title, abstract, introduction, main content\n"}
    
    title = title.encode('ascii', errors='ignore').strip().decode('ascii').lower()
    abstract = abstract.encode('ascii', errors='ignore').strip().decode('ascii').lower()
    introduction = intro.encode('ascii', errors='ignore').strip().decode('ascii').lower()
    main_body = body.encode('ascii', errors='ignore').strip().decode('ascii').lower()
    
    title = 'The title of paper is \'' + title +  '\'. '
    abstract = 'The abstract of paper is \'' + abstract +  '\'. '
    introduction = 'The introdction of paper is \'' + introduction +  '\'. '
    main_body = 'The main content of paper is \'' + main_body +  '\'. '

    user_input['content'] = user_input['content'] + title + '\n ' + abstract
    user_input['content'] = user_input['content'] + introduction + '\n ' + main_body

    openai.api_key = os.environ["OPENAI_API_KEY"]
    ft_model = os.environ["FT_MODEL"]

    response = openai.ChatCompletion.create(
        model = ft_model,
        messages = [
            {"role": "system", "content": "A prediction model that predicts review results based on the abstract"},
            user_input
        ]
    )
    output = response.choices[0].message.content

    start, end = 0, 0
    li = []
    while True:
        end = output.find("Here are the", end + 1)
        if end == -1:
            li.append(output[start:])
            break
        li.append(output[start:end])
        start = end
    print(li)
    final_grade = int(re.findall(r"<(\d+)>", li[2])[0])
    final_grade_comment = ['reject, not good enough','marginally below the acceptance threshold',
                            'marginally above the acceptance threshold', 'accept, good paper', 'strong accept, should be highlighted at the conference']
    final_grade_index = 0
    if 3 < final_grade and final_grade <= 5:
        final_grade_index = 1
    elif final_grade == 6 or final_grade == 7:
        final_grade_index = 2
    elif final_grade == 8 or final_grade == 9:
        final_grade_index = 3
    elif final_grade_index == 10:
        final_grade_index = 4

    novelty_grade = int(re.findall(r"<(\d+)>", li[1])[0])
    novelty_grade_comment = ['The contributions are neither significant nor novel', 'The contributions are only marginally significant or novel',
                             'The contributions are significant and somewhat new. Aspects of the contributions exist in prior work', 'The contributions are significant, and do not exist in prior works']

    review = {}
    review['decision'] = 'Accept' if final_grade > 5 else 'Reject'
    review['recommendation'] = [final_grade, final_grade_comment[final_grade_index]]
    review['strengths_weaknesses'] = li[0]
    review['tech_novelty'] = [novelty_grade, novelty_grade_comment[novelty_grade-1]]

    return review


def display_results(review):
    st.header("Review Results")
    if review['decision'] == "Accept" :
        st.subheader("Predicted Decision: 🎉" + review['decision']+ "🎉")
    else :
        st.subheader("Predicted Decision: 😭" + review['decision']+ "😭")


    with st.expander(f"##### Recommendation {stars_rec(review['recommendation'][0])}") :
        st.write(f"{review['recommendation'][1]}")

    with st.expander(f"##### Technical Novelty {stars(review['tech_novelty'][0])}"):
        st.write(f"{review['tech_novelty'][1]}")

    with st.expander("##### Strengths and Weaknesses"):
        sw = review['strengths_weaknesses']
        first_strength =  sw.find('strengths')
        first_weakness =  sw.find('weakness')
        second_strength = sw.find('strength', first_strength + len('strength'))
        second_weakness = sw.find('weakness', first_weakness + len('weakness'))
        st.write(sw[:second_strength])
        st.divider()
        st.subheader("Strength")
        st.write(sw[(second_strength+ len('strength')+2):second_weakness])
        st.divider()
        st.subheader("Weakness")
        st.write(sw[(second_weakness+ len('weakness')+2):])


def stars_rec(score):
    # 최대 점수는 5점으로 가정
    max_score = 10
    star_full = '⭐️' * score
    star_empty = '🔳' * (max_score - score)
    return star_full + star_empty

def stars(score):
    # 최대 점수는 5점으로 가정
    max_score = 4
    star_full = '⭐️' * score
    star_empty = '🔳' * (max_score - score)
    return star_full + star_empty



def is_uppercase(char):
    if 'A' <= char <= 'Z':
        return True
    else:
        return False


def pdf_to_dict(doc) :

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
                    
    paper['title'] = title
    paper['abstract'] = abst
    paper['intro'] = intro
    paper['body'] = body

    #paper['conclusion'] = concle
    return paper
   



def display_review():
    st.title("Paper Review System for ICLR")
    # 페이지 리셋 버튼
    if 'reset' in st.session_state and st.session_state.reset:
        st.session_state.reset = False
        st.experimental_rerun()
        
    # 사용자로부터 논문의 Abstract 입력받기
    #####################New Code########################
    uploaded_file = st.file_uploader("Upload Your Paper's PDF", type=["pdf"])

    if uploaded_file is not None:
            # 파일을 temp 파일로 저장
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success("Upload Success.")

            # PDF 파일 열기
            try:
                doc = fitz.open("temp.pdf")
                # 첫 페이지 내용 출력 (원하는 페이지를 선택 가능)
    
                parsing = pdf_to_dict(doc)
                title = parsing['title']
                abstract = parsing['abstract']
                intro = parsing['intro']
                body = parsing['body']

                with st.expander("Check Uploaded Content"):
                    st.subheader("Title")
                    st.write(title)
                    st.divider()

                    st.header("Abstract")
                    st.write(abstract)
                    st.divider()

                    st.header("Introduction")
                    st.write(intro)

                    st.header("Body")
                    st.write(body)

            except Exception as e:
                st.error(f"Parsing Failed : {e}")
    #####################################################

    #title = st.text_area("Please Insert Your Paper's Title", height=150)
    #abstract = st.text_area("Please Insert Your Paper's Abstract", height=150)

    if st.button("Create Review"):
        if abstract and title and intro and body:
            with st.spinner('Your Paper is being processed via GPT...'):
                result = process_with_langchain(abstract, title, intro, body)
            if result is not None:
                st.session_state.reset = True  # 결과를 표시한 후 세션 상태를 리셋하기 위해 플래그 설정
                st.session_state.result = result  # 결과를 세션 상태에 저장
                st.rerun()  # 페이지 리프레시
            else:
                st.error("No result.")
        else:
            st.error("Please enter the abstract of your paper.")
            
           



if __name__ == "__main__":
    load_dotenv()
    if 'reset' in st.session_state and st.session_state.reset:
        display_results(st.session_state.result)
    else:
        display_review()
