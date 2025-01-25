import streamlit as st
import fitz  # PyMuPDF
import fitz
import os
import re

def get_all_filenames(directory):
    try:
        # Get the list of all files and directories in the specified directory
        files = os.listdir(directory)
        
        # Filter out directories, only keeping files
        filenames = [f for f in files if os.path.isfile(os.path.join(directory, f))]
        
        # Sort the filenames by the numeric part before the underscore
        filenames.sort(key=lambda x: int(re.match(r'^(\d+)_', x).group(1)))
        
        return filenames
    except FileNotFoundError:
        return "The directory does not exist."
    except Exception as e:
        return f"An error occurred: {e}"



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
            


def main():
    st.title("PDF 파일 업로드 및 열기")

    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    if uploaded_file is not None:
        # 파일을 temp 파일로 저장
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("파일이 성공적으로 업로드되었습니다!")

        # PDF 파일 열기
        try:
            doc = fitz.open("temp.pdf")
            st.write("PDF 파일이 성공적으로 열렸습니다!")
            st.write(f"총 페이지 수: {doc.page_count}")
            # 첫 페이지 내용 출력 (원하는 페이지를 선택 가능)
            page = doc.load_page(0)
            parsing = pdf_to_dict(doc)
            st.text_area("Title", parsing['title'])
            st.text_area("Abstract", parsing['abstract'])
            st.text_area("Intoduction", parsing['intro'])
            st.text_area("Body", parsing['body'])
        except Exception as e:
            st.error(f"PDF 파일을 여는 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()