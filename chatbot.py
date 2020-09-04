
from gensim.models import FastText
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
import math

qa_flag = True   # 判断是否在问答循环中

def selfcheck():
    yes_num = 0
    if_five = False
    print('Please answer \'yes\' or \'no\'.')
    print('Do you have a dry cough?')
    q = str(input()).lower()
    if q.find('yes') != -1:
        yes_num += 1
    print('Do you have fatigue?')
    q = str(input()).lower()
    if q.find('yes') != -1:
        yes_num += 1
    print('Do you have a fever?')
    q = str(input()).lower()
    if q.find('yes') != -1:
        yes_num += 1
    print('Do you have difficulty breathing?')
    q = str(input()).lower()
    if q.find('yes') != -1:
        yes_num += 1
    print('Have you been in contact with a confirmed patient?')
    q = str(input()).lower()
    if q.find('yes') != -1:
        yes_num += 1
        if_five = True

    if if_five:
        print('You may have coronavirus, please isolate yourself until you get the results. https://www.nhs.uk/contact-us/get-help-with-asking-for-a-coronavirus-test/.')
    else:
        if yes_num == 2:
            print('You may not have coronavirus，You can check the information about Covid through the website https://www.nhs.uk/conditions/coronavirus-covid-19/.')
        elif yes_num >= 3:
            print('It looks like you might have Covid, please get a test and isolate yourself until you get the results. https://www.nhs.uk/contact-us/get-help-with-asking-for-a-coronavirus-test/.')
        else:
            print('You may not have coronavirus, please take care of your body.')
    

def qa_bot(query):
    query = str(query).lower()
    if query.find('selfcheck') != -1:
        selfcheck()
        return True
    if query.find('bye') != -1:
        print('Bye!')
        return False
    search_answer(query)
    return True

def get_stopwords():
    # Get stopwords list & punctuations list
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')

    # punctuations
    english_punctuations = [',', '--', '<', '>', '.', ':', ';', '?', '(', ')', '[', ']', '&', '\'\'', '``', '!', '*', '@', '#', '$', '%']
    return stopwords, english_punctuations

def preprocess_sen(q):
    stemmer = LancasterStemmer()
    # split words + stem
    wordlist = [stemmer.stem(word.lower()) for word in word_tokenize(q)]
    # wordlist = data['review_content'].lower().split()

    # remove stopwords
    outlist = []  # output - list
    for word in wordlist:
        if word not in stopwords and word not in english_punctuations:
            if word != '\t' and word != '\n' and word != ' ':
                outlist.append(word)
    return outlist

def read_dics():
    with open('qa_dic.pickle', 'rb+') as f:
        qa_dic = pickle.load(f)
    with open('num_a_dic.pickle', 'rb+') as f:
        num_a_dic = pickle.load(f)
    with open('questions.pickle', 'rb+') as f:
        questions = pickle.load(f)
    with open('questionsraw.pickle', 'rb+') as f:
        questionsraw = pickle.load(f)
    return qa_dic, num_a_dic, questions, questionsraw

def tfidf(list_words):
    #总词频统计
    doc_frequency = {}
    for word_list in list_words:
        for i in word_list:
            if i not in doc_frequency:
                doc_frequency[i] = 0
            doc_frequency[i]+=1
 
    #计算每个词的TF值
    word_tf={}  #存储没个词的tf值
    for i in doc_frequency:
        word_tf[i]=doc_frequency[i]/sum(doc_frequency.values())
 
    #计算每个词的IDF值
    doc_num=len(list_words)
    word_idf={} #stores the idf value for each word
    word_doc={} #Stores the number of documents that contain the word
    for i in doc_frequency:
        for j in list_words:
            if i in j:
                if i not in word_doc:
                    word_doc[i] = 0
                word_doc[i]+=1
    for i in doc_frequency:
        word_idf[i]=math.log(doc_num/(word_doc[i]+1))
 
    #计算每个词的TF*IDF的值
    word_tf_idf={}
    for i in doc_frequency:
        word_tf_idf[i]=word_tf[i]*word_idf[i]
 
    return word_tf_idf

def get_sen_vec(word_tf_idf, q):
    sum_vec = np.zeros((1, 100))

    for w in q:
        if w in model.wv.index2word and w in word_tf_idf.keys():
            sum_vec += word_tf_idf[w] * model[w]
        else:
            sum_vec += np.random.rand(1, 100) * 0.0001

    return sum_vec

def get_questions_vec(word_tf_idf):
    q_vecs = {}
    for qr in questionsraw:
        q = q_qr_dic[qr]
        sum_vec = get_sen_vec(word_tf_idf, q)
        q_vecs[qr] = sum_vec
    return q_vecs


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def find_highst_qr(qvecs, qvec):
    qs_cos_dic = {}
    for vec in qvecs:
        cosin = cosine_similarity(qvecs[vec], qvec)
        qs_cos_dic[vec] = cosin
    qs_cos_dic = sorted(qs_cos_dic.items(), key=lambda d: d[1], reverse=True)
    # print(qs_cos_dic[:5])
    return qs_cos_dic[0]

def search_answer(query):
    q = preprocess_sen(query)
    word_tf_idf = tfidf(questions+q)

    questions_vecs = get_questions_vec(word_tf_idf)
    query_vec = get_sen_vec(word_tf_idf, q)

    qcos_pair = find_highst_qr(questions_vecs, query_vec)
    if qcos_pair[1] > 0.5:
        answer_list = qa_dic[qcos_pair[0]]
        idx = int(np.random.rand() * len(answer_list))
        print(num_a_dic[answer_list[idx]])
    else:
        print('Sorry, Cov can\'t answer that question. You can ask me anthor!')





model = FastText.load('fasttext.model')
qa_dic, num_a_dic, questions, questionsraw = read_dics()

# create dic of questions-questionsraw
q_qr_dic = {}
for q, qr in zip(questions, questionsraw):
    q_qr_dic[qr] = q

stopwords, english_punctuations = get_stopwords()

print("Hi! I'm Cov, what can I help? Ask me about anything! If you ask 'selfcheck', I will ask you some questions to confirm whether you are infected with COVID-19. If you say 'Bye', we will end our talk.")

while qa_flag:
    query = str(input())
    qa_flag = qa_bot(query)
