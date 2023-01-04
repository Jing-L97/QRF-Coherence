# -*- coding: utf-8 -*-
"""
To extract features for the model to classify coherent and non-coherent responses
"""
import spacy
import pandas as pd
import re
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
# from torch.nn import CrossEntropyLoss
# from scipy.special import softmax
# import string

child = pd.read_csv("child_coherence.csv")
caregiver = pd.read_csv("caregiver_coherence.csv")


###################
#Existing features#
###################
def extract_exist(child): 
    # get the question part as the input fea
    child_Q = child.loc[(child["RespCohere"]=='--')&(child["FollowCohere"]=='--')]
    child_R = child.loc[((child["RespCohere"]=='Yes')|(child["RespCohere"]=='No'))&(child["FollowCohere"]=='--')]
    child_F = child.loc[((child["FollowCohere"]=='Yes')|(child["FollowCohere"]=='No'))&(child["RespCohere"]=='--')]
    
    # combine the fea with the datapoints
    # response coherence and follow up coherence
    child_Q["RespCohere"] = child_R["RespCohere"].tolist()
    # follow-up coherence takes two types of input
    child_R["FollowCohere"] = child_F["FollowCohere"].tolist()
    child_R["utterance_Q"] = child_Q["utterance"].tolist()
    child_R["label_type_Q"] = child_Q["label_type"].tolist()
    child_R["label_subtype_Q"] = child_Q["label_subtype"].tolist()
    return child_Q,child_R



#################
#Verbal features#
#################

# uae regexp to clean the line
def clean_line(line: str) -> str:
    """This function will remove any irrelevant annotations"""
    non_letters_re = re.compile(r"(\[[a-zA-Z]+\])|((\\\<([a-zA-Z]+\s*)+\\>)+)")
    return non_letters_re.sub("", line)

# pre-process all the utterances
# input: dataframe and column header(str); return the dataframe with cleaned utterances
def clean_utterance(child_R,utterance):
    cleaned_utt_lst = []
    for i in child_R[utterance].tolist():
        cleaned_utt = clean_line(i)
        cleaned_utt_lst.append(cleaned_utt)    
    #replace the column
    child_R[utterance] = cleaned_utt_lst
    
    return child_R


child_R_raw,child_F_raw = extract_exist(child)
caregiver_R_raw,caregiver_F_raw = extract_exist(caregiver)
child_R = clean_utterance(child_R_raw,"utterance")
child_F_temp = clean_utterance(child_F_raw,"utterance")
child_F = clean_utterance(child_F_temp,"utterance_Q")
caregiverd_R = clean_utterance(caregiver_R_raw,"utterance")
caregiver_F_temp = clean_utterance(caregiver_F_raw,"utterance")
caregiver_F = clean_utterance(caregiver_F_temp,"utterance_Q")


nlp = spacy.load("fr_core_news_lg")


utt = child_R['utterance'].tolist()
n = 0
info_utt = []
while n < len(utt): 
    utt_name = child_R['unit'].tolist()[n]
    doc = nlp(utt[n])
    for token in doc:
        word = token.text
        tag = token.pos_
        info_utt.append([utt_name,word,tag])
    n+=1
utt_frame = pd.DataFrame(info_utt, columns = ['UtteranceName','Word','POS'])

# convert into one-hot encoding


# input: selected FA dataframe; a list containing word+utterance name
def match_POS(result,utt_frame_lst,utt_frame):
    result1 = result[['Word', 'Start', 'End','Length','UtteranceName','Speaker', 
                     'Filename','Global_start','Global_end','standard']].values.tolist()
    # match the POS tags to the word dataframe based on word and filename
    suspicious = []
    POS_lst = []
    final = pd.DataFrame()
    n = 0
    while n < len(result1):
        utterance = result1[n][-1]
        index = utt_frame_lst.index(utterance)
        selected = utt_frame.iloc[[index]]
        if selected.shape[0] > 1:
            suspicious.append(selected)
        else:  
            # get POS list
            final = pd.concat([selected,final])
            new = selected.values.tolist()
            POS = new[0][-2]
            POS_lst.append(POS)
        n+=1   
    # append the POS column to the original dataset
    result['POS'] = POS_lst
    final = result[['Filename','UtteranceName','Speaker','Word','Start','End','Global_start','Global_end','Length','POS']]
    return final


def append_POS(transcription,Words,language):
    utt_frame = get_POS(transcription,language)      
    if language == 'French':
        Words = Words[(Words['Filename'] != 'CA-BO-IO') & (Words['Filename'] != 'AA-BO-CM')]
    if language == 'English':
        Words = Words[(Words['Filename'] == 'CA-BO-IO') | (Words['Filename'] == 'AA-BO-CM')]
    utt_frame['standard'] = utt_frame['UtteranceName']+utt_frame['Word']
    Words['standard'] = Words['UtteranceName']+Words['Word']  
    utt_frame_lst = utt_frame['standard'].tolist()
    # get the matching word from FA document
    result = Words.loc[Words['standard'].isin(utt_frame_lst)] 
    # get more candidates from the unmatching parts
    rest = Words.loc[~Words['standard'].isin(utt_frame_lst)] 
    # split words contracted by '\'' or '-'
    contracted_word = rest.loc[(rest['Word'].str.contains('\''))|(rest['Word'].str.contains('-'))]
    contracted_word = contracted_word.reset_index()
    #contracted_word.drop(['Index','Unnamed'],inplace=True, axis = 1)
    candi = pd.DataFrame()
    n = 0
    while n < contracted_word.shape[0]:
        word = contracted_word['Word'][n]
        # split words based on connecting symbols
        word_lst = word.replace('-','\'').split('\'')
        # duplicate rows based on the no. of subcomponenets
        if len(word_lst)>1:   
            selected = contracted_word.iloc[[n]]    
            updated = selected.loc[selected.index.repeat(len(word_lst))]
            # replace the word column with the segmented words
            updated['Word'] = word_lst
            # concatenate the renewed dataframes
            candi = pd.concat([updated,candi])
        n +=1
    renewed_standard = candi['UtteranceName'] + candi['Word'] 
    candi['standard'] = renewed_standard
    # get the additional matching words
    add_result = candi.loc[candi['standard'].isin(utt_frame_lst)] 
    final_whole = match_POS(result,utt_frame_lst,utt_frame)
    final_add = match_POS(add_result,utt_frame_lst,utt_frame)
    final = pd.concat([final_whole,final_add])
    final.to_csv('POS.csv')
    return final

def extract_POS(whole_data,BC,n,timewindow):
    parti = BC['participant'][n]
    filename = BC['filename'][n][:-15]
    # types of POS
    POS_lst = whole_data['POS'].tolist()
    tag_lst = list(dict.fromkeys(POS_lst))
    # get the speaker's tag
    speaker = get_speaker(parti)
    number_lst = []
    candi = []
    # filter the desired POS data from the whole data
    for tag in tag_lst: 
        selected = whole_data.loc[((whole_data['Speaker'] == speaker) & (whole_data['Filename'] == filename) & (whole_data['POS'] == tag))]
        candi.append(selected)
        for frame in candi:
        # get the behaviors based on the time interval
            start_point = BC['onset.1'][n]-timewindow
            end_point = BC['onset.1'][n]
            final = frame.loc[(frame['Global_start'] >= start_point)&(frame['Global_end'] <= end_point)] 
            # calculate the number of occurence of each POS in the given time window       
            number = final.shape[0]
        number_lst.append(number)
    # return a dataframe with all the listed features    
    df_no = pd.DataFrame(number_lst).T 
    return df_no       

def concat_POS(data,timewindow):
    POS = pd.read_csv('POS.csv')
    n = 0
    POS_all_no = pd.DataFrame()
    while n < data.shape[0]:   
        # get each datapoint's preceding POS cues
        POS_no = extract_POS(POS,data,n,timewindow)
        POS_all_no = pd.concat([POS_all_no, POS_no])
        n += 1 
    POS_no_result = POS_all_no.rename(columns={0:'PRON',1:'ADP',2:'ADV',3:'CCONJ',4:'ADJ',5:'DET',
                                               6:'NOUN',7:'VERB',8:'INTJ',9:'SCONJ',10:'AUX', 
                                               11:'PUNCT',12:'PROPN',13:'NUM',14:'X',15:'SYMS',16:'PART'})

    POS_no_result.to_csv('POS_no.csv')
    return POS_no_result
