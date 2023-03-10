# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 10:49:36 2023

preprocess the CHILDES data for tokenization
"""

import pysbd
import os
import re
import string

seg = pysbd.Segmenter(language="fr", clean=False)

temp = []
path = './whole/'
with open('frchildes.txt', "w") as outfile:
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        with open(path + filename, encoding = 'utf-8') as f:
            file = f.readlines()
            
            for script in file: 
                if script.startswith('*'):
                    # remove speaker label and additional space
                    content = script.split('\t')[1]
                    temp.append(content)
                    # split the sentences on each line
                    result = seg.segment(content)
                    
                    for sent in result:
                        
                        # remove action markers
                        removed_action = re.sub(r"\[[a-z\s:]+\]", "", sent)
                        # remove non-printable characters
                        printable_contents = re.sub(r'[^\x20-\x7E]+', '', removed_action)
                        # get lower case and remove digits and punctuations
                        translator = str.maketrans('', '', string.digits + string.punctuation) 
                        clean_string = printable_contents.translate(translator)
                        
                        # Split the string into a list of words
                        words = clean_string.split()
                        # Join the words back into a string using a single space as the separator
                        new_string = ' '.join(words)
                         # remove the obvious non-words
                        removed_words = re.sub(r"(yyy|xxx)\s*", "", new_string)
                        
                        # remove additional blank lines
                        splited_lines = removed_words.splitlines()
                        # Filter out any empty lines using a list comprehension
                        non_empty_lines = [line for line in splited_lines if line.strip() != '']
                       
                        # Join the remaining lines back into a string using the join() method
                        cleaned_string = '\n'.join(non_empty_lines)
                        # remove the empty lines
                        if len(cleaned_string) > 0:  
                            outfile.write(removed_words + '\n')
                            temp.append(new_string)
                        
                        
 













