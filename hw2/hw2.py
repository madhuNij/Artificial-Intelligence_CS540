import sys
import math
import string


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X = dict(zip(string.ascii_uppercase, [0]*26))
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        text = f.read()
        all_upper = text.upper()
        #print(uppercased_string)
        for i in all_upper:
            if i.isalpha():
                X[i] = X[i] + 1
        sorted_dict = {key: value for key, value in sorted(X.items())}
        #print(sorted_dict)

    return X

def calculate_F():
    F = dict()
    i = english_val = spanish_val = 0
    for key in X.keys():
        english_val = english_val + (X[key] * math.log(e[i]))
        spanish_val = spanish_val + (X[key] * math.log(s[i]))
        i = i + 1
    F['English'] = math.log(P['English']) + english_val
    F['Spanish'] = math.log(P['Spanish']) + spanish_val
    return F

# TODO: add your code here for the assignment
# You are free to implement it as you wish!
# Happy Coding!
filename = "letter.txt"
X = dict()
X = shred(filename)
print("Q1")
for i in X.keys():
    print(i, X[i])

e,s = get_parameter_vectors()
e1 = e[0]
s1 = s[0]
x1 = X['A']
print("Q2")
print(round(x1*math.log(e1), 4))
print(round(x1*math.log(s1), 4))

P = F = dict()
P['English'] = 0.6
P['Spanish'] = 0.4
F = calculate_F()
print("Q3")
print(round(F['English'], 4))
print(round(F['Spanish'], 4))

val = F['Spanish'] - F['English']
if val >= 100:
    probablity_english_given_X = 0
elif val <= -100:
    probablity_english_given_X = 1
else:
    probablity_english_given_X = 1 /(1 + math.exp(val))
print("Q4")
print(round(probablity_english_given_X, 4))