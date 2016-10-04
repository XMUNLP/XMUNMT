# tokenization.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import re


# tokenization from mteval
def tokenization(text):
    # language-independent part
    text = text.replace("<skipped>", "")
    text = text.replace("-\n", " ")
    text = text.replace("&quot;", "\"")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt", ">")

    # lowercase
    text = text.lower()
    # tokenize punctuation
    text = re.sub("([\{-~\[-` -&\(-\+:-@/])", " \g<1> ", text)
    # tokenize period and comma unless preceded by a digit
    text = re.sub("([^0-9])([.,])", "\g<1> \g<2> ", text)
    # tokenize period and comma unless follwed by a digit
    text = re.sub("([.,])([^0-9])", " \g<1> \g<2>", text)
    # tokenize dash when preceded by a digit
    text = re.sub("([0-9])(-)", "\g<1> \g<2> ", text)
    # one space only between words
    text = " ".join(text.split())
    # no leading and trailing space
    text = text.strip()

    return text
