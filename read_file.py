import json
import os
from amseg.amharicNormalizer import AmharicNormalizer as normalizer
from amseg.amharicSegmenter import AmharicSegmenter

path = '/home/bethel/ir_proj/amfiles_json/AA/'           # path to the Amharic wiki dump files in json format
dir_list = os.listdir(path)
dir_list = dir_list[:3]# the file names in the folder
# print(dir_list)


def read_wiki_files():
    """ To read the wiki files and return a list of the text"""
    make_list = []                                  # list for the title-text of each doc
    for file in dir_list:
        with open(path + file) as f:
            for line in f:
                line = json.loads(line)             # String to dictionary
                title = line["title"]
                text = line["text"]
                line['title_text'] = title + " " + text
                make_list.append(line['title_text'])
                if len(make_list) > 1:
                    break
    return make_list


def to_normalize(passage):
    """To normalize text in the file"""
    return normalizer.normalize(passage)


def to_segment(passage):
    """ To segment """
    sent_punct = []
    word_punct = []
    segmenter = AmharicSegmenter(sent_punct, word_punct)
    words = segmenter.amharic_tokenizer(passage)
    return words


list_doc_text = read_wiki_files()     # List of text in each doc

print(" ")

normalized = []
for element in list_doc_text:
    norm = to_normalize(element)
    normalized.append(norm)
# print(len(normalized))
print(normalized)

print(" ")

for element in normalized:
    segmented = to_segment(element)
    print(segmented)

