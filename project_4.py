import json
import math
import os
import numpy as np
from amseg.amharicNormalizer import AmharicNormalizer as normalizer
from amseg.amharicSegmenter import AmharicSegmenter
from tqdm import tqdm

path = 'amfiles_json/AA/'  # path to the Amharic wiki dump files in json format
dir_list = os.listdir(path)


# dir_list = dir_list  # the file names in the folder


# dir_list = dir_list[:4]  # the file names in the folder
# print(dir_list)                       # output: ['wiki_10', 'wiki_07', 'wiki_02', 'wiki_19']


def read_wiki_files():
    """ To read the wiki files and return a list of the text"""
    make_list = []  # list for the title-text of each doc
    for file in dir_list:
        with open(path + file) as f:
            for line in f:
                line = json.loads(line)  # String to dictionary
                line['title_text'] = line["title"] + " " + line["text"]  # to make a title-text
                make_list.append(line['title_text'])
                # make_list.append(line['title_text'][:30])
                # if len(make_list) > 1:
                #     break
    return make_list


def to_normalize(passage):  # function to normalize words
    """To normalize text in the file"""
    return normalizer.normalize(passage)


def for_normalize(list_doc_text):  # function to do it over all docs
    normalized = []
    for document in list_doc_text:
        norm_doc = to_normalize(document)
        normalized.append(norm_doc)
    return normalized


def to_segment(passage):  # function to segment the words
    """ To segment """
    sent_punct = []
    word_punct = []
    segmenter = AmharicSegmenter(sent_punct, word_punct)
    words = segmenter.amharic_tokenizer(passage)
    return words


def for_segment(list_doc_text):  # function to do it over all docs
    segmented = []
    for document in list_doc_text:
        segmented_doc = to_segment(document)
        segmented.append(segmented_doc)
    return segmented


def to_dic(the_segmented_list):
    dict_doc = {}
    for i in range(len(the_segmented_list)):
        dict_doc[i] = the_segmented_list[i]
    return dict_doc


def to_cf(the_segmented_list):
    dict_word = {}
    for i in range(len(the_segmented_list)):  # make a dictionary of the words to find the stop words
        for j in range(len(the_segmented_list[i])):
            word = the_segmented_list[i][j]
            if word in dict_word:
                dict_word[word] += 1
            else:
                dict_word[word] = 1
    return dict_word


def to_remove_punc(word_dic1):
    punc_to_remove = [":", "።", "፣", "፨", "፠", "፤", "፥", "፦"]
    for key in punc_to_remove:
        del word_dic1[key]
    return word_dic1


def to_sort_list(to_be_sorted):  # sorting the segment inorder to make tf
    sorted_list = []
    for i in range(len(to_be_sorted)):
        the_sorted = sorted(to_be_sorted[i])
        sorted_list.append(the_sorted)
    return sorted_list


def to_remove_duplicates(to_be_removed):  # here removed the duplicates from the sorted list
    remove_duplicates_fun = []
    for i in range(len(to_be_removed)):
        after_removed = list(sorted((set(to_be_removed[i]))))
        remove_duplicates_fun.append(after_removed)
    return remove_duplicates_fun


# def count(list_to_count):
#     single_doc = {}
#     for i in range(len(list_to_count)):
#         word = list_to_count[i]
#         occurrences = list_to_count.count(word)
#         single_doc[word] = occurrences
#     return single_doc

def cosine_normalisation(single_doc):
    w = list(single_doc.values())
    w_square = np.square(w)
    den_add = np.sum(w_square)
    den_root = math.sqrt(den_add)
    cosine = 1 / den_root
    for key in single_doc:
        single_doc[key] = single_doc[key] * cosine
    return single_doc


def count(list_to_count):
    single_doc = {}
    i = 0
    while i < len(list_to_count):
        word = list_to_count[i]
        cur_count = 1.0
        i += 1
        while i < len(list_to_count) and list_to_count[i] == word:
            cur_count += 1
            i += 1
        single_doc[word] = 1 + math.log(cur_count, 10)
    single_doc = cosine_normalisation(single_doc)
    return single_doc


def to_make_tf(the_list):
    whole_doc = {}
    for i in range(len(the_list)):
        dic = count(the_list[i])
        whole_doc[i] = dic
    return whole_doc


def to_make_idf(n, df):
    for term, freq in df.items():
        div = n / freq
        the_log = math.log(div, 10)
        df[term] = the_log
    return df


def write_text_in_json(text, file_name):
    with open(file_name, 'w') as f:
        json.dump(text, f)
    return file_name


def read_text_from_json(text_file):
    with open(text_file) as f:
        text_to_read = json.load(f)
    return text_to_read


def for_calculate_tf_idf(tf_queries, idf_document):
    for query_id in tf_queries:
        cur_query_tf = tf_queries[query_id]
        query_tf_idf = to_calculate_tf_idf(cur_query_tf, idf_document)
        tf_queries[query_id] = query_tf_idf
    return tf_queries


def to_calculate_tf_idf(tf_query, idf_document):
    term_value = {}
    freq_list = list(tf_query.values())
    term_list = list(tf_query)

    for j in range(len(term_list)):
        term_key = term_list[j]
        if term_key in idf_document:
            term_value[term_key] = freq_list[j] * idf_document[term_key]
        else:
            term_value[term_key] = freq_list[j] * 0
    return term_value


def score_tf_idf(tf_idf):
    score = {}
    term_value_list = list(tf_idf.values())
    for i in range(len(term_value_list)):
        freq_list = list(term_value_list[i].values())
        score_item = sum(freq_list)
        for j in range(len(tf_idf)):
            score[j] = score_item
    return score


def to_query_to_document_score(query_tf, doc_tf):
    single_score = 0

    for term, frequency in query_tf.items():
        if term in doc_tf:
            single_score += doc_tf[term] * query_tf[term]
    return single_score


def to_query_to_all_document_score(query_tf, doc_tf):
    score = {}

    for i in range(len(doc_tf)):
        score[i] = to_query_to_document_score(query_tf, doc_tf[i])

    return score


def sort_dic_by_value_tolist(dic_to_be_sorted):
    return list(sorted(dic_to_be_sorted.items(), key=lambda item: item[1], reverse=True))


def to_all_query_to_all_document_score(queries_tf, doc_tf, k):          # rename
    total_score = {}

    for key, query_tf in tqdm(queries_tf.items()):
        dic_to_be_sorted = to_query_to_all_document_score(query_tf, doc_tf)
        sorted_doc_scores = sort_dic_by_value_tolist(dic_to_be_sorted)
        top_k_doc = sorted_doc_scores[:k]
        total_score[key] = top_k_doc

    return total_score


def pre_process(list_doc_text):

    the_normalized = for_normalize(list_doc_text)                    # normalize the texts
    the_segmented = for_segment(the_normalized)                      # segment the words
    sort_segmented_list = to_sort_list(the_segmented)                # The sorted segment
    remove_duplicates = to_remove_duplicates(sort_segmented_list)    # remove duplicates from the segmented list

    return sort_segmented_list, remove_duplicates, the_normalized


def to_evaluate(normalized_answer, normalized_document, top_score, k):
    top_documents = [i[0] for i in top_score]

    score = 0
    for i in range(k):
        if normalized_answer in normalized_document[top_documents[i]]:
            score = 1
            break

    return score


def for_evaluation(normalized_answers, normalized_document, top_scores, k):
    return [to_evaluate(normalized_answers[i], normalized_document, top_scores[i], k) for i in range(len(normalized_answers))]


# """MAIN METHOD"""
def main():

    list_doc_text = read_wiki_files()                              # to read wiki files and return List of text in each doc

    sort_segmented_list, remove_duplicates, the_normalized_document = pre_process(list_doc_text) # we apply pre-processing techniques to the list of document texts

    print("\n************************************ for documents ****************************************\n")

    df_document = to_cf(remove_duplicates)                         # to make df for the doc - dictionary of {term_id : how many documents a term has occured in}

    tf_document = to_make_tf(sort_segmented_list)                  # to make tf for the doc - dictionary of a dictionary{term_id: number of times term occurs}

    number_of_doc = len(sort_segmented_list)                       # total number of docs in the collection
    idf_document = to_make_idf(number_of_doc, df_document)         # to make idf - apply idf to the docs

    print("\n************************************ for queries ****************************************\n")
    the_query = read_text_from_json('query_json/query1.json')                                        # we read the query from the json file

    sort_segmented_list_query, remove_duplicates_query, _ = pre_process(the_query)         # we apply pre-processing techniques to the query to get the sorted segmented query list and the list after removing the duplicates

    tf_query = to_make_tf(sort_segmented_list_query)              # to make tf for the query - dictionary (query_id) of a dictionary{term_id: number of times term has occured in a query}

    print("\n************************************ tf-idf & score ****************************************\n")

    tf_idf_query = for_calculate_tf_idf(tf_query, idf_document)                       # to calculate tf-idf

    top_scores = to_all_query_to_all_document_score(tf_idf_query, tf_document, 100)       # to calculate the score

    print("\n************************************ evaluation ****************************************")
    the_answer = read_text_from_json('answer_json/answer1.json')
    the_normalized_answer = for_normalize(the_answer)

    for k in [1, 5, 20, 100]:
        evaluation_list = for_evaluation(the_normalized_answer, the_normalized_document, top_scores, k)
        score = np.mean(evaluation_list) * 100
        print(f"Recall@{k} {score}")

main()
