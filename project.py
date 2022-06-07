import json
import math
import os
from amseg.amharicNormalizer import AmharicNormalizer as normalizer
from amseg.amharicSegmenter import AmharicSegmenter

path = '/home/bethel/ir_proj/amfiles_json/AA/'  # path to the Amharic wiki dump files in json format
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
        single_doc[word] = cur_count
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


def write_query_in_json(query):
    query_file = 'query_json/query1.json'
    with open(query_file, 'w') as f:
        json.dump(query, f)
    return query_file


def read_query_from_json(query_file):
    with open(query_file) as f:
        query_to_read = json.load(f)
    return query_to_read


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

    for key, query_tf in queries_tf.items():
        dic_to_be_sorted = to_query_to_all_document_score(query_tf, doc_tf)
        sorted_doc_scores = sort_dic_by_value_tolist(dic_to_be_sorted)
        top_k_doc = sorted_doc_scores[:k]
        total_score[key] = top_k_doc

    return total_score


def pre_process(list_doc_text):
    print("\n normalize the texts")
    the_normalized = for_normalize(list_doc_text)

    print("\nsegment the words")
    the_segmented = for_segment(the_normalized)

    print("\nThe sorted segment")
    sort_segmented_list = to_sort_list(the_segmented)

    print("\nremove duplicates from the segmented list")
    remove_duplicates = to_remove_duplicates(sort_segmented_list)

    return sort_segmented_list, remove_duplicates, the_normalized


"""MAIN METHOD"""


def main():
    print("\n read wiki files")
    list_doc_text = read_wiki_files()  # List of text in each doc
    print(list_doc_text[:2])

    sort_segmented_list, remove_duplicates, the_normalized = pre_process(list_doc_text)

    # This is not used currently, but we will use it to remove punctuations and stopwords
    # print("\ndocument to dictionary")
    # to_dictionary = to_dic(the_segmented)
    # print(to_dictionary)
    # print(len(to_dictionary))
    # print(" ")

    # print("word to dictionary")
    # word_dic = to_cf(the_segmented)
    # print(word_dic)
    # print(len(word_dic))
    #
    # print(" ")
    #
    # print("remove punctuation from the segment")
    # # removed_punc = to_remove_punc(word_dic)
    # # print(removed_punc)
    #
    #
    # print(" ")
    #
    # print("sort the dictionary")
    # # sorted_dict = sorted(removed_punc.items(), key=lambda item: item[1], reverse = True)
    # # print(sorted_dict[:100])
    #

    print("\ndf - dictionary of {term_id: how many documents a term occured}")
    make_df = to_cf(remove_duplicates)
    print(make_df)

    print("\ntf - dictionary of a dictionary{term_id: number of times term occurs}")
    make_tf = to_make_tf(sort_segmented_list)
    # print(make_tf)

    print("\nidf - apply idf to the docs")
    N = len(sort_segmented_list)  # total number of docs in the collection
    # print(N)
    # dic_eg = {'a': 1, 'b': 2, 'c': 10, 'd': 8}
    # a = 10
    idf = to_make_idf(N, make_df)
    print(idf)

    print("\n************************************ for queries ****************************************\n")

    print("the query")
    my_query = ["ሒሳብ እጅግ በጣም በጣም ጠቃሚና ውበት ያለው የጥናትና የምርምር መስክ ወይም ዘርፍ ነው ።", "ሒሳብ እጅግ በጣም በጣም"]
    my_answer = ["እጅግ በጣም", "ሒሳ"]
    my_normalized_answer = for_normalize(my_answer)
    the_query_file = write_query_in_json(my_query)
    the_query = read_query_from_json(the_query_file)

    sort_segmented_list_query, remove_duplicates_query, _ = pre_process(the_query)

    print("\ntf - dictionary of a dictionary{term_id: number of times term occurs}")
    make_tf_query = to_make_tf(sort_segmented_list_query)
    print(make_tf_query)

    print(" ")

    print("\n************************************ tf-idf (doc)****************************************")

    print(" ")
    tf_idf_query = for_calculate_tf_idf(make_tf_query, idf)
    print(tf_idf_query)

    # print(to_query_to_document_score(tf_idf_query[0], make_tf[0]))

    all_doc_scores = to_query_to_all_document_score(tf_idf_query[0], make_tf)
    print(all_doc_scores)

    print(" ")

    top_scores = to_all_query_to_all_document_score(tf_idf_query, make_tf, 2)

    # Evaluation -
    # For a given query
        # Score = 1 if atleast one of top-k documents contains answer
        # Else, score = 0
    # Find the scores for all queries

    print(top_scores)



main()
