import json
import math
import os
import numpy as np
from amseg.amharicNormalizer import AmharicNormalizer as normalizer
from amseg.amharicSegmenter import AmharicSegmenter
from tqdm import tqdm
import fasttext
import multiprocessing
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
import torch

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
        while len(norm_doc) > 0:
            length = min(100*4, len(norm_doc))
            normalized.append(norm_doc[:length])
            norm_doc = norm_doc[length:]
    return normalized


def to_segment(passage):  # function to segment the words
    """ To segment """
    sent_punct = ["።","፥","፨","::","፡፡","?","!", "፠", ]
    word_punct = ["።","፥","፤","፨","?","!",":","፡","፦","፣", "(", ")", "%", ","]
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
        single_doc[word] = 1 + math.log(cur_count, 10)
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


def to_query_to_document_score(query_tf, doc_tf, word_to_id, word_similarity_matrix):
    single_score = 0
    doc_words = list(doc_tf.keys())
    # query_words = list(query_tf.keys())
    if len(doc_words) == 0:
        return -100000

    doc_ids = [word_to_id[word] for word in doc_words]
    doc_similarity = word_similarity_matrix[doc_ids]
    # doc_most_similar = np.argmax(doc_similarity, axis=0)

    # for i in range(len(query_tf)):
    #     # term = query_words[i]
    #     # most_similar_word = doc_words[doc_most_similar[i]]
    #     # if term in doc_tf:
    #     # single_score += doc_tf[most_similar_word] * query_tf[term]
    #     single_score += np.sum(doc)
    return np.mean(doc_similarity)


def to_query_to_all_document_score(query_tf, doc_tf, word_vectors_document, word_to_id):
    score = {}

    query_vectors = np.array([ft.get_word_vector(word) for word in query_tf])
    word_similarity_matrix=np.matmul(word_vectors_document, query_vectors.T)

    for i in range(len(doc_tf)):
        score[i] = to_query_to_document_score(query_tf, doc_tf[i], word_to_id, word_similarity_matrix)

    return score


def sort_dic_by_value_tolist(dic_to_be_sorted):
    return list(sorted(dic_to_be_sorted.items(), key=lambda item: item[1], reverse=True))



def help_multi_cpu(args):
    key, query_tf, doc_tf, k, word_vectors_document, word_to_id = args
    dic_to_be_sorted = to_query_to_all_document_score(query_tf, doc_tf, word_vectors_document, word_to_id)
    sorted_doc_scores = sort_dic_by_value_tolist(dic_to_be_sorted)
    top_k_doc = sorted_doc_scores[:k]
    return top_k_doc


def to_all_query_to_all_document_score(queries_tf, doc_tf, k, word_vectors_document, word_to_id):          # rename
    total_score = {}

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        args_list = []
        for key, query_tf in queries_tf.items():
            args_list.append((key, query_tf, doc_tf, k, word_vectors_document, word_to_id))

        results = pool.map(help_multi_cpu, tqdm(args_list))
        for i in range(len(results)):
            total_score[args_list[i][0]] = results[i]

    return total_score


def pre_process(list_doc_text):

    the_normalized = for_normalize(list_doc_text)                    # normalize the texts
    the_segmented = for_segment(the_normalized)                      # segment the words
    sort_segmented_list = to_sort_list(the_segmented)                # The sorted segment
    remove_duplicates = to_remove_duplicates(sort_segmented_list)    # remove duplicates from the segmented list

    return sort_segmented_list, remove_duplicates, the_normalized


def to_evaluate(normalized_answer, normalized_document, top_score, k):
    top_documents = [i for i in top_score]

    score = 0
    for i in range(k):
        if normalized_answer in normalized_document[top_documents[i]]:
            score = 1
            break

    return score


def for_evaluation(normalized_answers, normalized_document, top_scores, k):
    return [to_evaluate(normalized_answers[i], normalized_document, top_scores[i], k) for i in range(len(normalized_answers))]


def convert_words_to_vectors(idf_document, ft):
    word_vectors = {}
    for key in idf_document:
        word_vectors[key] = ft.get_word_vector(key)
    return word_vectors

# """MAIN METHOD"""
def main():

    list_doc_text = read_wiki_files()                              # to read wiki files and return List of text in each doc

    _, _, the_normalized_document = pre_process(list_doc_text) # we apply pre-processing techniques to the list of document texts

    print("\n************************************ for documents ****************************************\n")

    the_normalized_document = the_normalized_document

    batch_size = 100
    embeddings_document_list = []
    for i in tqdm(range(math.ceil(len(the_normalized_document) / batch_size ))):
        batch = the_normalized_document[i*batch_size: (i+1) * batch_size]
        input_ids_document = tokenizer_context(batch, return_tensors="pt", padding=True, truncation=True, max_length=100)["input_ids"].to(device)
        embeddings_document = model_context(input_ids_document).pooler_output
        embeddings_document_list.append(embeddings_document.detach().cpu().numpy())
    embeddings_document = np.vstack(embeddings_document_list)

    print("\n************************************ for queries ****************************************\n")
    the_query = read_text_from_json('query_json/query1.json')                                        # we read the query from the json file
    _, _, normalized_query = pre_process(the_query)         # we apply pre-processing techniques to the query to get the sorted segmented query list and the list after removing the duplicates

    embeddings_query_list = []
    for i in tqdm(range(math.ceil(len(normalized_query) / batch_size ))):
        batch = normalized_query[i*batch_size: (i+1) * batch_size]
        input_ids_document = tokenizer_question(batch, return_tensors="pt", padding=True, truncation=True, max_length=30)["input_ids"].to(device)
        embeddings_query = model_question(input_ids_document).pooler_output
        embeddings_query_list.append(embeddings_query.detach().cpu().numpy())
    embeddings_query = np.vstack(embeddings_query_list)

    print("\n************************************ tf-idf & score ****************************************\n")

    score = np.matmul(embeddings_document, embeddings_query.T)
    # Score is of the shape num_docs * num_query
    scores_top = np.argsort(score, axis=0)[:100, :]
    scores_top = scores_top.T

    print("\n************************************ evaluation ****************************************")
    the_answer = read_text_from_json('answer_json/answer1.json')
    the_normalized_answer = for_normalize(the_answer)

    for k in [1, 5, 20, 100]:
        evaluation_list = for_evaluation(the_normalized_answer, the_normalized_document, scores_top, k)
        score = np.mean(evaluation_list) * 100
        print(f"Recall@{k} {score}")


device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer_context = DPRContextEncoderTokenizerFast.from_pretrained("voidful/dpr-ctx_encoder-bert-base-multilingual")
model_context = DPRContextEncoder.from_pretrained("voidful/dpr-ctx_encoder-bert-base-multilingual")
model_context.eval()
model_context = model_context.to(device)
tokenizer_question = DPRQuestionEncoderTokenizerFast.from_pretrained("voidful/dpr-question_encoder-bert-base-multilingual")
model_question = DPRQuestionEncoder.from_pretrained("voidful/dpr-question_encoder-bert-base-multilingual")
model_question.eval()
model_question = model_question.to(device)

main()
