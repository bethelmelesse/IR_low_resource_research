import json

IN_DATA_PATH = 'AmQA_Dataset.json'
OUT_QUERY_PATH = 'query_json/query1.json'
OUT_ANS_PATH = 'answer_json/answer1.json'


def write_text_in_json(text, file_name):
    with open(file_name, 'w', encoding='utf8') as f:
        json.dump(text, f, ensure_ascii=False)
    return file_name


def read_text_from_json(text_file):
    with open(text_file) as f:
        text_to_read = json.load(f)
    return text_to_read

def main():
    input_dataset = read_text_from_json(IN_DATA_PATH)
    input_dataset = input_dataset['data']
    questions = []
    answers = []
    for value in input_dataset:
        value = value['paragraphs']
        for paragraph in value:
            qas_list = paragraph['qas']
            for qas in qas_list:
                questions.append(qas['question'])
                answer = qas['answers']
                if len(answer) > 1:
                    print('AAAAAAAAAAAAAAAAAAAAAAa')
                    exit()
                answer = answer[0]
                answers.append(answer['text'])

    print('Converted Dataset, Writing to file')
    write_text_in_json(questions, OUT_QUERY_PATH)
    write_text_in_json(answers, OUT_ANS_PATH)

    print('Finished!')

main()