
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np
from vncorenlp import VnCoreNLP
from bert_crf import CustomNERCRF
import transformers

print("Transformers version {}".format(transformers.__version__))


# model = AutoModelForTokenClassification.from_pretrained(
#     "test-ner/checkpoint-6000")

ner_tags = ('B-AGE',
            'B-DATE',
            'B-GENDER',
            'B-JOB',
            'B-LOCATION',
            'B-NAME',
            'B-ORGANIZATION',
            'B-PATIENT_ID',
            'B-SYMPTOM_AND_DISEASE',
            'B-TRANSPORTATION',
            'I-AGE',
            'I-DATE',
            'I-GENDER',
            'I-JOB',
            'I-LOCATION',
            'I-NAME',
            'I-ORGANIZATION',
            'I-PATIENT_ID',
            'I-SYMPTOM_AND_DISEASE',
            'I-TRANSPORTATION',
            'O')

names = sorted(list(ner_tags))
id2label = dict(list(enumerate(names)))
label2id = {v: k for k, v in id2label.items()}
# model = CustomNERCRF.load_from_checkpoint(
#     checkpoint_path="test-ner/BestModelcrf-ep-20-lr-5e-5-bs-32-10warmup-dataaug-final.pth")
model = CustomNERCRF.load_from_checkpoint("test-ner/new_model.pth")
# tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('test-ner/covid19-tokenizer')
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
                         annotators="wseg", max_heap_size='-Xmx500m')


model.config.id2label = id2label
model.config.label2id = label2id
model.eval()


# tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
# rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
#                          annotators="wseg", max_heap_size='-Xmx500m')


def get_annotation(text):
    def get_label(x):
        return model.config.id2label[x]

    def convert_ids_to_string(ids):
        return tokenizer.convert_tokens_to_string(tokenizer._convert_id_to_token(ids))

    sentences = rdrsegmenter.tokenize(text)
    sentences = [" ".join(sentence) for sentence in sentences]
    input_ids = tokenizer.batch_encode_plus(
        sentences, return_tensors='pt', padding='longest')

    with torch.no_grad():
        output = model.forward(**input_ids)
    kqua = np.array(output['logits'].argmax(dim=2))
    kqua = np.vectorize(get_label)(kqua)

    input_ids_np = input_ids['input_ids'].numpy()
    words = np.vectorize(convert_ids_to_string)(input_ids_np)

    return_list = []

    for s in range(len(kqua)):
        for w in range(len(kqua[s])):
            if words[s, w] not in ("<s>", "<pad>", "</s>"):
                if kqua[s, w] != "O":
                    return_list.append((words[s, w], kqua[s, w]))
                else:
                    return_list.append(words[s, w])

    return return_list


def get_annotation_lightning(text):
    def get_label(x):
        return model.config.id2label[x]

    def convert_ids_to_string(ids):
        return tokenizer.convert_tokens_to_string(tokenizer._convert_id_to_token(ids))

    sentences = rdrsegmenter.tokenize(text)
    sentences = [" ".join(sentence) for sentence in sentences]
    input_ids = tokenizer.batch_encode_plus(
        sentences, return_tensors='pt', padding='longest')

    with torch.no_grad():
        output = model.forward(**input_ids)
    # kqua = np.array(output['logits'].argmax(dim=2))
    print(output['y_pred'])
    kqua = np.array(output['y_pred'])
    kqua = np.vectorize(get_label)(kqua)

    input_ids_np = input_ids['input_ids'].numpy()
    words = np.vectorize(convert_ids_to_string)(input_ids_np)

    return_list = []

    for s in range(len(kqua)):
        for w in range(len(kqua[s])):
            if words[s, w] not in ("<s>", "<pad>", "</s>"):
                if kqua[s, w] != "O":
                    return_list.append((words[s, w], kqua[s, w]))
                else:
                    return_list.append(words[s, w])

    return return_list
