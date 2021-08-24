from transformers import BertTokenizer, BertConfig, BertModel

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
model = BertModel.from_pretrained("dbmdz/bert-base-german-uncased")

tokenizer.save_pretrained('C:/stss-2021-eval-master/pretrained-model/')
model.save_pretrained('C:/stss-2021-eval-master/pretrained-model/')