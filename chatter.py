from transformers import BlenderbotForConditionalGeneration, BlenderbotSmallTokenizer


mname = 'facebook/blenderbot-90M'
model = BlenderbotForConditionalGeneration.from_pretrained(mname)
tokenizer = BlenderbotSmallTokenizer.from_pretrained(mname)


def say_something(text):
    inputs = tokenizer([text], return_tensors='pt')
    reply_ids = model.generate(**inputs)
    data = [tokenizer.decode(g,
                             skip_special_tokens=True,
                             clean_up_tokenization_spaces=True) for g in reply_ids][0]
    return data

