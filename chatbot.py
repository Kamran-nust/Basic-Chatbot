
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "facebook/blenderbot-400M-distill"
# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# There are several things you'll do to have an effective conversation with your chatbot.
# Before interacting with your model, you need to initialize an object where you can store your conversation history.
#     Initialize object to store conversation history
# Afterward, you'll do the following for each interaction with the model:
    #     Encode conversation history as a string
    #     Fetch prompt from user
    #     Tokenize (optimize) prompt
    #     Generate output from the model using prompt and history
    #     Decode output
    #     Update conversation history
conversation_history=[]
while True:
    history_string= "\n".join(conversation_history)

    input_text =input("> ")

    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    #print(inputs)

    tokenizer.pretrained_vocab_files_map

    outputs = model.generate(**inputs)
    #print(outputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)

    conversation_history.append(input_text)
    conversation_history.append(response)
    #print(conversation_history)
