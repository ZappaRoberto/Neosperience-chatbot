import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def model_fn(model_dir):
        
    # load model
    transformer = AutoModelForCausalLM.from_pretrained(model_dir)
    
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
    return transformer, tokenizer


def input_fn(data, content_type):
    
    data = json.loads(data)
    text = data['inputs']['text']
    past_user_inputs = data['inputs']['past_user_inputs']
    generated_responses = data['inputs']['generated_responses']
    
    return text, past_user_inputs, generated_responses


def predict_fn(data, model):
    
    transformer, tokenizer = model
    
    text, past_user_inputs, generated_responses = data

    chat_history = []
    chat_history = torch.LongTensor(chat_history)
    
    #create chat_history if generated_responses is empy
    if generated_responses:
                    
        for user_input,responses in zip(past_user_inputs,generated_responses):
            
            # tokenize the user_sentence
            user_sentence = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

            # append user input token to the chat history
            chat_history = torch.cat([chat_history, user_sentence], dim=-1)

            # tokenize the bot_sentence
            bot_sentence = tokenizer.encode(responses + tokenizer.eos_token, return_tensors='pt')

            # append the bot input token to the chat history
            chat_history = torch.cat([chat_history, bot_sentence], dim=-1)
            
    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')
    
    # append the new user input tokens to the chat history
    chat_history = torch.cat([chat_history, new_user_input_ids], dim=-1)
    
    # append the generated response to chat_history 
    chat_history = transformer.generate(chat_history, max_length=1000, do_sample=True, top_p=0.95, top_k=50, temperature=0.75, pad_token_id=tokenizer.eos_token_id).tolist()
    
    #decode all the chat_history
    chat_history = tokenizer.decode(chat_history[0])
    
    return chat_history


def output_fn(chat_history, accept):
    
    #inizialize data structure
    prediction = {"generated_text": "", 
                  "conversation": {
                      "past_user_inputs": [], 
                      "generated_responses": []
                      }
                 }
    
    chat_history = chat_history.split("<|endoftext|>")
    chat_history.pop(-1)
    
    # Using list slicing for separating odd and even index elements and update data structure
    prediction["conversation"]["past_user_inputs"] = chat_history[::2]
    prediction["conversation"]["generated_responses"] = chat_history[1::2]
    prediction["generated_text"] = chat_history[-1]
    
    return json.dumps(prediction)