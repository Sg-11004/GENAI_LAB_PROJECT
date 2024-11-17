from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set up generation parameters
input_prompt = "He was afraid of the people within , because the ones he trusts are the ones who could hurt him the most. He was not ready for the betrayal"
inputs = tokenizer(input_prompt, return_tensors="pt")
output = model.generate(**inputs, max_length=100, temperature=0.7, do_sample=True)

# Decode and print the generated text
story = tokenizer.decode(output[0], skip_special_tokens=True)
print(story)
