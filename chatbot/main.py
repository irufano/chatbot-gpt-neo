from transformers import GPTNeoForCausalLM, GPT2Tokenizer

import os


def check_model_exists(model_dir):
    # Define possible paths to model files
    safetensor_path = os.path.join(model_dir, "model.safetensors")
    config_path = os.path.join(model_dir, "config.json")
    tokenizer_path = os.path.join(
        model_dir, "vocab.json"
    )  # For GPT-Neo, tokenizer data is stored here

    # Check if either the safetensors or bin file exists along with the config and tokenizer files
    if (
        os.path.exists(model_dir)
        and os.path.exists(safetensor_path)
        and os.path.exists(config_path)
        and os.path.exists(tokenizer_path)
    ):
        return True
    else:
        return False


# Specify the local directory where the model should be saved
model_directory = "./models/gpt-neo-125m"
# model_directory = "./models/gpt-neo-1.3B"

# Check if the model exists
if check_model_exists(model_directory):
    print("Model exists locally.")

    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

    # Load the model specifying 'safetensors' as the weights format
    model = GPTNeoForCausalLM.from_pretrained(model_directory)

    # # Define the prompt
    # prompt = "good morning"

    # # Encode the input text
    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # # Generate text
    # outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # # Decode the output
    # generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # print(generated_text)

    def generate_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def chat():
        print("Chatbot (type 'quit' to exit)")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            response = generate_response(user_input)
            print("Bot:", response)

    if __name__ == "__main__":
        chat()
else:
    print("Model does not exist locally.")

# from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# # Load the tokenizer
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# # Load the model
# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# # Save model
# model.save_pretrained("./gpt-neo-1.3B")
# tokenizer.save_pretrained("./gpt-neo-1.3B")

# # Define the prompt
# prompt = "1, 100, 90, 40 which is higher?"

# # Encode the input text
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# # Generate text
# outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)

# # Decode the output
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# print(generated_text)
