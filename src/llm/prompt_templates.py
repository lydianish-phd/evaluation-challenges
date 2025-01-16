# Description: Contains the prompt templates for the LLM evaluation challenges.

LLAMA_MODEL_NAME = "Meta-Llama-3.1-8B-Instruct"

GENERAL_GUIDELINES = "Preserve the meaning, style and sentiment of the original text."

def get_llama_template(user_message, system_message="You are a helpful assistant"):
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
        f"{system_message}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>"
    )

def get_instruction(sentence, target_lang, standard=False, extra_guidelines=""):
    standard_level = "standard " if standard else ""
    return (
        f"Translate the following text to {standard_level}{target_lang}. "
        f"Give answer only. {extra_guidelines}:\n{sentence}"
        f"{sentence}"
    )

def get_prompt(sentence, target_lang, model_name=LLAMA_MODEL_NAME, guidelines="none"):
    if guidelines == "standard":
        prompt = get_instruction(sentence, target_lang, standard=True)
    elif guidelines == "general":
        prompt = get_instruction(sentence, target_lang, extra_guidelines=GENERAL_GUIDELINES)
    else: # default configuration (no guidelines)
        prompt = get_instruction(sentence, target_lang)
    if model_name == LLAMA_MODEL_NAME:
        return get_llama_template(prompt)
    return prompt
