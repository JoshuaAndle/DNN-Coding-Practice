### Package Libraries
import torch
import torchvision
import torch.nn as nn
from typing import Union, List
from packaging import version


### Project Libraries
import configs
import datasets
import models

### Use a pre-existing tokenizer+vocabulary for library version of code
from transformers import CLIPTokenizer




def tokenize(tokenizer, texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result








def main():
    args = configs.args_parser()

    ### Set up dataset 
    ### Note: Not worrying about validation split in this toy project
    train_dataset = datasets.prepare_dataset(args.dataset, location=args.dataset_location, train=True)
    test_dataset = datasets.prepare_dataset(args.dataset, location=args.dataset_location, train=False)

    train_dataloader = datasets.prepare_dataloader(train_dataset, batch_size=args.batch_size, train=True)
    test_dataloader = datasets.prepare_dataloader(test_dataset, batch_size=args.batch_size, train=False)


    ### Set up tokenizer and model
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    context_length = 20
    
    model = models.Minimal_VLM( 
                            output_dim=256,
                            img_size=32, patch_size=4, vision_width=256,             
                            vocab_size=49152, context_length=context_length, transformer_width=256
                        )

    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    prompt = "image of {}"
    
    class_prompts = [prompt.format(x) for x in class_names]
    
    print(class_prompts)
    tokens = tokenize(tokenizer, class_prompts, context_length=context_length)
    print(tokens)

    ### Set up optimizer

    ### Set up loss function

    ### Set up learning rate scheduler

    ### Apply training loop





    print("main() run complete")














if __name__ == '__main__':
    main()