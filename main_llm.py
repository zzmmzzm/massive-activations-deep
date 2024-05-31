import argparse
import os
from importlib.metadata import version

import numpy as np
import torch

import lib
import monkey_patch as mp

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument('--seed',type=int, default=1, help='Seed for sampling the calibration data.')
    parser.add_argument("--revision", type=str, default="main")

    parser.add_argument("--exp1", action="store_true", help="plot 3d feature")
    parser.add_argument("--exp2", action="store_true", help="layerwise analysis")
    parser.add_argument("--exp3", action="store_true", help="intervention analysis")
    parser.add_argument("--exp4", action="store_true", help="attention visualization")
    parser.add_argument("--layer_id", type=int, default=1)
    parser.add_argument("--reset_type", type=str, default="set_mean")
    parser.add_argument("--access_token", type=str, default="type in your access token here")
    parser.add_argument("--savedir", type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    print("use device ", device)

    if args.exp1: ### visualize the output feature of a layer in LLMs
        layer_id = args.layer_id - 1
        if "llama2" in args.model:
            mp.enable_llama_custom_decoderlayer(layers[layer_id], layer_id)
        elif "mistral" in args.model:
            mp.enable_mistral_custom_decoderlayer(layers[layer_id], layer_id)
        elif "phi-2" in args.model:
            mp.enable_phi2_custom_decoderlayer(layers[layer_id], layer_id)
        else:
            raise ValueError(f"model {args.model} not supported")

        stats = {}
        sequences = [
            ".\t .\n .\n Summer is warm. Winter is cold.",
            ".\n .\t .\n Leaves turn yellow in autumn.",
            "Spring flowers .\n .\n .\n bloom brightly.",
            "Spring flowers . .\n .\n bloom brightly.",
            "Spring flowers . bloom brightly.",
            "Spring flowers .\n bloom brightly.",
            "Spring flowers . .\n bloom brightly.",
            "Spring flowers . . .\n bloom brightly.",
            "Spring flowers . . . \n bloom brightly.",
            "Spring flowers \n bloom brightly.",
            "Spring flowers \n \n bloom brightly.",
            "Spring flowers \n \n .\n bloom brightly.",
            "Spring flowers . \n bloom brightly.",
            "Spring flowers .\n .\n bloom brightly.",
            "Spring . flowers .\n .\n .\n bloom brightly.",
            "The uncertainty principle suggests. that it's .\n impossible to simultaneously know the exact position and momentum of a particle.",
            "The uncertainty principle suggests. that it's impossible to simultaneously .\n know the exact position and momentum of a particle.",
            "The uncertainty principle suggests that it's impossible to simultaneously know the exact position and momentum of a particle.",
            "The uncertainty principle suggests that it's impossible to simultaneously know the exact position and .\n momentum of a particle.",
            "The uncertainty principle suggests that it's \n impossible to simultaneously know the exact position and momentum of a particle.",
            "The uncertainty principle suggests that it's impossible to simultaneously \n know the exact position and momentum of a particle.",
            "The uncertainty principle suggests. that it's impossible to simultaneously know the exact position and momentum of a particle.\n",
            " I have 8 demos for you:Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. So the answer is :39\\\\Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. So the answer is :33\\\\Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. So the answer is :8\\\\Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. So the answer is :5\\\\Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. So the answer is :6\\\\Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. So the answer is :29\\\\Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. So the answer is :8\\\\Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. So the answer is :9\\\\And here is the question for you:Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? You can think step by step.First tell me the reasoning steps. Then tell me the answer."

        ]

        seq = "Summer is warm. Winter is cold."
        valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

        with torch.no_grad():
            model(valenc)

        seq_decoded = []
        for i in range(valenc.shape[1]):
            seq_decoded.append(tokenizer.decode(valenc[0,i].item()))

        stats[f"seq"] = seq_decoded
        feat_abs = layers[layer_id].feat.abs()

        stats[f"{layer_id}"] = feat_abs

        lib.plot_3d_feat(stats, layer_id, args.model, args.savedir)
        
        ### 循环
        for seq_index, seq in enumerate(sequences):
            if "llama2" in args.model:
                mp.enable_llama_custom_decoderlayer(layers[layer_id], layer_id)
            elif "mistral" in args.model:
                mp.enable_mistral_custom_decoderlayer(layers[layer_id], layer_id)
            elif "phi-2" in args.model:
                mp.enable_phi2_custom_decoderlayer(layers[layer_id], layer_id)
            else:
                raise ValueError(f"model {args.model} not supported")

            stats = {}
            valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

            with torch.no_grad():
                model(valenc)

            seq_decoded = []
            for i in range(valenc.shape[1]):
                seq_decoded.append(tokenizer.decode(valenc[0, i].item()))

            stats[f"seq"] = seq_decoded
            feat_abs = layers[layer_id].feat.abs()

            stats[f"{layer_id}"] = feat_abs

            # Save the plot with a unique name for each sequence
            save_path = os.path.join(args.savedir, f"feature_visualization_0")
            lib.plot_3d_feat_index(stats, layer_id, args.model, save_path, seq_index)

    elif args.exp2: ### visualize the layerwise top activation magnitudes
        for layer_id in range(len(layers)):
            layer = layers[layer_id]
            if "llama2" in args.model:
                mp.enable_llama_custom_decoderlayer(layer, layer_id)
            elif "mistral" in args.model:
                mp.enable_mistral_custom_decoderlayer(layer, layer_id)
            elif "phi-2" in args.model:
                mp.enable_phi2_custom_decoderlayer(layers[layer_id], layer_id)
            else:
                raise ValueError(f"model {args.model} not supported")

        testseq_list = lib.get_data(tokenizer, nsamples=10, seqlen=seq_len, device=device)

        stats = []
        for seqid, testseq in enumerate(testseq_list):
            print(f"processing seq {seqid}")
            with torch.no_grad():
                model(testseq)

            seq_np = np.zeros((4, len(layers)))
            for layer_id in range(len(layers)):
                feat_abs = layers[layer_id].feat.abs()
                sort_res = torch.sort(feat_abs.flatten(), descending=True)
                seq_np[:3, layer_id] = sort_res.values[:3]
                seq_np[3, layer_id] = torch.median(feat_abs)

            stats.append(seq_np)

        lib.plot_layer_ax(stats, args.model, args.savedir)

    elif args.exp3: ### intervention analysis
        layer = layers[args.layer_id-1]
        lib.setup_intervene_hook(layer, args.model, args.reset_type)

        f = open(os.path.join(args.savedir, f"{args.model}_{args.reset_type}.log"), "a")

        ds_list = ["wikitext", "c4", "pg19"]
        res = {}
        for ds_name in ds_list:
            ppl = lib.eval_ppl(ds_name, model, tokenizer, args.seed, device)
            res[ds_name] = ppl 
            print(f"{ds_name} ppl: {ppl}", file=f, flush=True)

    elif args.exp4:
        layer_id = args.layer_id - 1
        if "llama2" in args.model:
            modified_attn_layer = mp.enable_llama_custom_attention(layers[layer_id], layer_id)
        elif "mistral" in args.model:
            modified_attn_layer = mp.enable_mistral_custom_attention(layers[layer_id], layer_id)
        elif "phi-2" in args.model:
            modified_attn_layer = mp.enable_phi2_custom_attention(layers[layer_id], layer_id)
        else:
            raise ValueError(f"model {args.model} not supported")

        seq = "The following are multiple choice questions (with answers) about machine learning.\n\n A 6-sided die is rolled 15 times and the results are: side 1 comes up 0 times;"
        valenc = tokenizer(seq, return_tensors='pt', add_special_tokens=False).input_ids.to(device)

        with torch.no_grad():
            model(valenc)

        attn_logit = layers[layer_id].self_attn.attn_logits.detach().cpu()
        lib.plot_attn(attn_logit, args.model, layer_id, args.savedir)