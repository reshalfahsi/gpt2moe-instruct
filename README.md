# **Instruction Fine-tuning of the GPT2MoE Model: GPT-2 with Mixture-of-Experts**


<div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/gpt2moe-instruct/blob/master/GPT2MoE_Instruct.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
</div>



<table>
    <tr>
        <td> 
            <img src="https://github.com/reshalfahsi/gpt2moe-instruct/blob/master/assets/gpt2moe-instruct-0.gif" alt="gpt2moe-instruct-0" > 
        </td>
        <td> 
            <img src="https://github.com/reshalfahsi/gpt2moe-instruct/blob/master/assets/gpt2moe-instruct-1.gif" alt="gpt2moe-instruct-1" > 
        </td>
        <td> 
            <img src="https://github.com/reshalfahsi/gpt2moe-instruct/blob/master/assets/gpt2moe-instruct-2.gif" alt="gpt2moe-instruct-2" > 
        </td>
    </tr>
    
</table>


<div align="center">

Some conversations with GPT2MoE.

</div>


## **Abstract**

Large Language Models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks. This project presents **GPT2MoE**, a pre‐trained GPT-2 model with Mixture-of-Experts (MoE) layers, replacing its original feed-forward networks with FFN<sub>SwiGLU</sub>-based experts. This project employs a transfer learning strategy where the original GPT-2 weights are initialized by loading the pre-trained weights, while the weights of the newly introduced MoE layers are trained from scratch. The model is then instruction fine-tuned on the Stanford Alpaca instruction‐tuning dataset using state-of-the-art frameworks, HuggingFace for model handling and PyTorch Lightning for the training infrastructure. This report details the model architecture, the transfer learning and fine-tuning methodology, and presents the experimental results.


## **Introduction**

The rapid advancements in Large Language Models (LLMs) have revolutionized natural language processing, enabling sophisticated applications ranging from text generation to complex reasoning. Models like the Transformer-based GPT-2 [1] have served as foundational architectures, demonstrating impressive capabilities through pre-training on vast text corpora. Standard Transformer architectures utilize dense feedforward networks (FFNs) in each layer, meaning every input token engages the entire FFN. Mixture-of-Experts (MoE) layers [2–4] provide an alternative by routing each token to a subset of "expert" networks. Also, FFN<sub>SwiGLU</sub> [5] is employed since it has become a de facto standard in state-of-the-art LLMs (e.g., LLaMA2 [6]). All in all, the core idea is to adapt a well-established base model (GPT-2) with modern architectural components (i.e., MoE and FFN<sub>SwiGLU</sub>), using a transfer learning paradigm where only the new components are trained from scratch. Furthermore, this modified, fine-tuned model is trained against the Alpaca dataset [7] to perform zero-shot and few-shot tasks guided by natural language instructions [8].


## **Related Works**

- **GPT-2**: A transformer‐based [10] language model with 124M parameters, pre‐trained on large web corpora [1].
- **Mixture-of-Experts**: Techniques such as GShard [3], Switch Transformer [4], Mixtral [9], and others route input dynamically to specialized experts.
- **FFN<sub>SwiGLU</sub>**: A feedforward network equipped with an activation function combining the Swish and GLU gates [4], shown to improve performance in LLMs like LLaMA2 [6].
- **Instruction Fine-Tuning**: Methods like Stanford Alpaca [7] and InstructGPT [8] refine LLMs to follow human instructions more reliably by supervised training on question–answer pairs.


## **Methodology**

### **Model Architecture**

This model is based on the pre-trained GPT-2 architecture. The core modification involves replacing the dense feedforward network (FFN) layers in each Transformer block with a Mixture-of-Experts layer.


#### **Mixture-of-Experts Layer with SwiGLU Experts**

Each replaced FFN layer is substituted with an MoE layer consisting of:

* **A Gating Network (Router):** This is typically a small neural network (e.g., a linear layer followed by a softmax or a gating function) that takes the token representation as input and outputs weights (or probabilities) for each expert. This project uses a top-k gating mechanism, routing each token to the top `2` experts.
* **Multiple Experts:** This project utilizes `8` independent expert networks. Each expert is a feedforward network employing the SwiGLU activation function.
    * FFN<sub>SwiGLU</sub> Formula: $\text{FFN}_\text{SwiGLU}(x, W_1, V, W_2) = (\text{Swish}_1(xW_1) \bigotimes x V ) W_2$
        * Where: 
            - $x$: input tensor
            - $W_1$: the fully-connected network of size $768 \times 256$ (Note: $768$ is the hidden or latent dimension of GPT-2)
            - $V$: the fully-connected network of size $768 \times 256$
            - $W_2$: the fully-connected network of size $256 \times 768$

The output of the MoE layer for a given token is a weighted sum of the outputs of the selected experts, where the weights are determined by the gating network.


### **Training Strategy: Transfer Learning**
This project employs a specific transfer learning approach:
1. **Base Model:** Load the pre-trained weights of the original GPT-2 model from HuggingFace Transformers.
2. **MoE Layer Initialization:** The weights of the newly introduced MoE layers (both the router and the experts) are **initialized from scratch** using standard initialization techniques similar to GPT-2.
3. **Training Objective:** All of the parameters are trainable. The training objective is to optimize these parameters with the GPT-2 components' weights are initialized from the pre-trained model.


### **Instruction Fine-Tuning**
The fine-tuning was performed as an instruction following task.
- **Input/Output Format:** The model was trained to generate the 'response' given the 'instruction' and 'input' fields from the Alpaca dataset, typically concatenated into a single input sequence with special tokens separating the components (e.g., ``"### Instruction:\n[instruction]\n\n### Input:\n[input]\n\n### Response:\n"``).
- **Dataset**: Stanford Alpaca ($52$ K instruction–response pairs).
- **Tokenizer**: Using GPT-2 BPE tokenizer with $50257$ vocab size.
- **Training Framework**: PyTorch Lightning for flexible multi-GPU support; HuggingFace Datasets and PyTorch Lightning DataModule for data handling. 
- **Loss Function:** Using the standard language modeling cross-entropy loss, calculated over the predicted tokens. Also, the load balancing loss is incorporated as suggested in [4].
- **Optimization**: 
  - AdamW optimizer with `learning_rate` = $6.9e-5$, `weight_decay` = $1e-2$, and `adam_epsilon` = $1e-8$ (also $420$ `warmup_steps` of the $3$-cycle cosine LR scheduler with hard restart). 
  - `batch_size` = $2$ sequences, `accumulate_gradient_batch_size` = $2$ sequences, and `max_length` = $1024$ tokens. 
  - Train for $12$ `epochs` with NVIDIA T4 GPU


### **Frameworks**

This project utilizes HuggingFace [11] and PyTorch Lightning [12] (also PyTorch [13]).

* **HuggingFace Transformers:** Used for loading the pre-trained GPT-2 model, its tokenizer, and potentially for implementing the custom MoE layer components, extending their `PreTrainedModel` classes.
* **PyTorch Lightning:** Used to structure the training loop, handle boilerplate like optimizer setup, learning rate scheduling, and device placement. This facilitated organized and reproducible experimentation.
* **HuggingFace Datasets and PyTorch Lightning DataModule:** for data handling.



## **Results**

### **Quantitative Evaluation**

The performance of the fine-tuned MoE-enhanced GPT-2 model is evaluated using the following metrics:


| Model         | Params | FLOPs/token | Token/sec. | PPL |
|---------------|------------|-------------|------------|---------------|
| **GPT2MoE**  | 138M | 139M | 34.069 | 71.736 |


- **Perplexity**: Calculated on the [HuggingFaceH4/instruction-dataset](https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset).
- **Inference Speed**: Measured in token/sec.
- **Compute Overhead**: Measured in FLOPs/token.


### **Qualitative Evaluation (Instruction Following Examples)**

#### **Non-Conversational**

To assess the model's instruction following capability, examples of generated text based on instructions from the [HuggingFaceH4/instruction-dataset](https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset) are provided.

##### **Example 1:**

* **Instruction:** 

```
On the basis of the subject of the email, determine whether the email should be treated as spam or not.
 Input: Dermatologists don't like her!
```

* **Model Output:**

```
Spam. The email should be treated as spam. The subject of the email is "Dermatologists", which means that the email should be treated as either a spam or not. The email should also be treated as a not. The recipient should be treated as a spam. The recipient should be treated as a not. The email should be treated as a spam. The email should be treated as a not. The recipient should be treated as a spam. The email should be
```

##### **Example 2:**

* **Instruction:**

```
Give me the plot for a silly short story using these three keywords as the influence: earth, aliens, orcas
```

* **Model Output:**

```
The story follows the journey of a young alien, a space mission that is both impossible and dangerous. The alien is driven by his desire to be a source of strength and courage, and to find a way to survive. Along the way, they face obstacles and obstacles that may affect their lives and the planet's future. On the other hand, the alien discovers that the planet is capable of its many incredible creatures, but they are determined to do whatever they make.
```

##### **Example 3:**

* **Instruction:**

```
List some interesting things to do in Idaho.
```

* **Model Output:**

```
Some interesting things to do in Idaho include exploring the local area, exploring the local city, playing art exhibitions, exploring the local wildlife, and visiting the local area. You should also research the local landscape, and be sure to research the local area before making a purchase. You should also take photographs, videos, and photographs that showcase the local nature and its unique culture. Finally, you should take a walk in the park and take a nearby cab. Finally, you should have a
```

##### **Example 4:**

* **Instruction:**

```
How far is the moon from the Earth?
```

* **Model Output:**

```
The moon from Earth is approximately 238.9 million miles (384,400 kilometers). It is estimated that it would take approximately 238 hours, making it much shorter than the Earth's closest point to the moon. The exact distance of the moon varies depending on the size of the Earth and the type of energy it emits. Additionally, the exact distance of the moon depends on the location of the Earth and the type of energy it produces. Generally, it is considered to be worth the
```

##### **Example 5:**

* **Instruction:**

```
I have these ingredients in my fridge: flour, eggs, milk, ham, and spinach. Generate a recipe for a meal I can make.
```

* **Model Output:**

```
1. Heat a large skillet over medium-high heat.
2. Add butter, eggs, and spinach to a skillet and cook for about 4 minutes.
3. Add the bacon, and cook for another 2 minutes.
4. Add the ham and cook for another 2 minutes.
5. Serve the bacon over cooked rice and serve with a side of your favorite sauce. Enjoy!
6. Enjoy your delicious meal!
7. Enjoy your delicious meal!
```

#### **Conversational**

This project also provides conversational results of the model.


##### **Live on Colab**

<p align="center"> <img src="https://github.com/reshalfahsi/gpt2moe-instruct/blob/master/assets/qualitative_colab.png" alt="qualitative_colab" > <br /> With conversational capabilities, GPT2MoE can remember past instructions. </p>

##### **Telegram Bot**

<p align="center"> <img src="https://github.com/reshalfahsi/gpt2moe-instruct/blob/master/assets/qualitative_telegram.gif" alt="qualitative_telegram" width=250> <br /> Conversing with GPT2MoE via Telegram Bot. It has conversational ability thanks to LangChain [14]. </p>


## **References**

1. [A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, and others, "Language models are unsupervised multitask learners," OpenAI blog, vol. 1, no. 8, p. 9, 2019.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2. [R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," *Neural Comput.*, vol. 3, no. 1, pp. 79–87, 1991.](https://www.cs.toronto.edu/~fritz/absps/jjnh91.pdf)
3. [D. Lepikhin et al., "Gshard: Scaling giant models with conditional computation and automatic sharding," *arXiv preprint arXiv:2006.16668*, 2020.](https://arxiv.org/pdf/2006.16668)
4. [W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *J. Mach. Learn. Res.*, vol. 23, no. 120, pp. 1–39, 2022.](https://arxiv.org/pdf/2101.03961)
5. [N. Shazeer, "Glu variants improve transformer," *arXiv preprint arXiv:2002.05202*, 2020.](https://arxiv.org/pdf/2002.05202)
6. [H. Touvron et al., "Llama 2: Open foundation and fine-tuned chat models," *arXiv preprint arXiv:2307.09288*, 2023.](https://arxiv.org/pdf/2307.09288)
7. R. Taori et al., "Alpaca: A strong, replicable instruction-following model," *Stanford Center for Research on Foundation Models*, vol. 3, no. 6, pp. 7, 2023. [Online]. Available: [https://crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)
8. [L. Ouyang et al., "Training language models to follow instructions with human feedback," in *Advances in Neural Information Processing Systems*, vol. 35, pp. 27730–27744, 2022.](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf)
9. [A. Q. Jiang et al., "Mixtral of experts," *arXiv preprint arXiv:2401.04088*, 2024.](https://arxiv.org/pdf/2401.04088)
10. [A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
11. [T. Wolf et al., "Huggingface's transformers: State-of-the-art natural language processing," *arXiv preprint arXiv:1910.03771*, 2019.](https://arxiv.org/pdf/1910.03771)
12. [W. Falcon and The PyTorch Lightning team, "PyTorch Lightning," 2019.](https://github.com/Lightning-AI/pytorch-lightning)
13. [A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems*, vol. 32, pp. 8026–8037, 2019.](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf)
14. H. Chase, "LangChain," GitHub, Oct. 2022. [Online]. Available: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)

