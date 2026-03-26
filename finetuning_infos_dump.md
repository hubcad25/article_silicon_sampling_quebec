https://www.superannotate.com/blog/llm-fine-tuning

https://unsloth.ai/docs/get-started/fine-tuning-llms-guide

 Hi!

I’ve been working on fine-tuning LLMs a bit later than everyone else (among the ones I know), and I’ve struggled to understand why I’m doing what I’m doing. I’ve compiled a small collection of everything I know about fine-tuning LLMs or transformer models for specific use cases. I’d like to hear your thoughts on these things!

Also, please share your experiences too! I'd love to hear those even more.

---------------------------------------

When you shouldn't fine-tune:
- When wanting the model to respond in a "specific" way in rare circumstances. That's what prompt engineering is for! Don't use a bulldozer to kill a fly.
- For the model to learn "new knowledge"
- When you have too little data. (Though it's being disproven that low data performs better than high data for mathematical reasoning! Still in research!)

Choosing the right data

    You want the model to learn the patterns, not the words. You need enough diverse samples, not large data of the same kind.

    More data isn't always better. Don't dump all the data you have onto the model.

    Every training example needs a clear input and a clear output. And optionally, context text to add additional information.

    The dataset must have enough cases, edge cases and everything in between. You can also augment the dataset by using data from a Larger LLM.

    Pack your datasets! They help!

    Determine if you're performing open-ended, Instruction or chat-based text generation**.**

Choosing the right model:

    You don't need a 100B model for every task you have. For real-world applications, 1-13B models are more practical.

    You must check the licensing to see if you use the model for commercial use cases. Some have very strict licensing.

    A good starting point? Llama-3.1-8B.

General fine-tuning:

    An 8B model needs ~16GB of memory to load up. So, mixed precision and quantisations are used to initialise a model in case of memory restrictions.

    If the batch size can't be increased, use the Gradient-accumulation approach. General accumulations are done for overall batch sizes of 16,32,128.

    Save checkpoints regularly, and use resume_from_checkpoint=True when needed.

    Consider using Model-parallelism or Data-parallelism techniques to work across multiple devices for large-scale training.

    Documentation will help in surprisingly weird situations. Maintain it.

LoRA finetuning:

    Don't use QLoRA for everything. Use it only if you realise that the model couldn't fit your device. Using QLoRA roughly comes with 39% more training time while saving roughly a third of the memory needed.

    SGD+Learning rate schedulers are useful. But using LR Schedulers with other optimizers like AdamW/Adam seems to give diminishing returns. (need to check sophia optimiser.)

    A high number of training epochs doesn't bode well for LoRA finetuning.

    Despite the general understanding of lora_alpha ~2*lora_rank, it's sometimes better to check with other values too! These two parameters need meticulous adjustments.

    The training times found outside might be confusing. It would take too long on your PC, but it seems very fast on the reported sites. Well, your choice of GPU would also be implicating the speed. So keep that in mind.

    LoRA is actively changing. Don't forget to check and test its different versions, such as LoRA-plus, DoRA, LoFTQ, AdaLoRA, DyLoRA, LoRA-FA etc. (still need to check many of these...)

Choosing the finetuning strategy:

    Determine the right task:

        You must "adapt" the model for task-specific finetuning, such as code generation, document summarisation, and question answering.

        For domain-specific needs like medical, financial, legal, etc., you need to push the model to update its knowledge => Use RAG when applicable or fine-tune the entire model. (EDIT: This is supposed to be re-training, not fine-tuning.)

    Utilise pruning depending on the kind of task you're trying to perform. Generally, in production environments, the faster the inference, the better the performance. In this case, pruning+finetuning helps. We need to keep that in mind.
