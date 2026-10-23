# BlenderBot Fine-Tuning on DailyDialog

A conversational AI chatbot built by fine-tuning the pretrained `facebook/blenderbot-400M-distill` model on the [DailyDialog](https://huggingface.co/datasets/daily_dialog) dataset using Hugging Face Transformers and PyTorch.

The project trains BlenderBot to generate contextually relevant responses to user messages and provides an interactive command-line interface for chatting with the fine-tuned model.

## Features

- **Pretrained model:** Uses BlenderBot 400M Distill.
- **Conversational dataset:** Trains on DailyDialog.
- **Fine-tuning:** Uses the Hugging Face `Trainer` API.
- **Response generation:** Generates replies using beam search.
- **Conversation history:** Maintains a running conversation context.
- **Model persistence:** Saves the fine-tuned model and tokenizer locally.
- **Evaluation:** Calculates the BLEU score on validation samples.
- **Hardware support:** Supports CUDA-enabled GPUs or CPU execution.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Hugging Face Evaluate
- BlenderBot

## Project Structure

After running the script, the project directory may look like this:

```text
.
├── README.md
├── Chat_Bot.py
├── blenderbot-dailydialog/
│   └── ... (training checkpoints)
├── blenderbot-dailydialog-finetuned/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors (or pytorch_model.bin)
│   └── tokenizer files
└── logs/
```

*Note: The model files and checkpoint structure can vary depending on the Transformers version.*

## Requirements

- Python 3.10 or newer is recommended.
- PyTorch
- Transformers
- Datasets
- Evaluate
- SacreBLEU, if required by the BLEU evaluation dependencies
- Sufficient RAM and disk space for the model and dataset

A CUDA-compatible GPU is recommended to speed up training, but CPU execution is also supported.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Replace the placeholders with your repository URL and directory name.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install torch transformers datasets evaluate sacrebleu
```

For GPU training, install the PyTorch build appropriate for your CUDA environment from the [official PyTorch installation page](https://pytorch.org/get-started/locally/).

## Dataset

This project uses the DailyDialog dataset, a collection of multi-turn dialogues covering everyday conversational topics.

The dataset is loaded through Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("daily_dialog")

train_data = dataset["train"]
val_data = dataset["validation"]
```

Each dialogue is converted into consecutive input-target pairs. For example:

| Input | Target |
|---|---|
| Hello! How are you? | I'm fine, thank you. |
| What are you doing? | I'm reading a book. |

The model learns to predict the next utterance from the previous utterance. The current implementation uses consecutive turns rather than the full dialogue history during training.

## Model

The pretrained model is [`facebook/blenderbot-400M-distill`](https://huggingface.co/facebook/blenderbot-400M-distill).

Load the tokenizer and model with:

```python
from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
)

model_name = "facebook/blenderbot-400M-distill"

tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
```

The model is fine-tuned on DailyDialog to adapt its response generation to the dataset's conversational examples.

## Training Configuration

The current script uses the following configuration:

| Parameter | Value |
|---|---|
| Base model | `facebook/blenderbot-400M-distill` |
| Number of epochs | 2 |
| Training batch size per device | 8 |
| Maximum input length | 128 tokens |
| Maximum target length | 64 tokens |
| Checkpoint saving | Every epoch |
| Maximum saved checkpoints | 2 |
| Logging interval | Every 100 steps |
| Mixed precision (`fp16`) | Disabled |

These values can be adjusted according to available hardware and training requirements.

### Start Training

Run the script:

```bash
python Chat_Bot.py
```

The script performs the following steps:

1. Loads the DailyDialog dataset.
2. Creates consecutive dialogue input-target pairs.
3. Loads the pretrained BlenderBot tokenizer and model.
4. Tokenizes the training and validation datasets.
5. Fine-tunes the model using Hugging Face Trainer.
6. Saves the trained model and tokenizer.
7. Evaluates generated responses using BLEU.
8. Loads the saved model and starts an interactive chat session.

The dataset and pretrained model may need to be downloaded from Hugging Face on the first run, so an internet connection is required unless the files are already cached.

## Saved Model

The fine-tuned model and tokenizer are saved to:

```text
./blenderbot-dailydialog-finetuned
```

Training checkpoints are stored in:

```text
./blenderbot-dailydialog
```

The saved model can be loaded later without repeating training:

```python
import torch
from transformers import (
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
)

model_path = "./blenderbot-dailydialog-finetuned"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BlenderbotTokenizer.from_pretrained(model_path)
model = BlenderbotForConditionalGeneration.from_pretrained(model_path)
model.to(device)
model.eval()
```

## Evaluation

The script evaluates generated responses using BLEU, which measures word and phrase overlap between generated text and reference responses.

The current implementation evaluates 100 examples from the validation dataset and prints the resulting score:

```text
BLEU score on validation set: <score>
```

The displayed value depends on the trained model and evaluation data. BLEU is useful for comparison, but it does not fully measure conversational relevance, fluency, or coherence.

**Evaluation note:** Before relying on the score, mask padding tokens in the labels and remove padding from decoded reference text. Also ensure the requested number of evaluation samples does not exceed the validation dataset length.

## Interactive Chat

After training, `Chat_Bot.py` loads the saved model and starts a command-line conversation.

Example (illustrative; actual responses will vary):

```text
Start chat with model (type 'exit' to exit):
You: Hello! How are you?
Chat_Bot: I'm doing well, thank you. How about you?
You: What do you like to do?
Chat_Bot: I enjoy talking to people.
You: exit
End Of Chat.
```

Type `exit` to end the conversation. The script appends user messages and generated responses to a running history, which is truncated to the configured maximum input length when tokenized.

## Hardware and Performance

The script automatically selects a CUDA GPU when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Fine-tuning a model with approximately 400 million parameters requires substantial memory and computation. If you encounter out-of-memory errors, consider reducing the batch size, using dynamic padding, or enabling supported mixed-precision training on compatible hardware.

## Limitations and Potential Improvements

- **Limited training context:** Training examples use consecutive utterance pairs rather than complete dialogue contexts.
- **Long conversation history:** Earlier turns may be truncated when the input exceeds the maximum token length.
- **Padding and loss:** Target padding tokens should be replaced with `-100` so they are ignored when computing training loss.
- **Evaluation:** BLEU alone is not sufficient to assess open-ended dialogue quality.
- **Generation settings:** Beam search may produce generic or repetitive responses. Sampling strategies such as `top_p` and `temperature` could also be tested.
- **Reproducibility:** Setting random seeds can make training and evaluation more reproducible.

## Future Improvements

- Train using full dialogue history.
- Add evaluation metrics such as ROUGE or BERTScore.
- Compare beam search with sampling-based generation.
- Build a web interface using Gradio or Streamlit.
- Add conversation reset and history management.
- Make generation settings configurable.
- Log training metrics and evaluation results.

## License

Review the licenses and usage terms of the pretrained BlenderBot model, the DailyDialog dataset, and all dependencies before redistributing the model or deploying the application.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [DailyDialog Dataset](https://huggingface.co/datasets/daily_dialog)
- [BlenderBot 400M Distill](https://huggingface.co/facebook/blenderbot-400M-distill)

---

**Project goal:** Fine-tune a pretrained conversational language model on DailyDialog and provide an interactive chatbot that generates responses to user messages.
