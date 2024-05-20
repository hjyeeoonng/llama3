from transformers import pipeline

transcriber = pipeline(
    model="openai/whisper-large-v2",
    generation_config={"language": "en"}
)
transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")