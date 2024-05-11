import time
from typing import Dict, Union

import gradio as gr
from gliner import GLiNER

model = GLiNER.from_pretrained("numind/NuZero_token")


def merge_entities(entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity["entity"] == current["entity"] and (
            next_entity["start"] == current["end"] + 1
            or next_entity["start"] == current["end"]
        ):
            current["word"] += " " + next_entity["word"]
            current["end"] = next_entity["end"]
        else:
            merged.append(current)
            current = next_entity
    merged.append(current)
    return merged


def ner(
    text, labels: str, threshold: float, nested_ner: bool
) -> Dict[str, Union[str, int, float]]:
    labels = labels.split(",")
    start = time.time()
    r = {
        "text": text,
        "entities": [
            {
                "entity": entity["label"],
                "word": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
                "score": 0,
            }
            for entity in model.predict_entities(
                text, labels, flat_ner=not nested_ner, threshold=threshold
            )
        ],
    }
    r["entities"] = merge_entities(r["entities"])
    print(time.time() - start)
    return r


with gr.Blocks(title="NER for HR 4.0") as demo:
    gr.Markdown(
        """
        # Extraction infor HR. 4.0
        """
    )
    input_text = gr.Textbox(label="Text input", placeholder="Enter your text here")
    with gr.Row() as row:
        labels = gr.Textbox(
            label="Labels",
            placeholder="Enter your labels here (comma separated)",
            scale=2,
        )
        threshold = gr.Slider(
            0,
            1,
            value=0.5,
            step=0.01,
            label="Threshold",
            info="Lower the threshold to increase how many entities get predicted.",
            scale=1,
        )
        nested_ner = gr.Checkbox(
            value=False,
            label="Nested NER",
            info="Allow for nested NER?",
            scale=0,
        )
    output = gr.HighlightedText(label="Predicted Entities")
    submit_btn = gr.Button("Submit")

    # Submitting
    input_text.submit(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    labels.submit(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    threshold.release(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    submit_btn.click(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )
    nested_ner.change(
        fn=ner, inputs=[input_text, labels, threshold, nested_ner], outputs=output
    )

demo.queue()
demo.launch(debug=True)
