# app.py
"""
Gradio UI.
Left: questions + text input
Right: LLM response (markdown)

Run:
  python app.py
"""

import gradio as gr

from llm_pipeline import run_rag_analysis


QUESTIONS = [
    "Are you a part of a union?",
    "Do you feel you have been bullied/harassed by someone?",
    "Was that act/action coming from a co-worker?",
    "Do you think they were trying to humiliate you?",
    "Do you think they were trying to intimidate you?",
]

DISCRIMINATION_Q = "Was the described behavior connected to any of these?"
DISCRIMINATION_OPTIONS = [
    "Indigenous identity",
    "Gender",
    "Sex",
    "Religion",
    "Skin color",
    "Sexual orientation",
    "Political beliefs",
]

DETAILS_Q = "Describe your situation with as many details as possible"


# ----------------------------
# CSS: force uniform font size in LLM output
# ----------------------------
CUSTOM_CSS = """
#llm-output * {
    font-size: 14px !important;
    line-height: 1.5;
}
"""


def _merge_inputs(a1, a2, a3, a4, a5, selected, details) -> str:
    a1 = a1 or ""
    a2 = a2 or ""
    a3 = a3 or ""
    a4 = a4 or ""
    a5 = a5 or ""
    selected = selected or []
    details = details or ""

    lines = [
        f"{QUESTIONS[0]} -> {a1}",
        f"{QUESTIONS[1]} -> {a2}",
        f"{QUESTIONS[2]} -> {a3}",
        f"{QUESTIONS[3]} -> {a4}",
        f"{QUESTIONS[4]} -> {a5}",
        f"{DISCRIMINATION_Q} -> {', '.join(selected) if selected else ''}",
        f"{DETAILS_Q} -> {details}",
    ]
    return "\n".join(lines)


def on_submit(a1, a2, a3, a4, a5, selected, details):
    user_situation = _merge_inputs(a1, a2, a3, a4, a5, selected, details)
    answer = run_rag_analysis(user_situation)
    return answer


def on_reset():
    # Clear all inputs and the output
    return [None, None, None, None, None, [], "", ""]


with gr.Blocks(title="Harassment Wizard", css=CUSTOM_CSS) as demo:
    with gr.Row(equal_height=True):
        # LEFT PANEL
        with gr.Column(scale=1):
            gr.Markdown("### Questions")

            q1 = gr.Radio(["Yes", "No"], label=QUESTIONS[0], value=None)
            q2 = gr.Radio(["Yes", "No"], label=QUESTIONS[1], value=None)
            q3 = gr.Radio(["Yes", "No"], label=QUESTIONS[2], value=None)
            q4 = gr.Radio(["Yes", "No"], label=QUESTIONS[3], value=None)
            q5 = gr.Radio(["Yes", "No"], label=QUESTIONS[4], value=None)

            discrimination = gr.Dropdown(
                choices=DISCRIMINATION_OPTIONS,
                multiselect=True,
                label=DISCRIMINATION_Q,
                value=[],
            )

            details = gr.Textbox(
                label=DETAILS_Q,
                lines=10,
                placeholder="Type here…",
                value="",
            )

            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")

        # RIGHT PANEL
        with gr.Column(scale=2):
            gr.Markdown("### Result")
            output_md = gr.Markdown(value="", elem_id="llm-output")

    submit_btn.click(
        fn=on_submit,
        inputs=[q1, q2, q3, q4, q5, discrimination, details],
        outputs=[output_md],
    )

    reset_btn.click(
        fn=on_reset,
        inputs=[],
        outputs=[q1, q2, q3, q4, q5, discrimination, details, output_md],
    )


if __name__ == "__main__":
    demo.launch()


