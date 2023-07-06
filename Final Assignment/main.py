import numpy as np
import gradio as gr
from recommender import Recommender

css="""
.gradio-row {
  flex-wrap: nowrap !important;
}

.btn {
  height: 50px !important;
  max-height: 50px !important
}
"""

# Create recommender object
recommender = Recommender()
initial_picks = recommender.get_descs_for_recommended(recommender.recommended_ids)

with gr.Blocks(css=css) as demo:
    gr.Markdown("# **Full report and code can be found here: [GitHub](https://github.com/SSBakh07/Statistical-ML---Spring-2023)**")
    gr.Markdown("## Basic Movie Recommender")
    with gr.Row(variant='compact', elem_classes="gradio-row", equal_height=True):

      # First Component
      with gr.Column(scale=1):
        col_1_number = gr.Number(value=1, visible=False)
        
        gr.Markdown("Based on similar movies...")

        movie_title_1 = gr.Textbox(initial_picks[0]['title'], label="Movie Title")
        movie_summary_1 = gr.Textbox(initial_picks[0]['overview'], label="Movie Summary")

        
        with gr.Column():
          gr.Markdown("How much did you enjoy this movie or how interested are you?")
          slider_1 = gr.Slider(minimum=1, maximum=5, editable=True)
          btn_submit_1 = gr.Button("Submit", elem_classes="btn")



      # Second Component
      with gr.Column(scale=1):
        col_2_number = gr.Number(value=2, visible=False)
        
        gr.Markdown("Based on similar users...")

        movie_title_2 = gr.Textbox(initial_picks[1]['title'], label="Movie Title")
        movie_summary_2 = gr.Textbox(initial_picks[1]['overview'], label="Movie Summary")

        
        with gr.Column():
          gr.Markdown("How much did you enjoy this movie or how interested are you?")
          slider_2 = gr.Slider(minimum=1, maximum=5, editable=True)
          btn_submit_2 = gr.Button("Submit", elem_classes="btn")



      # Third Component
      with gr.Column(scale=1):
        col_3_number = gr.Number(value=3, visible=False)

        gr.Markdown("Based on similar users and movies...")

        movie_title_3 = gr.Textbox(initial_picks[2]['title'], label="Movie Title")
        movie_summary_3 = gr.Textbox(initial_picks[2]['overview'], label="Movie Summary")


        with gr.Column():
          gr.Markdown("How much did you enjoy this movie or how interested are you?")
          slider_3 = gr.Slider(minimum=1, maximum=5, editable=True)
          btn_submit_3 = gr.Button("Submit", elem_classes="btn")



    #Handler functions    
    def submit_opinion(number, value):
      global recommender
      res = recommender.on_pick(int(number), value)
      text_res = recommender.get_descs_for_recommended(res)
      final = []
      for txt in text_res:
        final.append(txt['title'])
        final.append(txt['overview'])
      return final
    
    
    # Attach buttons to functions
    submit_outputs = [movie_title_1, movie_summary_1, movie_title_2, 
                      movie_summary_2, movie_title_3, movie_summary_3]
    btn_submit_1.click(submit_opinion, [col_1_number, slider_1], submit_outputs)
    btn_submit_2.click(submit_opinion, [col_2_number, slider_2], submit_outputs)
    btn_submit_3.click(submit_opinion, [col_3_number, slider_3], submit_outputs)
    


demo.launch(debug=True)