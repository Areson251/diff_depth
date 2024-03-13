from scripts.gradio import window


if __name__ == "__main__":
    window.demo.launch(share=True)
    window.demo.close()