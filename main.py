from scripts.gradio import window


if __name__ == "__main__":
    window.demo.launch(share=False, server_port=8080)
    window.demo.close()