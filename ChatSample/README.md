# Chat LLM Sample

Interactive chat interface powered by the LLaVA OneVision multimodal model running locally in Unity using Sentis.

![Chat Interface](Documentation/main.png)

## Runtime Inference

To power this experience we leverage the LLaVA OneVision model, a multimodal large language model.

The system processes both text and image inputs:
- Text is tokenized and embedded using the model's text encoder
- Images are processed through a vision encoder to extract visual features
- The decoder generates contextual responses based on both modalities

We use this to create a seamless conversational AI experience.

## Features

- **Multimodal Understanding**: Processes both text and images in conversation
- **Real-time Inference**: Fast GPU-accelerated inference using Unity's Sentis
- **Editor Integration**: Available as an Editor window for development and testing
- **Streaming Responses**: Token-by-token response generation for responsive interaction
- **Model Management**: HuggingFace model downloading

## Getting Started

1. Open the Unity project
2. Download models by navigating to **Sentis > Sample > Chat > Download Models** in the menu
3. Navigate to **Sentis > Sample > Chat > Start Chat** in the menu
4. Start chatting with the AI assistant!

Alternatively, you can manually download the models from [https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-si-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-si-hf) and place them in `ChatSample/Assets/ChatLLM/Resources/Models/`.

You can also use the runtime scene at `ChatSample/Assets/ChatLLM/Runtime/Scenes/Chat.unity`, but make sure to download the models beforehand using the editor menu.

The chat interface supports:
- Text-only conversations
- Image uploads with questions about the visual content
- Conversation history and context retention

## Technical Implementation

The sample demonstrates:
- Integration of multimodal LLM models in Unity
- Asynchronous model inference
- AppUI for modern editor interfaces
- State management using Redux patterns
- Model scheduling and resource management