# Text-to-Speech Sample

Interactive text-to-speech interface powered by the Kokoro TTS model running locally in Unity using Inference Engine.

![TTS Interface](Documentation/main.png)

## Runtime Inference

To power this experience we leverage the Kokoro model, a high-quality neural text-to-speech model.

The system processes text inputs through:
- Text tokenization and grapheme-to-phoneme conversion using eSpeak
- Neural voice synthesis using the Kokoro ONNX model
- Real-time audio generation with multiple voice options
- Configurable speech speed and voice selection

We use this to create a seamless text-to-speech experience with natural-sounding voices.

## Features

- **Multiple Voices**: Choose from various pre-trained voice styles
- **Speed Control**: Adjustable speech rate for different use cases
- **Real-time Generation**: Fast GPU-accelerated inference using Unity's Inference Engine
- **Editor Integration**: Available as an Editor window for development and testing
- **Cross-Platform**: Support for macOS, Windows, Linux, and Android
- **Model Management**: Automated model downloading and setup
- **Phoneme Processing**: Automatic grapheme-to-phoneme conversion using eSpeak

## Getting Started

1. Open the Unity project
2. Download models by navigating to **Inference Engine > Sample > TTS > Download Models** in the menu
3. Navigate to **Inference Engine > Sample > TTS > Start TTS** in the menu
4. Enter text and generate speech with your chosen voice!

Alternatively, you can use the runtime scene at `TextToSpeechSample/Assets/Scenes/App.unity`, but make sure to download the models beforehand using the editor menu.

The TTS interface supports:
- Multi-line text input
- Voice selection from available models
- Real-time speech generation
- Audio playback controls

## Technical Implementation

The sample demonstrates:
- Integration of neural TTS models in Unity
- Asynchronous model inference with Kokoro
- AppUI for modern editor interfaces
- State management using Redux patterns
- Model scheduling and resource management
- Cross-platform audio generation
- eSpeak integration for phoneme conversion