# AI Creative Partner 🎨

An intelligent pipeline that transforms text prompts into 3D models using local LLM and Openfabric apps.

## Features 🚀

- Text-to-Image-to-3D pipeline
- Local LLM (Mistral) integration
- Smart memory system for context retention
- Dynamic request handling with Openfabric SDK

## Architecture 🏗

```txt
User Prompt → Local LLM → Text-to-Image → Image-to-3D → 3D Model
```

## Memory System 🧠

The application uses a two-tier memory system:

### Short-term Memory

- Maintains context during active sessions
- Stores recent prompts, including original and expanded
- Enables quick reference to recent creations

### Long-term Memory (SQLite)

- Persistent storage across sessions
- Stores:
  - Original prompts
  - Expanded prompts
  - Creation timestamps
- Enables historical reference and iteration

## Example Usage 📝

Input a creative prompt:

```txt
"Create a blue lamborghini"
```

The system will:

- Use LLM to enhance the prompt
- Generate an image using Text-to-Image app
- Convert the image to a 3D model
- Store everything in memory

Reference previous creations:

```txt
"Create another lamborghini like yesterday's, but with white color"
```

## Dependencies 📦

- Python 3.11+
- Openfabric SDK
- Local LLM (Mistral)
- SQLite (for persistent storage)
- FAISS (vector storage)

## Development Setup 💻

Clone the repository:

```bash
git clone https://github.com/believemanasseh/openfabric
cd openfabric
```

Install dependencies:

```bash
poetry install
```

Configure the LLM:

```bash
# Follow LLM-specific setup instructions
```

Run the server:

```bash
./start.sh
```

Run the Streamlit client

```bash
streamlit run client.py
```
