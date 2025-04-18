import plotly.graph_objects as go
import requests
import streamlit as st
import trimesh
from streamlit.components.v1 import html

st.set_page_config(
    page_title="AI-powered Image and 3D Model Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_babylon_viewer(model_path: str) -> None:
    """
    Create a Babylon.js viewer component for the 3D model
    """
    # Read the HTML template
    with open("./templates/babylon_viewer.html", "r") as file:
        html_content = file.read()

    # Use st.components.html to embed the viewer
    html(html_content, height=300)


def load_3d_model(model_path) -> go.Figure:
    """
    Load and create a 3D model visualization from a GLB file.

    Args:
        model_path (str): Path to the GLB file to be loaded.

    Returns:
        plotly.graph_objects.Figure: A 3D scatter plot figure of the model's vertices.
    """
    # Load the GLB file
    scene = trimesh.load(model_path)

    # Extract vertices from the mesh
    if isinstance(scene, trimesh.Scene):
        # If it's a scene, get the first mesh
        mesh = next(iter(scene.geometry.values()))
    else:
        mesh = scene

    # Get vertices
    vertices = mesh.vertices
    x, y, z = vertices.T

    # Get vertex colors if they exist
    if hasattr(mesh.visual, "vertex_colors"):
        # Convert RGBA to hex colors for Plotly
        colors = [
            "#{:02x}{:02x}{:02x}".format(r, g, b)
            for r, g, b, _ in mesh.visual.vertex_colors
        ]
    else:
        # Default color if no vertex colors available
        colors = "rgb(100, 100, 100)"

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=1),
            )
        ]
    )

    # Make the figure more compact for sidebar
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
        height=300,  # Smaller height for sidebar
    )
    return fig


def main(recursed=False) -> None:
    """
    Main application function that sets up the Streamlit interface.
    Handles the chat interface, 3D model visualization, and image display.

    Args:
        recursed (bool): Flag to prevent infinite recursion when updating the UI.
                        Default is False.

    Returns:
        None: The function manages the Streamlit UI state directly.
    """
    # Initialize session state for containers if not exists
    if "sidebar_container" not in st.session_state:
        st.session_state.sidebar_container = st.sidebar.empty()

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Left sidebar
    with st.sidebar:
        st.session_state.sidebar_container.empty()

        with st.session_state.sidebar_container.container():
            st.title("Generation View")
            st.subheader("Generated Image")
            if "current_image" in st.session_state:
                st.image(st.session_state.current_image, use_column_width=True)

            st.markdown("---")

            st.subheader("3D Model")
            if "current_model" in st.session_state:
                # fig = load_3d_model(st.session_state.current_model)
                # st.plotly_chart(fig, use_container_width=True)
                load_babylon_viewer(st.session_state.current_model)
    if recursed:
        return

    # Main chat interface (right side)
    main_container = st.container()
    with main_container:
        st.title("How can I help you?")

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("What would you like to generate?"):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Show user message
            with st.chat_message("user"):
                st.write(prompt)

            # Show thinking message
            with st.chat_message("assistant"):
                with st.spinner("Generating..."):
                    # Make API call to your backend
                    response = requests.post(
                        "http://localhost:8888/execution", json={"prompt": prompt}
                    )

                    if response.status_code == 200:
                        # Update sidebar with new image and model
                        st.session_state.current_image = "./temp/image.png"
                        st.session_state.current_model = "./temp/model.glb"

                        # Add assistant response to chat
                        response_text = "I've generated your image and 3D model! You can see them in the sidebar."
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_text}
                        )
                        st.write(response_text)
                        main(recursed=True)  # Re-render to show new content
                    else:
                        error_text = (
                            "Sorry, there was an error generating your request."
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_text}
                        )
                        st.write(error_text)


if __name__ == "__main__":
    main()
