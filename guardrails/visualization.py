"""
Utilities for visualizing LangGraph workflows with guardrails.

Provides functions to generate visual representations of your graphs,
making it easier to understand and document guardrail integration.
"""

import os
from pathlib import Path
from typing import Optional


def visualize_graph(
    graph,
    output_path: str = None,
    format: str = "mermaid",
    title: str = None,
) -> str:
    """
    Visualize a LangGraph graph.

    Args:
        graph: Compiled LangGraph graph
        output_path: Path to save visualization (optional)
        format: "mermaid", "ascii", or "png"
        title: Optional title for the diagram

    Returns:
        String representation of the graph (for mermaid/ascii) or path to saved image (for png)
    """
    if format == "mermaid":
        return _visualize_mermaid(graph, output_path, title)
    elif format == "ascii":
        return _visualize_ascii(graph, output_path)
    elif format == "png":
        return _visualize_png(graph, output_path, title)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'mermaid', 'ascii', or 'png'")


def _visualize_mermaid(graph, output_path: Optional[str] = None, title: Optional[str] = None) -> str:
    """Generate Mermaid diagram"""
    try:
        # Get mermaid representation
        mermaid_str = graph.get_graph().draw_mermaid()

        # Add title if provided
        if title:
            mermaid_str = f"---\ntitle: {title}\n---\n{mermaid_str}"

        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(mermaid_str)
            print(f"Saved Mermaid diagram to: {output_path}")

        return mermaid_str

    except Exception as e:
        return f"Error generating Mermaid diagram: {e}"


def _visualize_ascii(graph, output_path: Optional[str] = None) -> str:
    """Generate ASCII diagram (requires grandalf: pip install grandalf)"""
    try:
        # Get ASCII representation
        ascii_str = graph.get_graph().draw_ascii()

        # Save to file if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(ascii_str)
            print(f"Saved ASCII diagram to: {output_path}")

        return ascii_str

    except ImportError as e:
        msg = f"Error: ASCII generation requires grandalf. Install with: pip install grandalf\n{e}"
        return msg
    except Exception as e:
        return f"Error generating ASCII diagram: {e}"


def _visualize_png(graph, output_path: str, title: Optional[str] = None) -> str:
    """Generate PNG image (requires graphviz)"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        # Try to generate PNG
        try:
            png_data = graph.get_graph().draw_mermaid_png()
        except Exception as e:
            return f"Error: PNG generation requires graphviz. Install with: pip install graphviz pygraphviz\n{e}"

        # Save to file
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # If title provided, add it to the image
            if title:
                img = Image.open(io.BytesIO(png_data))
                # Add title at top
                new_height = img.height + 40
                new_img = Image.new('RGB', (img.width, new_height), 'white')
                new_img.paste(img, (0, 40))

                # Draw title
                draw = ImageDraw.Draw(new_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                draw.text((10, 10), title, fill='black', font=font)

                new_img.save(output_path)
            else:
                with open(output_path, 'wb') as f:
                    f.write(png_data)

            print(f"Saved PNG diagram to: {output_path}")
            return output_path

    except ImportError as e:
        return f"Error: PNG generation requires additional packages. Install with: pip install pillow\n{e}"
    except Exception as e:
        return f"Error generating PNG diagram: {e}"


def print_graph_info(graph):
    """
    Print information about a graph's structure.

    Useful for debugging and understanding graph topology.
    """
    print("=" * 60)
    print("GRAPH STRUCTURE")
    print("=" * 60)

    try:
        graph_data = graph.get_graph()

        # Print nodes
        nodes = list(graph_data.nodes.keys())
        print(f"\nNodes ({len(nodes)}):")
        for node in nodes:
            print(f"  - {node}")

        # Print edges
        print(f"\nEdges:")
        edges_list = graph_data.edges if isinstance(graph_data.edges, list) else []
        for edge in edges_list:
            source = edge.source if hasattr(edge, 'source') else str(edge)
            target = edge.target if hasattr(edge, 'target') else str(edge)
            conditional = " (conditional)" if hasattr(edge, 'conditional') and edge.conditional else ""
            print(f"  {source} -> {target}{conditional}")

        print(f"\nTotal edges: {len(edges_list)}")

        # Print entry point
        if hasattr(graph_data, 'entry_point'):
            print(f"\nEntry point: {graph_data.entry_point}")

    except Exception as e:
        print(f"Error analyzing graph: {e}")

    print("=" * 60)


def create_graph_documentation(
    examples_dir: str = "examples",
    output_dir: str = "docs/diagrams",
    format: str = "mermaid",
):
    """
    Generate visualizations for all example files.

    Args:
        examples_dir: Directory containing example files
        output_dir: Directory to save visualizations
        format: Format for visualizations ("mermaid", "ascii", or "png")
    """
    import importlib.util
    import sys

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all example files
    example_files = sorted(Path(examples_dir).glob("*.py"))

    print(f"Generating {format} diagrams for {len(example_files)} examples...")
    print()

    for example_file in example_files:
        if example_file.name.startswith("_"):
            continue

        print(f"Processing {example_file.name}...")

        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(example_file.stem, example_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[example_file.stem] = module
            spec.loader.exec_module(module)

            # Look for a graph building function
            build_fn = None
            for attr_name in dir(module):
                if attr_name.startswith("build_") and callable(getattr(module, attr_name)):
                    build_fn = getattr(module, attr_name)
                    break

            if build_fn:
                graph = build_fn()
                output_file = f"{output_dir}/{example_file.stem}.{format}"
                title = example_file.stem.replace("_", " ").title()

                visualize_graph(graph, output_file, format=format, title=title)
            else:
                print(f"  No build function found in {example_file.name}")

        except Exception as e:
            print(f"  Error processing {example_file.name}: {e}")

        print()

    print(f"Done! Diagrams saved to {output_dir}/")


# Example usage
if __name__ == "__main__":
    print("Graph Visualization Utilities")
    print()
    print("Usage:")
    print("  from guardrails.visualization import visualize_graph, print_graph_info")
    print()
    print("  # Visualize a graph")
    print("  graph = build_my_graph()")
    print("  visualize_graph(graph, 'my_graph.mermaid', format='mermaid')")
    print()
    print("  # Print graph structure")
    print("  print_graph_info(graph)")
    print()
    print("  # Generate docs for all examples")
    print("  create_graph_documentation('examples', 'docs/diagrams')")
