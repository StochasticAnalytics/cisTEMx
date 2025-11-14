# Graph Authoring Guide

This guide explains how to create interactive architecture knowledge graphs for cisTEMx documentation using Cytoscape.js.

## Overview

Knowledge graphs provide an interactive way to explore component relationships, architecture patterns, and system dependencies. Our implementation uses:

- **Cytoscape.js**: Industry-standard graph visualization library
- **JSON data format**: Version-controlled, human-readable graph definitions
- **Rich interactivity**: Click for details, hover for tooltips, zoom and pan
- **Material theme integration**: Supports light/dark modes automatically

## Quick Start

### 1. Create Graph Data (JSON)

Create a JSON file in `/docs/graphs/` with the following structure:

```json
{
  "title": "My Graph Title",
  "description": "Brief description of what this graph shows",
  "layout": "cose",
  "nodes": [
    {
      "data": {
        "id": "unique-node-id",
        "label": "Node Display Name",
        "category": "core",
        "description": "Brief description for tooltip",
        "files": ["path/to/file1.cpp", "path/to/file2.h"],
        "dependencies": ["Library1", "Component2"],
        "docLink": "/path/to/docs/",
        "metadata": {
          "Key1": "Value1",
          "Key2": "Value2"
        }
      }
    }
  ],
  "edges": [
    {
      "data": {
        "id": "edge-id",
        "source": "source-node-id",
        "target": "target-node-id",
        "relationship": "depends",
        "description": "What this relationship means"
      }
    }
  ]
}
```

### 2. Add Graph to Documentation Page

In your markdown file, add a graph container:

```html
<div id="my-graph" class="architecture-graph"
     data-graph-data="../graphs/my-graph-data.json"
     data-layout="cose">
</div>
```

### 3. Build and View

Build the documentation and navigate to your page:

```bash
mkdocs serve
```

## JSON Schema Reference

### Top-Level Structure

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | No | Graph title (for documentation) |
| `description` | string | No | Graph description (for documentation) |
| `layout` | string | No | Layout algorithm (default: "cose") |
| `nodes` | array | Yes | Array of node objects |
| `edges` | array | Yes | Array of edge objects |

### Node Data Structure

Each node in the `nodes` array should have a `data` object with:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the node |
| `label` | string | Yes | Display name shown on the graph |
| `category` | string | No | Category for styling (core, gui, database, algorithm) |
| `description` | string | No | Brief description shown in tooltip |
| `files` | array[string] | No | Source files implementing this component |
| `dependencies` | array[string] | No | Required libraries or components |
| `docLink` | string | No | Link to detailed documentation |
| `metadata` | object | No | Additional key-value pairs for the modal |

### Edge Data Structure

Each edge in the `edges` array should have a `data` object with:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier for the edge |
| `source` | string | Yes | ID of source node |
| `target` | string | Yes | ID of target node |
| `relationship` | string | No | Type of relationship (depends, inherits, uses) |
| `description` | string | No | Description shown on hover |

## Node Categories and Styling

The `category` field determines node color:

| Category | Color | Use For |
|----------|-------|---------|
| `core` | Blue | Core infrastructure, engine components |
| `gui` | Purple | Graphical user interface components |
| `database` | Orange-Red | Database and persistence layer |
| `algorithm` | Green | Algorithms and processing modules |

To add custom categories, modify `/docs/stylesheets/architecture_graph.css`.

## Edge Relationship Types

The `relationship` field determines edge style:

| Relationship | Style | Use For |
|--------------|-------|---------|
| `depends` | Solid line | Hard dependencies, required for compilation/runtime |
| `inherits` | Dashed line | Inheritance relationships, extends functionality |
| `uses` | Dotted line | Functional usage, optional dependencies |

## Layout Algorithms

Cytoscape.js supports multiple layout algorithms. Specify in the `layout` field or `data-layout` attribute:

### COSE (Default)
**Compound Spring Embedder** - Physics-based, good for general graphs

```json
"layout": "cose"
```

**Best for**: General architecture graphs with moderate complexity

**Characteristics**:
- Physics-based simulation
- Handles overlaps well
- Good for discovering clusters
- Moderate computation time

### Circle
Nodes arranged in a circle

```json
"layout": "circle"
```

**Best for**: Small graphs showing cyclical relationships

### Grid
Nodes arranged in a grid

```json
"layout": "grid"
```

**Best for**: Uniform, ordered components

### Breadth-First
Hierarchical layout based on graph traversal

```json
"layout": "breadthfirst"
```

**Best for**: Clear hierarchies, dependency trees

**Additional options** can be specified in the JSON:

```json
"layout": {
  "name": "breadthfirst",
  "directed": true,
  "spacingFactor": 1.5
}
```

### Preset
Manual positioning (you specify x, y coordinates)

```json
"layout": "preset"
```

Each node needs `position` field:

```json
{
  "data": { "id": "node1", "label": "Node 1" },
  "position": { "x": 100, "y": 100 }
}
```

**Best for**: When you need precise control over layout

## HTML Container Options

### Container Attributes

```html
<div id="unique-id"
     class="architecture-graph [compact|fullpage]"
     data-graph-data="path/to/data.json"
     data-layout="cose">
</div>
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `id` | Yes | Unique ID for this graph instance |
| `class` | Yes | Must include `architecture-graph`, can add size modifiers |
| `data-graph-data` | Yes | Relative path to JSON data file |
| `data-layout` | No | Override layout algorithm from JSON |

### Size Modifiers

Add to `class` attribute:

- `compact`: 400px height (for inline graphs)
- `fullpage`: Full viewport height minus header
- (default): 600px height

Example:
```html
<div class="architecture-graph compact" ...></div>
```

## Best Practices

### Graph Complexity

**Optimal node count**: 5-15 nodes per graph

**Why**: Graphs with more than ~20 nodes become difficult to navigate. Consider:
- Creating multiple focused graphs for different subsystems
- Using hierarchical graphs with drill-down
- Grouping related components into higher-level nodes

### Node Labels

**Keep labels short**: 1-3 words ideal

**Use abbreviations** if necessary, but explain in description:
- Good: "CTF Estimation"
- Too long: "Contrast Transfer Function Estimation Module"

### Descriptions

**Tooltip descriptions** (node.description): 1-2 sentences, < 100 words

**Modal details** (node.metadata, node.files): Can be more comprehensive

### File Paths

**Use relative paths** from project root:
```json
"files": ["src/core/image.cpp", "src/core/image.h"]
```

**Use wildcards** for clarity:
```json
"files": ["src/gui/*.cpp", "src/gui/*.h"]
```

### Documentation Links

**Internal links**: Relative to docs root
```json
"docLink": "/developer/architecture/#section"
```

**External links**: Full URLs
```json
"docLink": "https://example.com/docs"
```

### Metadata

**Keep metadata relevant** and concise:

```json
"metadata": {
  "Technology": "wxWidgets",
  "Language": "C++",
  "Complexity": "High",
  "Maintainer": "Team Name"
}
```

Avoid dumping excessive technical details—link to documentation instead.

## Examples

### Simple Component Graph

```json
{
  "layout": "circle",
  "nodes": [
    {
      "data": {
        "id": "a",
        "label": "Component A",
        "category": "core"
      }
    },
    {
      "data": {
        "id": "b",
        "label": "Component B",
        "category": "algorithm"
      }
    }
  ],
  "edges": [
    {
      "data": {
        "source": "a",
        "target": "b",
        "relationship": "uses"
      }
    }
  ]
}
```

### Hierarchical Dependency Graph

```json
{
  "layout": "breadthfirst",
  "nodes": [
    {
      "data": {
        "id": "app",
        "label": "Application",
        "category": "gui",
        "description": "Main application entry point"
      }
    },
    {
      "data": {
        "id": "lib1",
        "label": "Library 1",
        "category": "core"
      }
    },
    {
      "data": {
        "id": "lib2",
        "label": "Library 2",
        "category": "core"
      }
    }
  ],
  "edges": [
    {
      "data": {
        "source": "app",
        "target": "lib1",
        "relationship": "depends"
      }
    },
    {
      "data": {
        "source": "app",
        "target": "lib2",
        "relationship": "depends"
      }
    }
  ]
}
```

## Troubleshooting

### Graph Not Rendering

**Symptoms**: Empty container or error message

**Check**:
1. Cytoscape.js loaded? (View page source, check for script tag)
2. JSON file path correct? (Check browser console for 404 errors)
3. JSON valid? (Use a JSON validator)
4. Container has unique ID?

### Tooltips Not Showing

**Check**:
1. Node has `description` field in data
2. No JavaScript errors in console

### Modal Not Opening

**Check**:
1. Clicking on nodes (not edges)
2. No JavaScript errors in console
3. Modal stylesheet loaded

### Layout Looks Wrong

**Try**:
1. Different layout algorithm (`data-layout="breadthfirst"`)
2. Adjust graph container size
3. Reduce number of nodes
4. Use preset layout with manual positions

## Advanced Customization

### Custom Styling

Edit `/docs/stylesheets/architecture_graph.css` to customize:

- Node colors and shapes
- Edge styles
- Tooltip appearance
- Modal layout
- Dark/light mode colors

### Custom Interactivity

Edit `/docs/javascripts/architecture_graph.js` to add:

- Custom click handlers
- Additional modal sections
- External data fetching
- Graph animations
- Export functionality

## Version Control

**Commit graph data**: JSON files should be version-controlled

**Document changes**: When updating graphs, note in commit message:
- Which components added/removed
- Why relationships changed
- If layout algorithm changed

**Review together**: Architecture graphs document system design—review with team

## Future Enhancements

Potential features for future development:

- **Auto-generation**: Generate graphs from code analysis
- **Search and filter**: Search nodes, filter by category
- **Export**: Download graph as PNG/SVG
- **Diff visualization**: Show changes between versions
- **Interactive editing**: Edit graphs in browser (would need backend)

---

**See Also**:
- [Architecture Overview](../developer/architecture.md)
- [Architecture Graph Example](../developer/architecture-graph.md)
- [Cytoscape.js Documentation](https://js.cytoscape.org/)
