# Graph Authoring Guide

This guide explains how to create interactive architecture knowledge graphs for cisTEMx documentation using Cytoscape.js.

## Overview

Knowledge graphs provide an interactive way to explore component relationships, architecture patterns, and system dependencies. Our implementation uses:

- **Cytoscape.js**: Industry-standard graph visualization library
- **JSON data format**: Version-controlled, human-readable graph definitions
- **Rich interactivity**: Click for details, hover for tooltips, zoom and pan
- **Material theme integration**: Supports light/dark modes automatically

### Terminology

Understanding the interaction model:

- **Tooltip**: Small popup appearing on hover over nodes/edges, showing brief contextual information (category, label, short description)
- **Modal**: Detailed popup appearing when clicking a node, displaying comprehensive information (files, dependencies, documentation links, metadata)
- **Node**: Graph component representing a system element (module, class, subsystem)
- **Edge**: Connection between nodes representing relationships (dependencies, inheritance, usage)

This follows the **progressive disclosure** pattern: minimal information on hover, complete details on explicit interaction.

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
| `level` | string | No | Progressive disclosure level (basic, advanced, developer) |
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
| `level` | string | No | Progressive disclosure level (basic, advanced, developer) |
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

## Progressive Disclosure with Levels

The `level` attribute enables a single JSON file to serve multiple audiences by filtering complexity:

| Level | Description | Typical Audience |
|-------|-------------|------------------|
| `basic` | Essential components and relationships only | End users, newcomers |
| `advanced` | Additional implementation details | Experienced users |
| `developer` | Complete technical details | Contributors, maintainers |

**How it works**: Levels are hierarchical (basic ⊂ advanced ⊂ developer). A graph set to "advanced" shows both basic and advanced elements, but hides developer-only details.

### Using Levels in HTML

Add the `data-level` attribute to the container:

```html
<div id="my-graph" class="architecture-graph"
     data-graph-data="../graphs/my-graph-data.json"
     data-level="basic">
</div>
```

### Filtering Strategy

**Apply levels to both nodes AND edges**: If a "developer" level edge connects two "basic" nodes, that edge won't show in basic view.

**Example use case**: A single `cistemx-architecture.json` could contain:
- `basic`: High-level subsystems (GUI, Core, Database)
- `advanced`: Major classes and modules within subsystems
- `developer`: Individual source files and detailed dependencies

This creates one maintainable source of truth serving three documentation tiers.

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
     data-layout="cose"
     data-level="basic">
</div>
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `id` | Yes | Unique ID for this graph instance |
| `class` | Yes | Must include `architecture-graph`, can add size modifiers |
| `data-graph-data` | Yes | Relative path to JSON data file |
| `data-layout` | No | Override layout algorithm from JSON |
| `data-level` | No | Filter graph by level (basic, advanced, developer) |

### Size Modifiers

Add to `class` attribute:

- `compact`: 400px height (for inline graphs)
- `fullpage`: Full viewport height minus header
- (default): 600px height

Example:
```html
<div class="architecture-graph compact" ...></div>
```

## Design Strategies

### Multiple Graphs vs. Single Monolithic Graph

**Recommendation**: Create multiple focused graphs organized hierarchically rather than one large comprehensive graph.

**Rationale**:

1. **Cognitive Load**: Humans process 5-15 nodes effectively; beyond 20 nodes, graphs become difficult to navigate
2. **LLM Context Efficiency**: Smaller, focused graphs can be selectively loaded into LLM context for specific tasks
3. **Maintenance**: Easier to update specific subsystems without affecting unrelated components
4. **Performance**: Faster rendering and interaction with smaller graphs

**Hierarchical Organization Example**:

```
graphs/
├── cistemx-overview.json          # Top-level architecture (8 nodes)
├── gui/
│   ├── gui-panels.json            # Panel hierarchy (12 nodes)
│   ├── gui-events.json            # Event system (10 nodes)
│   └── gui-socket-comm.json       # Socket communication (7 nodes)
├── core/
│   ├── core-engine.json           # Core processing engine (9 nodes)
│   ├── core-database.json         # Database schema (11 nodes)
│   └── core-file-io.json          # File format handlers (8 nodes)
└── algorithms/
    ├── image-processing.json      # Image processing pipeline (13 nodes)
    └── gpu-acceleration.json      # CUDA components (6 nodes)
```

Each graph is self-contained and digestible, while the directory structure shows the overall organization.

### LLM-Assisted Development Utility

Knowledge graphs serve dual purposes:

1. **Human Documentation**: Interactive exploration and learning
2. **LLM Context**: Structured architectural information for AI-assisted development

**For LLM context, graphs provide**:

- **Relationship mapping**: What depends on what, inheritance hierarchies, usage patterns
- **Component boundaries**: Where responsibilities begin and end
- **File-to-concept mapping**: Which files implement which architectural concepts
- **Dependency information**: External libraries, internal module dependencies
- **Entry points**: Where to start reading code for specific features

**Best practices for LLM utility**:

- **Keep graphs focused**: 5-15 nodes per graph ensures the full graph fits comfortably in context
- **Rich metadata**: Include file paths, key dependencies, and conceptual descriptions
- **Accurate relationships**: LLMs use edges to understand data flow and control flow
- **Link to documentation**: `docLink` fields guide deeper investigation

**Example usage pattern**:

```
Developer: "I need to add a new GUI panel for particle refinement"
→ Provide: graphs/gui/gui-panels.json
→ LLM learns: Panel inheritance hierarchy, existing panel patterns, event handling
→ Guides implementation following established patterns
```

Focused graphs prevent context pollution—load only relevant architectural knowledge.

### Data Collection Strategies

Three approaches to populating graph data, each with tradeoffs:

#### 1. Manual Curation (Current Approach)

**Process**: Domain expert selects representative components and relationships

**Advantages**:
- Pedagogical value: emphasizes important patterns, omits noise
- Accurate conceptual relationships
- Controlled abstraction level
- Editorial oversight ensures quality

**Disadvantages**:
- Time-intensive to create and maintain
- May become outdated as code evolves
- Subjective component selection

**Best for**: High-level architecture graphs, onboarding documentation, conceptual overviews

#### 2. Semi-Automated Collection

**Process**: Scripts extract data, humans curate and annotate

**Examples**:
```bash
# Count files in subsystem
find src/gui -name "*.cpp" -o -name "*.h" | wc -l

# Identify maintainers
git shortlog -sn -- src/gui/

# Extract includes for dependencies
grep -rh "^#include" src/gui/ | sort -u

# Find class definitions
grep -rh "^class.*{" src/gui/*.h
```

**Advantages**:
- Accurate file counts and paths
- Real dependency information from includes
- Can be re-run to detect changes
- Balances accuracy with effort

**Disadvantages**:
- Still requires human interpretation
- Include parsing doesn't capture conceptual dependencies
- May miss implicit relationships

**Best for**: Subsystem detail graphs, file-level dependency tracking, metadata accuracy

#### 3. Fully Automated Generation

**Process**: Parse source code, generate graphs programmatically

**Potential tools**:
- Clang AST parsing for C++ structure
- CMake dependency extraction
- Doxygen relationship extraction
- Custom graph builders

**Advantages**:
- Always accurate to current codebase
- Scalable to large codebases
- Can regenerate on every build
- Objective, comprehensive

**Disadvantages**:
- May be overwhelming (too much detail)
- Lacks conceptual organization
- Requires significant tooling investment
- Often produces "hairball" graphs without curation

**Best for**: Large implementation graphs, dependency audits, automated documentation pipelines

### Recommended Hybrid Approach

**For cisTEMx knowledge graphs**:

1. **High-level architecture** (overview, major subsystems): Manual curation
2. **Subsystem details** (GUI panels, database schema): Semi-automated with human refinement
3. **Implementation details** (class relationships): Consider automated generation if needed, with careful filtering

This balances accuracy, maintainability, and pedagogical value.

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
