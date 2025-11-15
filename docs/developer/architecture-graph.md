# Architecture Knowledge Graph

This interactive visualization shows the high-level architecture of cisTEMx, including major components and their relationships.

## How to Use This Graph

**Interactive Features:**

- **Hover over nodes**: See a quick tooltip with component description
- **Click on nodes**: Open a detailed modal with files, dependencies, and links to documentation
- **Hover over edges**: View the relationship type between components
- **Pan and zoom**: Click and drag to pan, scroll to zoom in/out
- **Select nodes**: Click to highlight a node and its connections

## cisTEMx Architecture Overview

<div id="cistemx-arch-graph" class="architecture-graph"
     data-graph-data="/graphs/cistemx-overview.json"
     data-layout="preset"
     data-saved-layout="/javascripts/saved-layouts/architecture-graph.json">
</div>

<script>
// Load saved positions for refined architecture graph layout
(function() {
  function applySavedPositions() {
    const container = document.getElementById('cistemx-arch-graph');
    if (!container || !window.cytoscapeInstances) {
      setTimeout(applySavedPositions, 100);
      return;
    }

    const cy = window.cytoscapeInstances['cistemx-arch-graph']?.cy;
    if (!cy) {
      setTimeout(applySavedPositions, 100);
      return;
    }

    // Load saved positions from JSON file
    const layoutPath = container.getAttribute('data-saved-layout');
    if (layoutPath) {
      fetch(layoutPath)
        .then(response => response.json())
        .then(positions => {
          cy.nodes().forEach(node => {
            if (positions[node.id()]) {
              node.position(positions[node.id()]);
            }
          });
          // Fit viewport with padding
          cy.fit(null, 40);
          console.log('✓ Applied saved layout positions for architecture graph');
        })
        .catch(error => {
          console.error('Could not load saved positions:', error);
          // Fallback to algorithmic layout if saved positions fail
          cy.layout({ name: 'cose' }).run();
        });
    }
  }

  // Start initialization when page loads
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applySavedPositions);
  } else {
    applySavedPositions();
  }
})();
</script>

## Legend

<div class="graph-legend">

### Component Categories

<div class="graph-legend-items">
  <div class="graph-legend-item">
    <span class="graph-legend-color core"></span>
    <span class="graph-legend-label">Core</span>
  </div>
  <div class="graph-legend-item">
    <span class="graph-legend-color gui"></span>
    <span class="graph-legend-label">GUI</span>
  </div>
  <div class="graph-legend-item">
    <span class="graph-legend-color database"></span>
    <span class="graph-legend-label">Database</span>
  </div>
  <div class="graph-legend-item">
    <span class="graph-legend-color algorithm"></span>
    <span class="graph-legend-label">Algorithm</span>
  </div>
</div>

### Relationship Types

<div class="graph-legend-edges">
  <div class="graph-legend-edge-item">
    <span class="graph-legend-line solid"></span>
    <span class="graph-legend-label">depends — Required dependency</span>
  </div>
  <div class="graph-legend-edge-item">
    <span class="graph-legend-line dashed"></span>
    <span class="graph-legend-label">inherits — Inheritance relationship</span>
  </div>
  <div class="graph-legend-edge-item">
    <span class="graph-legend-line dotted"></span>
    <span class="graph-legend-label">uses — Functional usage</span>
  </div>
</div>

</div>

## Component Details

Click on any component in the graph above to see:

- **Description**: What the component does and why it exists
- **Source Files**: Key files implementing the component
- **Dependencies**: External libraries and internal components it requires
- **Documentation**: Links to detailed documentation
- **Metadata**: Technology stack, language, and other relevant information

## Architecture Insights

The graph reveals several key architectural patterns in cisTEMx:

1. **Layered Architecture**: Clear separation between GUI, core engine, and algorithms
2. **Database-Centric**: Both GUI and core engine interact with persistent storage
3. **GPU Acceleration**: Optional CUDA support for performance-critical operations
4. **Modular Design**: CLI programs and simulation tools built on shared core infrastructure
5. **File Format Abstraction**: Centralized I/O layer for scientific data formats

## Creating Your Own Graphs

Want to create additional architecture graphs for specific subsystems? See the [Graph Authoring Guide](../reference/graph-authoring.md) for details on:

- JSON schema for graph data
- Node and edge properties
- Layout algorithms
- Styling customization
- Best practices for effective visualizations
