/**
 * Architecture Graph Visualization using Cytoscape.js
 *
 * This module provides interactive knowledge graph visualization for cisTEMx
 * architecture documentation. It supports rich interactivity including:
 * - Click nodes to open detailed pop-ups
 * - Hover for quick tooltips
 * - Navigate to related documentation
 * - Explore component relationships
 *
 * Usage:
 * 1. Create a container div with a unique ID and class 'architecture-graph'
 * 2. Add data-graph-data attribute pointing to JSON graph data
 * 3. This script auto-initializes on page load
 *
 * Example:
 * <div id="main-arch-graph" class="architecture-graph"
 *      data-graph-data="path/to/graph-data.json"></div>
 */

(function() {
    'use strict';

    /**
     * Level hierarchy for progressive disclosure
     * Higher numbers include everything from lower levels
     */
    const LEVEL_ORDER = {
        'basic': 1,
        'advanced': 2,
        'developer': 3
    };

    /**
     * Filter graph data by visibility level
     * Implements progressive disclosure: basic ⊂ advanced ⊂ developer
     *
     * @param {Object} graphData - Complete graph data
     * @param {string} targetLevel - Level to display ('basic', 'advanced', 'developer')
     * @returns {Object} Filtered graph data
     */
    function filterByLevel(graphData, targetLevel) {
        // If no level specified, show everything
        if (!targetLevel) {
            return graphData;
        }

        const maxLevel = LEVEL_ORDER[targetLevel.toLowerCase()] || 3;

        // Filter nodes based on level
        const filteredNodes = (graphData.nodes || []).filter(node => {
            const nodeLevel = LEVEL_ORDER[node.data.level || 'basic'];
            return nodeLevel <= maxLevel;
        });

        // Get IDs of included nodes for edge filtering
        const includedNodeIds = new Set(filteredNodes.map(n => n.data.id));

        // Filter edges: include if level appropriate AND both endpoints exist
        const filteredEdges = (graphData.edges || []).filter(edge => {
            const edgeLevel = LEVEL_ORDER[edge.data.level || 'basic'];
            return edgeLevel <= maxLevel &&
                   includedNodeIds.has(edge.data.source) &&
                   includedNodeIds.has(edge.data.target);
        });

        // Return filtered data, explicitly removing 'elements' property if it exists
        const { elements, ...restOfGraphData } = graphData;
        return {
            ...restOfGraphData,
            nodes: filteredNodes,
            edges: filteredEdges
        };
    }

    /**
     * Initialize Cytoscape graph in a container
     * @param {string} containerId - DOM element ID for the graph container
     * @param {Object} graphData - Graph data with nodes and edges (will be filtered if level specified)
     * @param {Object} options - Optional configuration overrides
     */
    function initializeGraph(containerId, graphData, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return null;
        }

        // Default configuration
        const defaultConfig = {
            container: container,

            elements: graphData.elements || {
                nodes: graphData.nodes || [],
                edges: graphData.edges || []
            },

            style: [
                // Node styles
                {
                    selector: 'node',
                    style: {
                        'background-color': '#4CAF50',
                        'label': 'data(label)',
                        'color': '#fff',
                        'text-halign': 'center',
                        'text-valign': 'center',
                        'font-size': '12px',
                        'font-weight': 'bold',
                        'width': 'label',
                        'height': 'label',
                        'padding': '10px',
                        'shape': 'roundrectangle',
                        'text-wrap': 'wrap',
                        'text-max-width': '80px',
                        'border-width': 2,
                        'border-color': '#2E7D32'
                    }
                },
                // Node hover state
                {
                    selector: 'node:active',
                    style: {
                        'overlay-opacity': 0.2,
                        'overlay-color': '#4CAF50'
                    }
                },
                // Selected node
                {
                    selector: 'node:selected',
                    style: {
                        'border-width': 3,
                        'border-color': '#FFC107'
                    }
                },
                // Node categories (can be extended)
                {
                    selector: 'node[category="core"]',
                    style: {
                        'background-color': '#2196F3',
                        'border-color': '#1565C0'
                    }
                },
                {
                    selector: 'node[category="gui"]',
                    style: {
                        'background-color': '#9C27B0',
                        'border-color': '#6A1B9A'
                    }
                },
                {
                    selector: 'node[category="database"]',
                    style: {
                        'background-color': '#FF5722',
                        'border-color': '#D84315'
                    }
                },
                {
                    selector: 'node[category="algorithm"]',
                    style: {
                        'background-color': '#4CAF50',
                        'border-color': '#2E7D32'
                    }
                },

                // Edge styles
                {
                    selector: 'edge',
                    style: {
                        'width': 2,
                        'line-color': '#999',
                        'target-arrow-color': '#999',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'arrow-scale': 1.5
                    }
                },
                // Edge hover
                {
                    selector: 'edge:active',
                    style: {
                        'line-color': '#4CAF50',
                        'target-arrow-color': '#4CAF50',
                        'width': 3
                    }
                },
                // Edge relationship types
                {
                    selector: 'edge[relationship="depends"]',
                    style: {
                        'line-style': 'solid'
                    }
                },
                {
                    selector: 'edge[relationship="inherits"]',
                    style: {
                        'line-style': 'dashed'
                    }
                },
                {
                    selector: 'edge[relationship="uses"]',
                    style: {
                        'line-style': 'dotted'
                    }
                }
            ],

            layout: Object.assign({
                // Default COSE (Compound Spring Embedder) layout options
                name: 'cose',
                animate: true,
                animationDuration: 500,
                fit: true,
                padding: 30,
                nodeRepulsion: 400000,
                idealEdgeLength: 100,
                edgeElasticity: 100,
                nestingFactor: 5,
                gravity: 80,
                numIter: 1000,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0
            }, graphData.layout || {}, options.layout || {}),

            // Interaction options
            minZoom: 0.3,
            maxZoom: 3,
            wheelSensitivity: 0.001  // Default: very low (nearly disabled)
        };

        // Merge with custom options (excluding layout which was already merged)
        const {layout: optionsLayout, ...otherOptions} = options;
        const config = Object.assign({}, defaultConfig, otherOptions);

        // Store layout config BEFORE Cytoscape modifies it
        // Deep copy the layout object to preserve it
        const layoutConfig = JSON.parse(JSON.stringify(config.layout));

        // Initialize Cytoscape
        const cy = cytoscape(config);

        // Setup interactivity
        setupNodeInteractions(cy, container);
        setupEdgeInteractions(cy);

        // Store cy instance globally for tab visibility handling
        if (!window.cytoscapeInstances) {
            window.cytoscapeInstances = {};
        }
        console.log(`[${containerId}] Storing layout config:`, layoutConfig);
        window.cytoscapeInstances[containerId] = { cy: cy, layout: layoutConfig, resized: false };

        return cy;
    }

    /**
     * Setup node click and hover interactions
     */
    function setupNodeInteractions(cy, container) {
        // Tooltip element
        let tooltip = createTooltip();

        // Hover: show tooltip
        cy.on('mouseover', 'node', function(evt) {
            const node = evt.target;
            const description = node.data('description') || 'No description available';
            const category = node.data('category') || 'unknown';

            tooltip.innerHTML = `
                <div class="graph-tooltip-category">${category}</div>
                <div class="graph-tooltip-title">${node.data('label')}</div>
                <div class="graph-tooltip-desc">${description}</div>
            `;
            tooltip.style.display = 'block';
        });

        cy.on('mouseout', 'node', function(evt) {
            tooltip.style.display = 'none';
        });

        cy.on('mousemove', function(evt) {
            tooltip.style.left = evt.originalEvent.pageX + 10 + 'px';
            tooltip.style.top = evt.originalEvent.pageY + 10 + 'px';
        });

        // Click: show detailed modal
        cy.on('tap', 'node', function(evt) {
            const node = evt.target;
            showNodeModal(node, container);
        });
    }

    /**
     * Setup edge interactions
     */
    function setupEdgeInteractions(cy) {
        let tooltip = createTooltip();

        cy.on('mouseover', 'edge', function(evt) {
            const edge = evt.target;
            const relationship = edge.data('relationship') || 'related to';
            const source = edge.source().data('label');
            const target = edge.target().data('label');

            tooltip.innerHTML = `
                <div class="graph-tooltip-edge">
                    <strong>${source}</strong> ${relationship} <strong>${target}</strong>
                </div>
            `;
            tooltip.style.display = 'block';
        });

        cy.on('mouseout', 'edge', function(evt) {
            tooltip.style.display = 'none';
        });
    }

    /**
     * Create tooltip element
     */
    function createTooltip() {
        let tooltip = document.getElementById('graph-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'graph-tooltip';
            tooltip.className = 'graph-tooltip';
            document.body.appendChild(tooltip);
        }
        return tooltip;
    }

    /**
     * Show detailed modal for a node
     */
    function showNodeModal(node, container) {
        const data = node.data();

        // Create modal if it doesn't exist
        let modal = document.getElementById('graph-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'graph-modal';
            modal.className = 'graph-modal';
            modal.innerHTML = `
                <div class="graph-modal-content">
                    <span class="graph-modal-close">&times;</span>
                    <div class="graph-modal-body"></div>
                </div>
            `;
            document.body.appendChild(modal);

            // Close button handler
            modal.querySelector('.graph-modal-close').onclick = function() {
                modal.style.display = 'none';
            };

            // Click outside to close
            window.onclick = function(event) {
                if (event.target == modal) {
                    modal.style.display = 'none';
                }
            };
        }

        // Populate modal content
        const modalBody = modal.querySelector('.graph-modal-body');
        modalBody.innerHTML = generateModalContent(data);

        // Show modal
        modal.style.display = 'block';
    }

    /**
     * Generate modal content from node data
     */
    function generateModalContent(data) {
        let html = `
            <div class="graph-modal-header">
                <span class="graph-modal-category">${data.category || 'Component'}</span>
                <h2>${data.label}</h2>
            </div>
        `;

        // Description
        if (data.description) {
            html += `
                <div class="graph-modal-section">
                    <h3>Description</h3>
                    <p>${data.description}</p>
                </div>
            `;
        }

        // File paths
        if (data.files && data.files.length > 0) {
            html += `
                <div class="graph-modal-section">
                    <h3>Source Files</h3>
                    <ul class="graph-modal-files">
                        ${data.files.map(file => `<li><code>${file}</code></li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Dependencies
        if (data.dependencies && data.dependencies.length > 0) {
            html += `
                <div class="graph-modal-section">
                    <h3>Dependencies</h3>
                    <ul class="graph-modal-deps">
                        ${data.dependencies.map(dep => `<li>${dep}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        // Documentation links
        if (data.docLink) {
            html += `
                <div class="graph-modal-section">
                    <h3>Documentation</h3>
                    <a href="${data.docLink}" class="graph-modal-link" target="_blank">
                        View detailed documentation →
                    </a>
                </div>
            `;
        }

        // Additional metadata
        if (data.metadata) {
            html += `
                <div class="graph-modal-section">
                    <h3>Additional Information</h3>
                    <dl class="graph-modal-metadata">
            `;
            for (const [key, value] of Object.entries(data.metadata)) {
                html += `
                    <dt>${key}</dt>
                    <dd>${value}</dd>
                `;
            }
            html += `
                    </dl>
                </div>
            `;
        }

        return html;
    }

    /**
     * Auto-initialize all graphs on page load
     */
    function autoInitialize() {
        // Wait for DOM and Cytoscape to be ready
        if (typeof cytoscape === 'undefined') {
            console.error('Cytoscape.js not loaded. Please include it before architecture_graph.js');
            return;
        }

        // Find all graph containers
        const containers = document.querySelectorAll('.architecture-graph');

        containers.forEach(container => {
            const containerId = container.id;
            const graphDataPath = container.getAttribute('data-graph-data');
            const layoutType = container.getAttribute('data-layout') || 'cose';
            const wheelZoomAttr = container.getAttribute('data-wheel-zoom');
            const levelFilter = container.getAttribute('data-level');

            if (!containerId) {
                console.warn('Graph container missing ID attribute');
                return;
            }

            if (!graphDataPath) {
                console.warn(`Graph container ${containerId} missing data-graph-data attribute`);
                return;
            }

            // Build options object
            const initOptions = { layout: layoutType };

            // Only set wheelSensitivity if explicitly provided
            if (wheelZoomAttr !== null) {
                initOptions.wheelSensitivity = parseFloat(wheelZoomAttr);
            }

            // Load graph data
            fetch(graphDataPath)
                .then(response => response.json())
                .then(graphData => {
                    // Apply level filtering if specified
                    const filteredData = levelFilter ? filterByLevel(graphData, levelFilter) : graphData;
                    console.log(`[${containerId}] Level: ${levelFilter}, Nodes: ${filteredData.nodes?.length}, Edges: ${filteredData.edges?.length}`);
                    console.log(`[${containerId}] Layout:`, filteredData.layout);
                    initializeGraph(containerId, filteredData, initOptions);
                })
                .catch(error => {
                    console.error(`Failed to load graph data from ${graphDataPath}:`, error);
                    container.innerHTML = `
                        <div class="graph-error">
                            Failed to load graph data. Please check the console for details.
                        </div>
                    `;
                });
        });
    }

    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', autoInitialize);
    } else {
        autoInitialize();
    }

    /**
     * Handle MkDocs Material tab clicks
     * When tabs are clicked, resize Cytoscape graphs that may have been hidden
     */
    function setupTabHandlers() {
        document.addEventListener('click', function(event) {
            // Check if a tab label was clicked
            const tabLabel = event.target.closest('.tabbed-labels label');
            if (tabLabel) {
                // Give the tab content time to become visible
                setTimeout(function() {
                    // Resize all Cytoscape instances
                    if (window.cytoscapeInstances) {
                        Object.keys(window.cytoscapeInstances).forEach(function(containerId) {
                            const instance = window.cytoscapeInstances[containerId];
                            const container = document.getElementById(containerId);

                            // Only resize if container is now visible
                            if (container && container.offsetWidth > 0 && container.offsetHeight > 0) {
                                instance.cy.resize();

                                // Only run layout on first resize (when tab first becomes visible)
                                if (!instance.resized) {
                                    console.log(`[${containerId}] About to run layout with:`, instance.layout);
                                    instance.cy.layout(instance.layout).run();
                                    instance.resized = true;
                                    console.log(`[${containerId}] Layout run after tab visibility`);
                                }
                            }
                        });
                    }
                }, 100); // Small delay to let tab transition complete
            }
        });
    }

    // Setup tab handlers after DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupTabHandlers);
    } else {
        setupTabHandlers();
    }

    // Expose API for manual initialization if needed
    window.ArchitectureGraph = {
        init: initializeGraph
    };

})();
