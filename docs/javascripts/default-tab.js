/**
 * Auto-select Developer tab on all pages with tabbed content
 *
 * MkDocs Material with pymdownx.tabbed extension doesn't provide
 * a configuration option for default tab selection. This script
 * automatically selects the "Developer" tab when pages load.
 */

(function() {
    'use strict';

    /**
     * Select the Developer tab if it exists on the page
     */
    function selectDeveloperTab() {
        // Find all tab labels
        const tabLabels = document.querySelectorAll('.tabbed-labels label');

        // Look for the "Developer" tab
        tabLabels.forEach(function(label) {
            if (label.textContent.trim() === 'Developer') {
                // Find the associated radio input and check it
                const input = label.previousElementSibling;
                if (input && input.type === 'radio') {
                    input.checked = true;
                }
            }
        });
    }

    // Run when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', selectDeveloperTab);
    } else {
        selectDeveloperTab();
    }

    // Also run when navigating with instant loading (Material theme feature)
    document$.subscribe(selectDeveloperTab);

})();
