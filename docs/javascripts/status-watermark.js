/*
 * Copyright (c) 2025, Stochastic Analytics, LLC
 *
 * This file is part of cisTEMx.
 *
 * For academic and non-profit use, this file is licensed under the
 * Mozilla Public License Version 2.0. See license_details/LICENSE-MPL-2.0.txt.
 *
 * Commercial use requires a separate license from Stochastic Analytics, LLC.
 * Contact: <commercial-license@clearnoise.org>
 */

/**
 * Status Watermark System for cisTEMx Documentation
 *
 * Reads page status from frontmatter metadata and applies data-status attribute
 * to the article element for CSS watermarking.
 *
 * Status values trigger different watermark styles:
 * - legacy: Amber/yellow watermark (exclusive color for legacy content)
 * - deprecated: Orange watermark
 * - draft: Blue watermark
 * - placeholder: Gray watermark
 */

document.addEventListener('DOMContentLoaded', function() {
  // Look for status metadata in page head
  const metaStatus = document.querySelector('meta[name="status"]');

  if (metaStatus) {
    const status = metaStatus.content.toLowerCase();
    const validStatuses = ['draft', 'legacy', 'deprecated', 'placeholder'];

    if (validStatuses.includes(status)) {
      // Apply data-status to article element for CSS targeting
      const article = document.querySelector('.md-content article');
      if (article) {
        article.setAttribute('data-status', status);

        // Log for debugging (can be removed in production)
        console.log(`[Status Watermark] Applied status: ${status}`);
      } else {
        console.warn('[Status Watermark] Could not find article element');
      }
    } else {
      console.warn(`[Status Watermark] Invalid status value: ${status}`);
    }
  }
  // No warning if meta tag not found - most pages won't have status
});
