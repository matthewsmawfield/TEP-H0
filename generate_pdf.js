#!/usr/bin/env node
/**
 * PDF generator for MNRAS submission.
 * Serves dist/journal.html locally, waits for MathJax to finish,
 * then prints to PDF with correct A4 sizing and no browser headers/footers.
 */

const puppeteer = require('puppeteer');
const http = require('http');
const fs = require('fs');
const path = require('path');

const DIST_DIR = path.join(__dirname, 'site', 'dist');
const OUTPUT_PDF = path.join(__dirname, 'Smawfield_2026_CepheidBias_MNRAS_submission.pdf');
const PORT = 8765;

// Minimal static file server for the dist directory
function startServer() {
    return new Promise((resolve) => {
        const server = http.createServer((req, res) => {
            let filePath = path.join(DIST_DIR, req.url === '/' ? 'journal.html' : req.url);
            // Prevent directory traversal
            if (!filePath.startsWith(DIST_DIR)) {
                res.writeHead(403); res.end(); return;
            }
            fs.readFile(filePath, (err, data) => {
                if (err) { res.writeHead(404); res.end(); return; }
                const ext = path.extname(filePath);
                const mime = {
                    '.html': 'text/html', '.css': 'text/css',
                    '.js': 'application/javascript', '.json': 'application/json',
                    '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml'
                }[ext] || 'application/octet-stream';
                res.writeHead(200, { 'Content-Type': mime });
                res.end(data);
            });
        });
        server.listen(PORT, '127.0.0.1', () => {
            console.log(`Server running at http://127.0.0.1:${PORT}`);
            resolve(server);
        });
    });
}

async function generatePDF() {
    const server = await startServer();

    const browser = await puppeteer.launch({
        headless: 'new',
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });

    try {
        const page = await browser.newPage();

        // Set A4 viewport
        await page.setViewport({ width: 794, height: 1123, deviceScaleFactor: 1 });

        console.log('Loading page...');
        await page.goto(`http://127.0.0.1:${PORT}/journal.html`, {
            waitUntil: 'networkidle0',
            timeout: 60000
        });

        // Wait for manuscript content to load (fetch-based)
        await page.waitForSelector('#manuscript-content', { timeout: 30000 });
        await page.waitForFunction(
            () => document.getElementById('manuscript-content').style.display !== 'none',
            { timeout: 30000 }
        );

        // Wait for MathJax to finish typesetting
        console.log('Waiting for MathJax...');
        await page.waitForFunction(
            () => {
                if (!window.MathJax || !window.MathJax.typesetPromise) return false;
                // Check MathJax has processed — look for rendered mjx-container elements
                return document.querySelectorAll('mjx-container').length > 5;
            },
            { timeout: 30000 }
        );

        // Extra settle time for MathJax to fully render all equations
        await new Promise(r => setTimeout(r, 3000));

        console.log('Generating PDF...');
        await page.pdf({
            path: OUTPUT_PDF,
            format: 'A4',
            printBackground: true,
            displayHeaderFooter: true,
            headerTemplate: '<span></span>',
            footerTemplate: `<div style="width:100%; text-align:center; font-size:9pt; font-family:'Times New Roman',serif; color:#444; padding-bottom:4mm;">
                <span class="pageNumber"></span> of <span class="totalPages"></span>
            </div>`,
            margin: {
                top: '20mm',
                bottom: '22mm',
                left: '20mm',
                right: '20mm'
            },
            scale: 0.85   // slight scale-down to fit wide content properly
        });

        console.log(`✅ PDF written to: ${OUTPUT_PDF}`);
    } finally {
        await browser.close();
        server.close();
    }
}

generatePDF().catch(err => {
    console.error('PDF generation failed:', err);
    process.exit(1);
});
