#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

class HTMLToMarkdownConverter {
    constructor() { this.output = ''; }

    htmlToMarkdown(html) {
        const stripTags = (s) => {
            if (s == null) return '';
            return String(s)
                .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
                .replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')
                // Only strip actual tags like <p>, </div>, <img ...>. Do NOT treat
                // literal '<' in math/text (e.g. '$R < 5$') as an HTML tag.
                .replace(/<\/?[A-Za-z][^>]*>/g, '')
                .replace(/\s+/g, ' ')
                .trim();
        };

        const escapePipes = (s) => String(s).replace(/\|/g, '\\|');

        const tableToMarkdown = (tableHtml) => {
            const rows = [];
            const trRe = /<tr[^>]*>([\s\S]*?)<\/tr>/gi;
            let tr;
            while ((tr = trRe.exec(tableHtml)) !== null) {
                const rowHtml = tr[1];
                const cells = [];
                const cellRe = /<(th|td)[^>]*>([\s\S]*?)<\/(th|td)>/gi;
                let cell;
                while ((cell = cellRe.exec(rowHtml)) !== null) {
                    const raw = stripTags(cell[2]);
                    cells.push(escapePipes(raw));
                }
                if (cells.length) rows.push(cells);
            }

            if (!rows.length) return stripTags(tableHtml);

            const header = rows[0];
            const colCount = header.length;
            const padRow = (r) => {
                const rr = r.slice(0, colCount);
                while (rr.length < colCount) rr.push('');
                return rr;
            };

            const headerRow = `| ${padRow(header).join(' | ')} |`;
            const sepRow = `| ${Array(colCount).fill('---').join(' | ')} |`;
            const bodyRows = rows.slice(1).map((r) => `| ${padRow(r).join(' | ')} |`);

            return `\n\n${[headerRow, sepRow, ...bodyRows].join('\n')}\n\n`;
        };

        html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
        html = html.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');
        html = html.replace(/<!--[\s\S]*?-->/g, '');

        // Tables (convert before other tag stripping)
        html = html.replace(/<table[^>]*>[\s\S]*?<\/table>/gi, (m) => tableToMarkdown(m));

        // Images
        html = html.replace(/<img[^>]*src=["']([^"']+)["'][^>]*alt=["']([^"']*)["'][^>]*>/gi, '![$2]($1)');
        html = html.replace(/<img[^>]*alt=["']([^"']*)["'][^>]*src=["']([^"']+)["'][^>]*>/gi, '![$1]($2)');
        html = html.replace(/<img[^>]*src=["']([^"']+)["'][^>]*>/gi, '![]($1)');

        // Section boundaries (static build uses <section>, older versions used <div>)
        // Do not emit headings here: components already contain their own <h2>/<h3>
        // and emitting wrappers causes duplicated headings in the generated Markdown.
        html = html.replace(/<section[^>]*class=["']manuscript-section[^"']*["'][^>]*data-section=["']([^"']*)["'][^>]*>/gi, '\n\n');
        html = html.replace(/<div[^>]*class=["']manuscript-section[^"']*["'][^>]*data-section=["']([^"']*)["'][^>]*>/gi, '\n\n');

        html = html.replace(/<div[^>]*id=["']code-availability["'][^>]*>/gi, '\n> ');
        html = html.replace(/<div[^>]*class=["']key-finding[^"']*["'][^>]*>/gi, '\n> ');
        html = html.replace(/<h1[^>]*>(.*?)<\/h1>/gi, '\n# $1\n\n');
        html = html.replace(/<h2[^>]*>(.*?)<\/h2>/gi, '\n## $1\n\n');
        html = html.replace(/<h3[^>]*>(.*?)<\/h3>/gi, '\n### $1\n\n');
        html = html.replace(/<h4[^>]*>(.*?)<\/h4>/gi, '\n#### $1\n\n');
        // Paragraphs - trim content to remove indentation from HTML formatting
        html = html.replace(/<p[^>]*>([\s\S]*?)<\/p>/gi, (match, content) => {
            const trimmed = content.replace(/^\s+/gm, '').replace(/\s+$/gm, '').trim();
            return trimmed ? trimmed + '\n\n' : '';
        });
        html = html.replace(/<a[^>]*href=["']([^"']*)["'][^>]*>(.*?)<\/a>/gi, '[$2]($1)');
        html = html.replace(/<(strong|b)[^>]*>(.*?)<\/(strong|b)>/gi, '**$2**');
        html = html.replace(/<(em|i)[^>]*>(.*?)<\/(em|i)>/gi, '*$2*');
        html = html.replace(/<li[^>]*>(.*?)<\/li>/gi, (match, content) => {
            const trimmed = content.replace(/^\s+/gm, '').replace(/\s+$/gm, '').trim();
            return '- ' + trimmed + '\n';
        });
        html = html.replace(/<\/?[A-Za-z][^>]*>/g, '');
        // Clean up: remove indentation from lines, collapse multiple blank lines
        html = html.replace(/^[ \t]+/gm, '');
        return html.replace(/\n{3,}/g, '\n\n').trim();
    }

    async convertSiteToMarkdown() {
        console.log('🔄 Converting HTML site to markdown (TEP-H0)...');
        try {
            const htmlPath = path.join(__dirname, 'dist', 'index.html');
            if (!fs.existsSync(htmlPath)) throw new Error('Built HTML file not found.');
            const html = fs.readFileSync(htmlPath, 'utf8');
            const mainMatch = html.match(/<div[^>]*id=["']manuscript-content["'][^>]*>([\s\S]*?)<\/main>/i);
            if (!mainMatch) throw new Error('Could not find manuscript content.');
            
            const today = new Date().toISOString().split('T')[0];
            const [year, month, day] = today.split('-');
            const monthNames = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
            const formattedDate = `${parseInt(day)} ${monthNames[parseInt(month) - 1]} ${year}`;
            const header = `# The Cepheid Bias: Resolving the Hubble Tension
**Matthew Lukin Smawfield**  
Version: v0.7 (Kingston upon Hull)  
First published: 11 January 2026 · Last updated: ${formattedDate}  
DOI: 10.5281/zenodo.18209702

---

`;
            const markdown = header + this.htmlToMarkdown(mainMatch[1]);
            const versionInfo = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'VERSION.json'), 'utf8'));
            const version = versionInfo.version;
            const outputPath = path.join(__dirname, '..', `11-TEP-H0-v${version}-KingstonUponHull.md`);
            fs.writeFileSync(outputPath, markdown, 'utf8');
            console.log(`✅ Markdown saved to: ${outputPath}`);
        } catch (error) {
            console.error('❌ Markdown conversion failed:', error.message);
        }
    }
}

if (require.main === module) { const c = new HTMLToMarkdownConverter(); c.convertSiteToMarkdown(); }
module.exports = { HTMLToMarkdownConverter };
