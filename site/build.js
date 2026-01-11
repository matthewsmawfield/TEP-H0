#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

function safeRemoveDirSync(dirPath) {
    if (!fs.existsSync(dirPath)) return;
    try {
        fs.rmSync(dirPath, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
    } catch (err) {
        // In dev mode, live-reload servers can race with filesystem deletes.
        // Fallback: move the directory out of the way, then attempt deletion.
        const fallbackName = `${dirPath}_old_${Date.now()}`;
        try {
            fs.renameSync(dirPath, fallbackName);
        } catch (renameErr) {
            throw err;
        }
        try {
            fs.rmSync(fallbackName, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
        } catch (_) {
            // Best-effort cleanup; ignore.
        }
    }
}

async function buildStaticSite() {
    console.log('🔨 Building static site...');
    
    try {
        // Clean dist directory
        const distDir = path.join(__dirname, 'dist');
        if (fs.existsSync(distDir)) {
            console.log('🧹 Cleaning dist directory...');
            safeRemoveDirSync(distDir);
        }
        
        // Read the manifest
        const manifestPath = path.join(__dirname, 'manifest.json');
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
        
        // Read the base index.html
        const indexPath = path.join(__dirname, 'index.html');
        let indexContent = fs.readFileSync(indexPath, 'utf8');
        
        // Build the component content
        let componentsHtml = '';
        
        // Sort sections by order and load each component
        const sortedSections = manifest.sections.sort((a, b) => a.order - b.order);
        
        for (const section of sortedSections) {
            console.log(`📄 Loading section: ${section.title}`);
            
            const componentPath = path.join(__dirname, 'components', section.file);
            
            if (fs.existsSync(componentPath)) {
                const componentHtml = fs.readFileSync(componentPath, 'utf8');
                
                // Wrap component in section container (matching the dynamic loader)
                componentsHtml += `
                <section id="${section.id}" class="manuscript-section" data-section="${section.title}">
                    ${componentHtml}
                </section>`;
            } else {
                console.warn(`⚠️  Component not found: ${section.file}`);
                componentsHtml += `
                <section id="${section.id}" class="manuscript-section" data-section="${section.title}">
                    <div style="background-color: #ffe6e6; border: 1px solid #ff9999; padding: 15px; margin: 20px 0; border-radius: 5px;">
                        <h3 style="color: #cc0000; margin-top: 0;">Missing Section: ${section.title}</h3>
                        <p>Component file <code>components/${section.file}</code> not found</p>
                    </div>
                </section>`;
            }
        }
        
        // Replace the dynamic content with static content
        const staticContent = indexContent
            .replace(
                /<div id="loading".*?<\/div>\s*<div id="manuscript-content".*?<\/div>/s,
                `<div id="manuscript-content">${componentsHtml}</div>`
            )
            .replace(/<div id="loading"[^>]*>[\s\S]*?<\/div>/g, '')
            .replace(/<div id="manuscript-content"[^>]*style="display:\s*none;"[^>]*>[\s\S]*?<\/div>/g, '')
            .replace(
                /<!-- Component Loading Script -->[\s\S]*?<\/script>/,
                '<!-- Static build - components pre-loaded -->'
            )
            .replace(
                '<main id="main-content" role="main">',
                '<!-- This is a statically built version for SEO/deployment -->\n    <main id="main-content" role="main">'
            );
        
        // Create dist directory if it doesn't exist
        if (!fs.existsSync(distDir)) {
            fs.mkdirSync(distDir, { recursive: true });
        }
        
        // Write the built file
        const outputPath = path.join(distDir, 'index.html');
        fs.writeFileSync(outputPath, staticContent, 'utf8');
        
        // Copy necessary static assets to dist
        // EXPLICITLY COPY STYLES DIRECTORY
        const assetDirs = ['public', 'figures', 'data', 'styles'];
        for (const assetDir of assetDirs) {
            const srcPath = path.join(__dirname, assetDir);
            const destPath = path.join(distDir, assetDir);
            
            if (fs.existsSync(srcPath)) {
                console.log(`📁 Copying ${assetDir}/`);
                copyRecursiveSync(srcPath, destPath);
            }
        }

        // Copy simple-styles.css if it exists
        const simpleStylesSrc = path.join(__dirname, 'simple-styles.css');
        const simpleStylesDest = path.join(distDir, 'simple-styles.css');
        if (fs.existsSync(simpleStylesSrc)) {
            fs.copyFileSync(simpleStylesSrc, simpleStylesDest);
            console.log('📁 Copied simple-styles.css');
        }
        
        // Copy figures from results/figures/ to dist/figures/ (main figure source)
        const resultsFiguresPath = path.join(__dirname, '..', 'results', 'figures');
        const distFiguresPath = path.join(distDir, 'figures');
        if (fs.existsSync(resultsFiguresPath)) {
            console.log('📁 Copying results/figures/ → dist/figures/');
            copyRecursiveSync(resultsFiguresPath, distFiguresPath);
        }
        
        // Copy manifest.json for reference
        fs.copyFileSync(manifestPath, path.join(distDir, 'manifest.json'));
        
        // Copy .nojekyll to dist root for GitHub Pages
        const nojekyllSrc = path.join(__dirname, 'public', '.nojekyll');
        const nojekyllDest = path.join(distDir, '.nojekyll');
        if (fs.existsSync(nojekyllSrc)) {
            fs.copyFileSync(nojekyllSrc, nojekyllDest);
            console.log('📁 Copied .nojekyll to dist root');
        } else {
            // Create it if it doesn't exist
            fs.writeFileSync(nojekyllDest, '');
            console.log('📁 Created .nojekyll in dist root');
        }

        // Copy robots.txt and sitemap.xml to dist root
        const rootFiles = ['404.html', 'robots.txt', 'sitemap.xml', 'CNAME', '29c6507763d2303d801cc8ed89d39f88.txt'];
        for (const file of rootFiles) {
            const src = path.join(__dirname, 'public', file);
            const dest = path.join(distDir, file);
            if (fs.existsSync(src)) {
                fs.copyFileSync(src, dest);
                console.log(`📁 Copied ${file} to dist root`);
            }
        }
        
        // Copy citation files to dist root
        const citationFiles = ['CITATION.cff', 'CITATION.bib', 'citation.json', 'codemeta.json'];
        for (const file of citationFiles) {
            const src = path.join(__dirname, file);
            const dest = path.join(distDir, file);
            if (fs.existsSync(src)) {
                fs.copyFileSync(src, dest);
                console.log(`📁 Copied ${file} to dist root`);
            }
        }
        
        // Copy .well-known directory
        const wellKnownSrc = path.join(__dirname, 'public', '.well-known');
        const wellKnownDest = path.join(distDir, '.well-known');
        if (fs.existsSync(wellKnownSrc)) {
            if (!fs.existsSync(wellKnownDest)) {
                fs.mkdirSync(wellKnownDest, { recursive: true });
            }
            const files = fs.readdirSync(wellKnownSrc);
            for (const file of files) {
                fs.copyFileSync(path.join(wellKnownSrc, file), path.join(wellKnownDest, file));
            }
            console.log(`📁 Copied .well-known/ to dist`);
        }
        
        // Generate markdown version
        console.log('📝 Generating markdown version...');
        const { HTMLToMarkdownConverter } = require('./html-to-markdown.js');
        const converter = new HTMLToMarkdownConverter();
        await converter.convertSiteToMarkdown();
        
        console.log('✅ Static site built successfully!');
        console.log(`📁 Output: ${outputPath}`);
        console.log(`📊 Generated ${manifest.sections.length} sections`);
        console.log('🚀 Ready for deployment');
        
    } catch (error) {
        console.error('❌ Build failed:', error);
        process.exit(1);
    }
}

// Helper function to copy directories recursively
function copyRecursiveSync(src, dest) {
    const exists = fs.existsSync(src);
    const stats = exists && fs.statSync(src);
    const isDirectory = exists && stats.isDirectory();
    
    if (isDirectory) {
        if (!fs.existsSync(dest)) {
            fs.mkdirSync(dest, { recursive: true });
        }
        fs.readdirSync(src).forEach(childItemName => {
            copyRecursiveSync(
                path.join(src, childItemName),
                path.join(dest, childItemName)
            );
        });
    } else {
        // Filter out raw data files
        if (src.endsWith('.csv') || src.endsWith('.dat') || src.endsWith('.nc')) {
            return;
        }
        fs.copyFileSync(src, dest);
    }
}

// Run if called directly
if (require.main === module) {
    buildStaticSite();
}

module.exports = { buildStaticSite };
