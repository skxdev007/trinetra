// Simple Markdown Parser for SHARINGAN-DEEP
// Parses content.md and renders the website

class ContentParser {
    constructor() {
        this.sections = [];
        this.config = {};
    }

    async load() {
        try {
            const response = await fetch('content.md');
            const markdown = await response.text();
            this.parse(markdown);
            this.render();
        } catch (error) {
            console.error('Error loading content:', error);
            document.body.innerHTML = '<div class="container"><h1>Error loading content.md</h1></div>';
        }
    }

    parse(markdown) {
        // Split by section separators (---)
        const parts = markdown.split(/\n---\n/);
        
        parts.forEach(part => {
            const lines = part.trim().split('\n');
            if (lines.length === 0) return;

            // Check if it's a config section
            if (lines[0].startsWith('## Site Configuration')) {
                this.parseConfig(lines);
            } else if (lines[0].startsWith('##')) {
                this.parseSection(part);
            }
        });
    }

    parseConfig(lines) {
        lines.forEach(line => {
            const match = line.match(/\*\*(.+?):\*\*\s*(.+)/);
            if (match) {
                const key = match[1].toLowerCase().replace(/ /g, '-');
                this.config[key] = match[2];
            }
        });
    }

    parseSection(content) {
        const lines = content.trim().split('\n');
        const titleMatch = lines[0].match(/^##\s+(.+)/);
        
        if (!titleMatch) return;

        const fullTitle = titleMatch[1];
        let title = fullTitle;
        let type = 'general';

        // Check if it's an architecture component
        if (fullTitle.startsWith('Architecture:')) {
            type = 'architecture';
            title = fullTitle.replace('Architecture:', '').trim();
        } else if (fullTitle === 'Core Insight') {
            type = 'core-insight';
        } else if (fullTitle === 'Overview') {
            type = 'overview';
        } else if (fullTitle === 'Why This Works') {
            type = 'comparison';
        } else if (fullTitle === 'Performance') {
            type = 'performance';
        } else if (fullTitle === 'Use Cases') {
            type = 'use-cases';
        } else if (fullTitle === 'Technical Details') {
            type = 'technical';
        } else if (fullTitle === 'Getting Started') {
            type = 'getting-started';
        } else if (fullTitle === 'Future Directions') {
            type = 'future';
        } else if (fullTitle === 'Footer') {
            type = 'footer';
        }

        const section = {
            title,
            type,
            content: lines.slice(1).join('\n')
        };

        this.sections.push(section);
    }

    render() {
        try {
            // Set page metadata
            document.getElementById('page-title').textContent = `${this.config.title} | ${this.config.subtitle}`;
            document.getElementById('site-title').textContent = this.config.title;
            document.getElementById('site-subtitle').textContent = this.config.subtitle;
            document.getElementById('site-tagline').textContent = this.config.tagline;

            // Add architecture link if present
            if (this.config['architecture-link']) {
                const heroSection = document.querySelector('.hero');
                if (heroSection) {
                    const linkDiv = document.createElement('div');
                    linkDiv.className = 'architecture-link-container';
                    linkDiv.innerHTML = this.config['architecture-link'];
                    heroSection.appendChild(linkDiv);
                }
            }

            // Render sections
            this.sections.forEach(section => {
                try {
                    switch (section.type) {
                        case 'core-insight':
                            this.renderCoreInsight(section);
                            break;
                        case 'overview':
                            this.renderOverview(section);
                            break;
                        case 'architecture':
                            this.renderArchitecture(section);
                            break;
                        case 'comparison':
                            this.renderComparison(section);
                            break;
                        case 'performance':
                            this.renderPerformance(section);
                            break;
                        case 'use-cases':
                            this.renderUseCases(section);
                            break;
                        case 'technical':
                            this.renderTechnical(section);
                            break;
                        case 'getting-started':
                            this.renderGettingStarted(section);
                            break;
                        case 'future':
                            this.renderFuture(section);
                            break;
                        case 'footer':
                            this.renderFooter(section);
                            break;
                    }
                } catch (sectionError) {
                    console.error(`Error rendering section ${section.type}:`, sectionError);
                }
            });

            // Initialize Mermaid diagrams with error handling
            try {
                mermaid.run().catch(err => console.error('Mermaid error:', err));
            } catch (mermaidError) {
                console.error('Mermaid initialization error:', mermaidError);
            }
            
            document.body.classList.add('loaded');
        } catch (error) {
            console.error('Render error:', error);
            document.body.classList.add('loaded');
        }
    }

    renderCoreInsight(section) {
        document.getElementById('core-insight-title').textContent = section.title;
        
        const lines = section.content.trim().split('\n');
        let highlight = '';
        let explanation = '';
        let inQuote = false;

        lines.forEach(line => {
            if (line.startsWith('>')) {
                highlight += line.substring(1).trim() + ' ';
                inQuote = true;
            } else if (line.trim() && !inQuote) {
                explanation += line + ' ';
            } else if (!line.trim() && inQuote) {
                inQuote = false;
            }
        });

        document.getElementById('core-insight-highlight').textContent = highlight.trim();
        document.getElementById('core-insight-explanation').textContent = explanation.trim();
    }

    renderOverview(section) {
        document.getElementById('overview-title').textContent = section.title;
        
        const html = this.parseMarkdown(section.content);
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;

        // Extract intro (first paragraph before any diagram)
        const firstP = tempDiv.querySelector('p');
        if (firstP) {
            document.getElementById('overview-intro').innerHTML = firstP.innerHTML;
        }

        // Extract list
        const ul = tempDiv.querySelector('ul');
        if (ul) {
            document.getElementById('overview-features').innerHTML = ul.innerHTML;
        }

        // Extract conclusion (last paragraph after list, before diagram)
        const paragraphs = tempDiv.querySelectorAll('p');
        if (paragraphs.length > 1) {
            document.getElementById('overview-conclusion').innerHTML = paragraphs[paragraphs.length - 1].innerHTML;
        }
        
        // Add diagram if present
        const diagram = tempDiv.querySelector('.diagram-container');
        if (diagram) {
            const overviewSection = document.querySelector('.overview');
            overviewSection.appendChild(diagram);
        }
    }

    renderArchitecture(section) {
        const container = document.getElementById('architecture-components');
        if (!container.querySelector('h2')) {
            document.getElementById('architecture-title').textContent = 'Architecture Components';
        }

        const componentDiv = document.createElement('div');
        componentDiv.className = 'component';
        
        const html = this.parseMarkdown(section.content);
        componentDiv.innerHTML = `<h3>${section.title}</h3>${html}`;
        
        container.appendChild(componentDiv);
    }

    renderComparison(section) {
        document.getElementById('comparison-title').textContent = section.title;
        
        const lines = section.content.trim().split('\n');
        let reactivePoints = [];
        let proactivePoints = [];
        let conclusion = '';
        let currentSection = null;
        let inMermaid = false;

        lines.forEach(line => {
            // Skip mermaid diagram lines
            if (line.startsWith('```mermaid')) {
                inMermaid = true;
                return;
            }
            if (line.startsWith('```') && inMermaid) {
                inMermaid = false;
                return;
            }
            if (inMermaid) return;
            
            if (line.startsWith('### Reactive Models')) {
                currentSection = 'reactive';
            } else if (line.startsWith('### TRINETRA-DEEP')) {
                currentSection = 'proactive';
            } else if (line.startsWith('**The key insight:**')) {
                conclusion = line;
            } else if (line.startsWith('- ') && currentSection === 'reactive') {
                reactivePoints.push(line.substring(2));
            } else if (line.startsWith('- ') && currentSection === 'proactive') {
                proactivePoints.push(line.substring(2));
            }
        });

        document.getElementById('reactive-title').textContent = 'Reactive Models (Gemini, GPT-4o)';
        document.getElementById('proactive-title').textContent = 'TRINETRA-DEEP';

        const reactiveList = document.getElementById('reactive-points');
        reactivePoints.forEach(point => {
            const li = document.createElement('li');
            li.textContent = point;
            reactiveList.appendChild(li);
        });

        const proactiveList = document.getElementById('proactive-points');
        proactivePoints.forEach(point => {
            const li = document.createElement('li');
            li.textContent = point;
            proactiveList.appendChild(li);
        });

        document.getElementById('comparison-conclusion').innerHTML = this.parseMarkdown(conclusion);
        
        // Add diagram if present
        const html = this.parseMarkdown(section.content);
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;
        const diagram = tempDiv.querySelector('.diagram-container');
        if (diagram) {
            const comparisonSection = document.querySelector('.why-it-works');
            comparisonSection.appendChild(diagram);
        }
    }

    renderPerformance(section) {
        document.getElementById('results-title').textContent = section.title;
        
        const lines = section.content.trim().split('\n');
        document.getElementById('results-intro').textContent = lines[0];

        const metricsContainer = document.getElementById('results-metrics');
        lines.slice(2).forEach(line => {
            const parts = line.split('|').map(p => p.trim());
            if (parts.length === 3) {
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = `
                    <div class="result-number">${parts[0]}</div>
                    <div class="result-label">${parts[1]}</div>
                    <div class="result-note">${parts[2]}</div>
                `;
                metricsContainer.appendChild(card);
            }
        });
    }

    renderUseCases(section) {
        document.getElementById('use-cases-title').textContent = section.title;
        
        const lines = section.content.trim().split('\n');
        const container = document.getElementById('use-cases-list');
        
        let currentCase = null;
        lines.forEach(line => {
            if (line.startsWith('###')) {
                if (currentCase) {
                    container.appendChild(currentCase);
                }
                currentCase = document.createElement('div');
                currentCase.className = 'use-case';
                currentCase.innerHTML = `<h4>${line.substring(3).trim()}</h4>`;
            } else if (line.trim() && currentCase) {
                const p = document.createElement('p');
                p.textContent = line;
                currentCase.appendChild(p);
            }
        });
        if (currentCase) {
            container.appendChild(currentCase);
        }
    }

    renderTechnical(section) {
        document.getElementById('technical-title').textContent = section.title;
        
        const lines = section.content.trim().split('\n');
        const container = document.getElementById('technical-specs');
        
        lines.forEach(line => {
            const match = line.match(/\*\*(.+?):\*\*\s*(.+)/);
            if (match) {
                const div = document.createElement('div');
                div.className = 'tech-item';
                div.innerHTML = `<strong>${match[1]}:</strong> ${match[2]}`;
                container.appendChild(div);
            }
        });
    }

    renderGettingStarted(section) {
        document.getElementById('getting-started-title').textContent = section.title;
        
        const codeMatch = section.content.match(/```python\n([\s\S]+?)\n```/);
        if (codeMatch) {
            document.getElementById('getting-started-code').textContent = codeMatch[1];
        }
    }

    renderFuture(section) {
        document.getElementById('future-title').textContent = section.title;
        
        const html = this.parseMarkdown(section.content);
        document.getElementById('future-directions').innerHTML = html.match(/<ul>([\s\S]+?)<\/ul>/)[1];
    }

    renderFooter(section) {
        const lines = section.content.trim().split('\n');
        
        lines.forEach(line => {
            if (line.startsWith('**Tagline:**')) {
                document.getElementById('footer-tagline').textContent = line.replace('**Tagline:**', '').trim();
            } else if (line.startsWith('- [')) {
                const links = [];
                const linkMatches = section.content.matchAll(/\[(.+?)\]\((.+?)\)/g);
                for (const match of linkMatches) {
                    links.push(`<a href="${match[2]}">${match[1]}</a>`);
                }
                document.getElementById('footer-links').innerHTML = links.join(' | ');
            }
        });
    }

    parseMarkdown(text) {
        // Simple markdown parser
        let html = text;

        // Code blocks
        html = html.replace(/```mermaid\n([\s\S]+?)\n```/g, (match, code) => {
            return `<div class="diagram-container"><div class="mermaid">${code}</div></div>`;
        });

        html = html.replace(/```(\w+)?\n([\s\S]+?)\n```/g, (match, lang, code) => {
            return `<div class="code-block"><pre><code>${this.escapeHtml(code)}</code></pre></div>`;
        });

        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Lists
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

        // Paragraphs
        html = html.split('\n\n').map(para => {
            if (!para.startsWith('<') && para.trim()) {
                return `<p>${para}</p>`;
            }
            return para;
        }).join('\n');

        return html;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize Mermaid
mermaid.initialize({ 
    startOnLoad: false,
    theme: 'base',
    themeVariables: {
        primaryColor: '#e3f2fd',
        primaryTextColor: '#1a237e',
        primaryBorderColor: '#90caf9',
        lineColor: '#64b5f6',
        secondaryColor: '#fff3e0',
        tertiaryColor: '#e8f5e9',
        fontSize: '14px'
    }
});

// Load content when page loads
document.addEventListener('DOMContentLoaded', () => {
    const parser = new ContentParser();
    parser.load();
});
