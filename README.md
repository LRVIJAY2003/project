<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Policies Search</title>
    <link rel="stylesheet" href="css/main.css">
    <link rel="stylesheet" href="css/search.css">
    <link rel="stylesheet" href="css/report.css">
</head>
<body>
    <div class="background-animation"></div>
    
    <header>
        <h1>Network Policies</h1>
        <div class="search-container">
            <input type="text" id="search-input" placeholder="Search policies...">
            <button id="search-button">Search</button>
        </div>
    </header>
    
    <main>
        <section class="search-patterns">
            <h2>Define Search Pattern</h2>
            <p>Select from predefined patterns or manually define your own</p>
            
            <div class="pattern-boxes">
                <div class="pattern-box" data-pattern="GRAPH">GRAPH</div>
                <div class="pattern-box" data-pattern="JSON">JSON</div>
                <div class="pattern-box" data-pattern="YAML">YAML</div>
                <div class="pattern-box" data-pattern="XML">XML</div>
                <div class="pattern-box" data-pattern="CSV">CSV</div>
                <div class="pattern-box" data-pattern="TEXT">TEXT</div>
            </div>
            
            <div class="manual-pattern">
                <h3>Manual Pattern</h3>
                <input type="text" id="manual-pattern" placeholder="Enter custom search pattern...">
                <button id="add-pattern">Add Pattern</button>
            </div>
            
            <div class="selected-patterns">
                <h3>Selected Patterns</h3>
                <ul id="pattern-list"></ul>
            </div>
        </section>
        
        <section class="report-container" id="report-container">
            <h2>Search Results</h2>
            <div class="report-info">
                <p>Click on a result to view details</p>
            </div>
            <div class="report-content" id="report-content">
                <!-- Report content will be dynamically generated here -->
            </div>
        </section>
    </main>
    
    <footer>
        <p>Network Policies Search Tool &copy; 2025</p>
    </footer>
    
    <script src="js/app.js"></script>
    <script src="js/search.js"></script>
    <script src="js/report.js"></script>
</body>
</html>









/* main.css - Main styles for the application */

:root {
    --primary-color: #3a86ff;
    --secondary-color: #8338ec;
    --accent-color: #ff006e;
    --background-color: #f8f9fa;
    --text-color: #212529;
    --border-color: #dee2e6;
    --success-color: #28a745;
    --warning-color: #ffc107;
    --error-color: #dc3545;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: var(--text-color);
    background-color: var(--background-color);
    min-height: 100vh;
    position: relative;
    overflow-x: hidden;
}

.background-animation {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    background: linear-gradient(120deg, rgba(58, 134, 255, 0.05) 0%, rgba(131, 56, 236, 0.05) 100%);
    overflow: hidden;
}

.background-animation::before {
    content: '';
    position: absolute;
    width: 200%;
    height: 200%;
    top: -50%;
    left: -50%;
    background-image: linear-gradient(0deg, transparent 24%, rgba(58, 134, 255, 0.1) 25%, rgba(58, 134, 255, 0.1) 26%, transparent 27%, transparent 74%, rgba(58, 134, 255, 0.1) 75%, rgba(58, 134, 255, 0.1) 76%, transparent 77%, transparent),
                      linear-gradient(90deg, transparent 24%, rgba(58, 134, 255, 0.1) 25%, rgba(58, 134, 255, 0.1) 26%, transparent 27%, transparent 74%, rgba(58, 134, 255, 0.1) 75%, rgba(58, 134, 255, 0.1) 76%, transparent 77%, transparent);
    background-size: 50px 50px;
    animation: moveBackground 30s linear infinite;
}

@keyframes moveBackground {
    0% {
        transform: rotate(0deg) scale(1.1);
    }
    50% {
        transform: rotate(1deg) scale(1.2);
    }
    100% {
        transform: rotate(0deg) scale(1.1);
    }
}

header {
    background-color: white;
    padding: 1.5rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    position: relative;
    z-index: 10;
}

header h1 {
    color: var(--primary-color);
    margin-bottom: 1rem;
    font-weight: 600;
}

main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 2rem;
}

section {
    background-color: white;
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

h2 {
    color: var(--secondary-color);
    margin-bottom: 1rem;
    font-weight: 600;
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 0.5rem;
}

h3 {
    color: var(--primary-color);
    margin: 1rem 0 0.5rem;
    font-weight: 500;
}

p {
    color: #666;
    margin-bottom: 1rem;
}

button {
    background-color: var(--primary-color);
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.2s;
    font-weight: 500;
}

button:hover {
    background-color: #2a75e6;
}

button:active {
    transform: translateY(1px);
}

input[type="text"] {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    font-size: 1rem;
}

input[type="text"]:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 2px rgba(58, 134, 255, 0.2);
}

footer {
    text-align: center;
    padding: 1.5rem;
    margin-top: 2rem;
    color: #666;
    border-top: 1px solid var(--border-color);
}

/* Responsive design */
@media (max-width: 768px) {
    main {
        grid-template-columns: 1fr;
    }
    
    .search-patterns {
        order: 1;
    }
    
    .report-container {
        order: 2;
    }
}







/* search.css - Styles for the search functionality */

.search-container {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.search-container input {
    flex: 1;
}

.search-container button {
    width: auto;
}

.pattern-boxes {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    gap: 0.75rem;
    margin: 1rem 0;
}

.pattern-box {
    background-color: #f0f4ff;
    border: 1px solid #d0d8ff;
    color: var(--primary-color);
    padding: 1rem 0.5rem;
    text-align: center;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
    font-weight: 500;
}

.pattern-box:hover {
    background-color: #e0e8ff;
    transform: translateY(-2px);
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.pattern-box.selected {
    background-color: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}

.manual-pattern {
    margin: 1.5rem 0;
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 6px;
    border: 1px dashed var(--border-color);
}

.manual-pattern h3 {
    margin-top: 0;
}

.manual-pattern input {
    margin: 0.5rem 0;
}

.selected-patterns {
    margin-top: 1.5rem;
}

#pattern-list {
    list-style: none;
    margin-top: 0.5rem;
}

#pattern-list li {
    background-color: #f0f4ff;
    padding: 0.5rem 1rem;
    margin-bottom: 0.5rem;
    border-radius: 4px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

#pattern-list li .pattern-text {
    font-family: 'Courier New', monospace;
    color: var(--secondary-color);
}

#pattern-list li .remove-pattern {
    background-color: transparent;
    color: var(--error-color);
    padding: 0.25rem 0.5rem;
    font-size: 0.875rem;
}

.pattern-hint {
    font-size: 0.875rem;
    color: #666;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background-color: #fff9e6;
    border-left: 3px solid var(--warning-color);
}






/* report.css - Styles for the report display */

.report-container {
    display: flex;
    flex-direction: column;
}

.report-info {
    margin-bottom: 1rem;
    padding: 0.75rem;
    background-color: #f0f4ff;
    border-radius: 4px;
    border-left: 3px solid var(--primary-color);
}

.report-content {
    min-height: 300px;
    border: 1px solid var(--border-color);
    border-radius: 4px;
    overflow: hidden;
}

.report-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 300px;
    color: #aaa;
    text-align: center;
    padding: 1rem;
}

.report-placeholder svg {
    width: 64px;
    height: 64px;
    margin-bottom: 1rem;
    color: #ddd;
}

.policy-item {
    padding: 1rem;
    border-bottom: 1px solid var(--border-color);
    transition: background-color 0.2s;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.policy-item:hover {
    background-color: #f8f9fa;
}

.policy-item:last-child {
    border-bottom: none;
}

.policy-number {
    background-color: var(--primary-color);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    flex-shrink: 0;
}

.policy-details {
    flex: 1;
}

.policy-title {
    font-weight: 500;
    margin-bottom: 0.25rem;
    color: var(--text-color);
}

.policy-url {
    font-size: 0.875rem;
    color: var(--primary-color);
    word-break: break-all;
}

.policy-date {
    font-size: 0.75rem;
    color: #666;
    margin-left: auto;
    text-align: right;
    flex-shrink: 0;
}

.policy-detail-view {
    padding: 1.5rem;
    background-color: #f8f9fa;
    border-top: 1px solid var(--border-color);
    display: none;
}

.policy-detail-view.active {
    display: block;
}

.detail-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
}

.detail-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--secondary-color);
}

.detail-close {
    background-color: transparent;
    color: #666;
    padding: 0.25rem;
    font-size: 1.25rem;
}

.detail-content {
    border: 1px solid var(--border-color);
    border-radius: 4px;
    padding: 1rem;
    background-color: white;
    max-height: 400px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 0.875rem;
    line-height: 1.5;
}

.detail-actions {
    margin-top: 1rem;
    display: flex;
    gap: 0.5rem;
    justify-content: flex-end;
}

.loading-indicator {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 300px;
}

.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid rgba(58, 134, 255, 0.1);
    border-left-color: var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

/* Status indicators */
.status-badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-left: 0.5rem;
}

.status-active {
    background-color: rgba(40, 167, 69, 0.1);
    color: var(--success-color);
}

.status-deprecated {
    background-color: rgba(220, 53, 69, 0.1);
    color: var(--error-color);
}

.status-draft {
    background-color: rgba(255, 193, 7, 0.1);
    color: var(--warning-color);
}










/**
 * app.js - Main application initialization and event listeners
 * 
 * This file handles the overall application logic and initialization,
 * connecting the search and report components together.
 */

// Mock network policies data that would normally come from a backend
const NETWORK_POLICIES = [
    {
        id: 728,
        title: "Network Policy - GKE-Default",
        url: "https://console.example.com/rest/api/v1.0/projects/GKE/repo/gke-rs-network-policies/browse/k8s/kms/pr/app-namespaces/f/tmp/app-728-gke-default/networkpolicy-apps-fqdn.yaml",
        type: "YAML",
        date: "2025-01-15T14:32:17Z",
        content: "apiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n  name: default-deny-all\n  namespace: gke-system\nspec:\n  podSelector: {}\n  policyTypes:\n  - Ingress\n  - Egress",
        status: "active"
    },
    {
        id: 196,
        title: "Network Policy - Apps-FQDN",
        url: "https://console.example.com/rest/api/v1.0/projects/GKE/repo/gke-rs-network-policies/browse/k8s/kms/pr/app-namespaces/f/tmp/app-196-gke-default/networkpolicy-apps-fqdn.yaml",
        type: "YAML",
        date: "2025-02-03T09:15:42Z",
        content: "apiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n  name: allow-app-egress\n  namespace: app-namespace\nspec:\n  podSelector:\n    matchLabels:\n      app: frontend\n  policyTypes:\n  - Egress\n  egress:\n  - to:\n    - ipBlock:\n        cidr: 10.0.0.0/8",
        status: "active"
    },
    {
        id: 197,
        title: "Network Policy - Apps-FQDN-yaml",
        url: "https://console.example.com/rest/api/v1.0/projects/GKE/repo/gke-rs-network-policies/browse/k8s/kms/pr/app-namespaces/f/tmp/app-197-gke-default/values.yaml",
        type: "YAML",
        date: "2025-02-10T11:22:33Z",
        content: "networkPolicies:\n  enabled: true\n  defaultDenyIngress: true\n  defaultDenyEgress: false\n  ingress:\n    - from:\n      - namespaceSelector:\n          matchLabels:\n            name: monitoring\n      - podSelector:\n          matchLabels:\n            app: prometheus",
        status: "draft"
    },
    {
        id: 345,
        title: "Network Policy - JSON",
        url: "https://console.example.com/rest/api/v1.0/projects/GKE/repo/gke-rs-network-policies/browse/k8s/kms/pr/app-namespaces/f/tmp/app-345-json/policy.json",
        type: "JSON",
        date: "2025-02-15T16:45:22Z",
        content: "{\n  \"apiVersion\": \"networking.k8s.io/v1\",\n  \"kind\": \"NetworkPolicy\",\n  \"metadata\": {\n    \"name\": \"allow-api-ingress\",\n    \"namespace\": \"api-namespace\"\n  },\n  \"spec\": {\n    \"podSelector\": {\n      \"matchLabels\": {\n        \"role\": \"api\"\n      }\n    },\n    \"ingress\": [\n      {\n        \"from\": [\n          {\n            \"podSelector\": {\n              \"matchLabels\": {\n                \"role\": \"frontend\"\n              }\n            }\n          }\n        ],\n        \"ports\": [\n          {\n            \"port\": 8080,\n            \"protocol\": \"TCP\"\n          }\n        ]\n      }\n    ]\n  }\n}",
        status: "active"
    },
    {
        id: 903,
        title: "Network Policy - Graph-def",
        url: "https://console.example.com/rest/api/v1.0/projects/GKE/repo/gke-rs-network-policies/browse/k8s/kms/pr/app-namespaces/f/cpr-903/def/networkpolicy-apps-fqdn.yaml",
        type: "GRAPH",
        date: "2025-03-01T10:12:45Z",
        content: "graph TD;\n    A[Frontend] -->|request| B[API Server];\n    B -->|validate| C[Auth Service];\n    B -->|query| D[Database];\n    B -->|cache| E[Redis];\n    C -->|check| F[User DB];\n    style A fill:#f9f,stroke:#333,stroke-width:2px;\n    style B fill:#bbf,stroke:#333,stroke-width:2px;\n    style C fill:#bfb,stroke:#333,stroke-width:2px;",
        status: "active"
    },
    {
        id: 1065,
        title: "Network Policy - Connectivity Matrix",
        url: "https://console.example.com/rest/api/v1.0/projects/GKE/repo/gke-rs-network-policies/browse/k8s/kms/pr/app-namespaces/f/tmp/app-1065-csv/matrix.csv",
        type: "CSV",
        date: "2025-03-15T14:32:17Z",
        content: "Source,Destination,Port,Protocol,Allow\nfrontend,api,8080,TCP,true\nfrontend,db,5432,TCP,false\napi,db,5432,TCP,true\napi,cache,6379,TCP,true\nmonitoring,*,9090,TCP,true",
        status: "active"
    },
    {
        id: 1133,
        title: "Network Policy - Apps-FQDN",
        url: "https://console.example.com/rest/api/v1.0/projects/GKE/repo/gke-rs-network-policies/browse/k8s/kms/pr/cre-test/v1/app-1133-def/networkpolicy-apps-fqdn.yaml",
        type: "YAML",
        date: "2025-04-01T09:45:00Z",
        content: "apiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n  name: restrict-egress\n  namespace: frontend\nspec:\n  podSelector:\n    matchLabels:\n      app: web\n  policyTypes:\n  - Egress\n  egress:\n  - to:\n    - ipBlock:\n        cidr: 10.0.0.0/16\n    - ipBlock:\n        cidr: 172.16.0.0/12\n    ports:\n    - port: 443\n      protocol: TCP\n    - port: 53\n      protocol: UDP",
        status: "deprecated"
    },
    {
        id: 2418,
        title: "Network Policy - Base Default",
        url: "https://console.example.com/rest/api/v1.0/projects/GKE/repo/gke-rs-network-policies/browse/k8s/kms/pr/app-namespaces/ctrl/gke-bas/app-2418-def/default/",
        type: "YAML",
        date: "2025-04-15T11:22:00Z",
        content: "apiVersion: networking.k8s.io/v1\nkind: NetworkPolicy\nmetadata:\n  name: default-base-policy\n  namespace: default\nspec:\n  podSelector: {}\n  policyTypes:\n  - Ingress\n  - Egress\n  ingress: []\n  egress:\n  - to:\n    - ipBlock:\n        cidr: 0.0.0.0/0\n        except:\n        - 169.254.169.254/32",
        status: "active"
    }
];

// Application state
const state = {
    selectedPatterns: [],
    searchResults: [],
    isLoading: false,
    activeDetailView: null
};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

/**
 * Initialize the application and set up event listeners
 */
function initializeApp() {
    setupBackgroundAnimation();
    setupPatternBoxes();
    setupManualPatternInput();
    setupSearchButton();
    
    // Display initial placeholder in report section
    showReportPlaceholder();
}

/**
 * Set up event listeners for pattern selection boxes
 */
function setupPatternBoxes() {
    const patternBoxes = document.querySelectorAll('.pattern-box');
    
    patternBoxes.forEach(box => {
        box.addEventListener('click', () => {
            const pattern = box.getAttribute('data-pattern');
            
            // Toggle selection state
            if (box.classList.contains('selected')) {
                box.classList.remove('selected');
                removePattern(pattern);
            } else {
                box.classList.add('selected');
                addPattern(pattern);
            }
        });
    });
}

/**
 * Set up event listener for manual pattern input
 */
function setupManualPatternInput() {
    const addPatternButton = document.getElementById('add-pattern');
    const manualPatternInput = document.getElementById('manual-pattern');
    
    addPatternButton.addEventListener('click', () => {
        const pattern = manualPatternInput.value.trim();
        if (pattern) {
            addPattern(pattern);
            manualPatternInput.value = '';
        }
    });
    
    // Also allow pressing Enter to add pattern
    manualPatternInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const pattern = manualPatternInput.value.trim();
            if (pattern) {
                addPattern(pattern);
                manualPatternInput.value = '';
            }
        }
    });
}

/**
 * Set up event listener for search button
 */
function setupSearchButton() {
    const searchButton = document.getElementById('search-button');
    const searchInput = document.getElementById('search-input');
    
    searchButton.addEventListener('click', () => {
        const searchText = searchInput.value.trim();
        performSearch(searchText);
    });
    
    // Also allow pressing Enter to search
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const searchText = searchInput.value.trim();
            performSearch(searchText);
        }
    });
}

/**
 * Add pattern to the selected patterns list
 * @param {string} pattern - The pattern to add
 */
function addPattern(pattern) {
    // Check if pattern already exists
    if (state.selectedPatterns.includes(pattern)) {
        return;
    }
    
    state.selectedPatterns.push(pattern);
    updatePatternList();
}

/**
 * Remove pattern from the selected patterns list
 * @param {string} pattern - The pattern to remove
 */
function removePattern(pattern) {
    state.selectedPatterns = state.selectedPatterns.filter(p => p !== pattern);
    updatePatternList();
    
    // Unselect the corresponding pattern box if exists
    const patternBoxes = document.querySelectorAll('.pattern-box');
    patternBoxes.forEach(box => {
        if (box.getAttribute('data-pattern') === pattern) {
            box.classList.remove('selected');
        }
    });
}

/**
 * Update the display of the selected patterns list
 */
function updatePatternList() {
    const patternList = document.getElementById('pattern-list');
    patternList.innerHTML = '';
    
    if (state.selectedPatterns.length === 0) {
        const emptyItem = document.createElement('li');
        emptyItem.textContent = 'No patterns selected';
        emptyItem.style.fontStyle = 'italic';
        emptyItem.style.color = '#999';
        patternList.appendChild(emptyItem);
        return;
    }
    
    state.selectedPatterns.forEach(pattern => {
        const listItem = document.createElement('li');
        
        const patternText = document.createElement('span');
        patternText.textContent = pattern;
        patternText.className = 'pattern-text';
        
        const removeButton = document.createElement('button');
        removeButton.textContent = 'Remove';
        removeButton.className = 'remove-pattern';
        removeButton.addEventListener('click', () => {
            removePattern(pattern);
        });
        
        listItem.appendChild(patternText);
        listItem.appendChild(removeButton);
        patternList.appendChild(listItem);
    });
}

/**
 * Set up the animated background
 */
function setupBackgroundAnimation() {
    const animation = document.querySelector('.background-animation');
    
    // Add some random animated elements to the background
    for (let i = 0; i < 5; i++) {
        const line = document.createElement('div');
        line.className = 'animated-line';
        line.style.position = 'absolute';
        line.style.width = `${Math.random() * 200 + 100}px`;
        line.style.height = '1px';
        line.style.backgroundColor = 'rgba(58, 134, 255, 0.1)';
        line.style.top = `${Math.random() * 100}%`;
        line.style.left = `${Math.random() * 100}%`;
        line.style.transform = `rotate(${Math.random() * 360}deg)`;
        line.style.animation = `moveLine ${Math.random() * 20 + 10}s linear infinite`;
        animation.appendChild(line);
    }
    
    // Add keyframes style
    const style = document.createElement('style');
    style.textContent = `
        @keyframes moveLine {
            0% {
                transform: translate(0, 0) rotate(${Math.random() * 360}deg);
            }
            50% {
                transform: translate(${Math.random() * 200 - 100}px, ${Math.random() * 200 - 100}px) rotate(${Math.random() * 360}deg);
            }
            100% {
                transform: translate(0, 0) rotate(${Math.random() * 360}deg);
            }
        }
    `;
    document.head.appendChild(style);
}

/**
 * Display a placeholder message when no search results are available
 */
function showReportPlaceholder() {
    const reportContent = document.getElementById('report-content');
    reportContent.innerHTML = '';
    
    const placeholder = document.createElement('div');
    placeholder.className = 'report-placeholder';
    
    // SVG icon
    placeholder.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        </svg>
        <h3>No Results Yet</h3>
        <p>Select patterns and search to see results here</p>
    `;
    
    reportContent.appendChild(placeholder);
}








/**
 * search.js - Handles search functionality and filtering
 * 
 * This file contains functions related to searching and filtering network policies
 * based on user-defined patterns and search criteria.
 */

/**
 * Performs a search operation based on search text and selected patterns
 * @param {string} searchText - The text to search for
 */
function performSearch(searchText) {
    // If no patterns selected and no search text, show an alert
    if (state.selectedPatterns.length === 0 && !searchText) {
        alert('Please select at least one pattern or enter search text');
        return;
    }
    
    // Show loading indicator
    showLoadingIndicator();
    
    // Simulate network request delay (would be a real API call in production)
    setTimeout(() => {
        // Filter policies based on selected patterns and search text
        const results = filterPolicies(NETWORK_POLICIES, state.selectedPatterns, searchText);
        
        // Update state with search results
        state.searchResults = results;
        
        // Display the results
        displaySearchResults(results);
    }, 800); // Simulated delay for the "backend" to return results
}

/**
 * Filter policies based on selected patterns and search text
 * @param {Array} policies - The array of policy objects to filter
 * @param {Array} patterns - The array of selected pattern types
 * @param {string} searchText - The search text to filter by
 * @returns {Array} - Filtered array of policies
 */
function filterPolicies(policies, patterns, searchText) {
    return policies.filter(policy => {
        // Filter by pattern type if patterns array is not empty
        const matchesPattern = patterns.length === 0 || patterns.includes(policy.type);
        
        // Filter by search text if provided
        let matchesSearchText = true;
        if (searchText) {
            const searchLower = searchText.toLowerCase();
            matchesSearchText = policy.title.toLowerCase().includes(searchLower) || 
                               policy.content.toLowerCase().includes(searchLower) ||
                               policy.url.toLowerCase().includes(searchLower);
        }
        
        return matchesPattern && matchesSearchText;
    });
}

/**
 * Show loading indicator in the report content area
 */
function showLoadingIndicator() {
    const reportContent = document.getElementById('report-content');
    reportContent.innerHTML = '';
    
    const loadingIndicator = document.createElement('div');
    loadingIndicator.className = 'loading-indicator';
    loadingIndicator.innerHTML = '<div class="spinner"></div>';
    
    reportContent.appendChild(loadingIndicator);
}

/**
 * Format date string to a more readable format
 * @param {string} dateString - ISO date string
 * @returns {string} - Formatted date string
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

/**
 * Get status badge HTML based on policy status
 * @param {string} status - The policy status
 * @returns {string} - HTML for the status badge
 */
function getStatusBadge(status) {
    const statusMap = {
        'active': 'status-active',
        'deprecated': 'status-deprecated',
        'draft': 'status-draft'
    };
    
    const className = statusMap[status] || '';
    return `<span class="status-badge ${className}">${status}</span>`;
}

/**
 * Truncate a URL to a displayable length
 * @param {string} url - The full URL
 * @param {number} maxLength - Maximum displayed length
 * @returns {string} - Truncated URL
 */
function truncateUrl(url, maxLength = 60) {
    if (url.length <= maxLength) {
        return url;
    }
    
    // Extract domain from URL
    const urlObj = new URL(url);
    const domain = urlObj.hostname;
    
    // Calculate remaining characters
    const remaining = maxLength - domain.length - 10; // 10 for "https://..." and "/..."
    
    // Get the path and truncate it if necessary
    let path = urlObj.pathname;
    if (path.length > remaining) {
        path = path.substring(0, Math.floor(remaining / 2)) + 
               '...' + 
               path.substring(path.length - Math.floor(remaining / 2));
    }
    
    return `https://${domain}${path}`;
}

/**
 * Display search results in the report content area
 * @param {Array} results - Array of policy objects to display
 */
function displaySearchResults(results) {
    const reportContent = document.getElementById('report-content');
    reportContent.innerHTML = '';
    
    if (results.length === 0) {
        // No results found
        const noResults = document.createElement('div');
        noResults.className = 'report-placeholder';
        noResults.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="8" y1="12" x2="16" y2="12"></line>
            </svg>
            <h3>No Results Found</h3>
            <p>Try changing your search criteria or selecting different patterns</p>
        `;
        reportContent.appendChild(noResults);
        return;
    }
    
    // Create result items
    results.forEach((policy, index) => {
        const policyItem = document.createElement('div');
        policyItem.className = 'policy-item';
        policyItem.setAttribute('data-policy-id', policy.id);
        
        policyItem.innerHTML = `
            <div class="policy-number">${index + 1}</div>
            <div class="policy-details">
                <div class="policy-title">${policy.title} ${getStatusBadge(policy.status)}</div>
                <div class="policy-url">${truncateUrl(policy.url)}</div>
            </div>
            <div class="policy-date">${formatDate(policy.date)}</div>
        `;
        
        // Add click event to show details
        policyItem.addEventListener('click', () => {
            showPolicyDetails(policy);
        });
        
        reportContent.appendChild(policyItem);
    });
    
    // Create detail view container (initially hidden)
    const detailView = document.createElement('div');
    detailView.className = 'policy-detail-view';
    detailView.id = 'policy-detail-view';
    reportContent.appendChild(detailView);
}

/**
 * Show detailed view of a selected policy
 * @param {Object} policy - The policy object to display
 */
function showPolicyDetails(policy) {
    const detailView = document.getElementById('policy-detail-view');
    detailView.innerHTML = '';
    detailView.classList.add('active');
    
    // Update active detail view in state
    state.activeDetailView = policy.id;
    
    // Highlight selected item
    const policyItems = document.querySelectorAll('.policy-item');
    policyItems.forEach(item => {
        if (parseInt(item.getAttribute('data-policy-id')) === policy.id) {
            item.style.backgroundColor = '#f0f4ff';
        } else {
            item.style.backgroundColor = '';
        }
    });
    
    // Create detail header
    const detailHeader = document.createElement('div');
    detailHeader.className = 'detail-header';
    
    const detailTitle = document.createElement('div');
    detailTitle.className = 'detail-title';
    detailTitle.textContent = policy.title;
    
    const closeButton = document.createElement('button');
    closeButton.className = 'detail-close';
    closeButton.innerHTML = '&times;';
    closeButton.addEventListener('click', () => {
        detailView.classList.remove('active');
        state.activeDetailView = null;
        
        // Remove highlighting
        policyItems.forEach(item => {
            item.style.backgroundColor = '';
        });
    });
    
    detailHeader.appendChild(detailTitle);
    detailHeader.appendChild(closeButton);
    
    // Create content container
    const detailContent = document.createElement('div');
    detailContent.className = 'detail-content';
    
    // Format based on type
    let formattedContent;
    switch (policy.type) {
        case 'YAML':
        case 'JSON':
            formattedContent = `<pre>${policy.content}</pre>`;
            break;
        case 'CSV':
            formattedContent = formatCSV(policy.content);
            break;
        case 'GRAPH':
            formattedContent = `<div class="graph-content">${policy.content}</div>`;
            break;
        default:
            formattedContent = `<pre>${policy.content}</pre>`;
    }
    
    detailContent.innerHTML = formattedContent;
    
    // Create action buttons
    const detailActions = document.createElement('div');
    detailActions.className = 'detail-actions';
    
    const viewButton = document.createElement('button');
    viewButton.textContent = 'View Full Policy';
    viewButton.addEventListener('click', () => {
        // In a real app, this would open the policy URL or navigate to a detailed view
        window.open(policy.url, '_blank');
    });
    
    detailActions.appendChild(viewButton);
    
    // Add all elements to detail view
    detailView.appendChild(detailHeader);
    detailView.appendChild(detailContent);
    detailView.appendChild(detailActions);
    
    // Scroll to detail view
    detailView.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Format CSV content as an HTML table
 * @param {string} csvContent - Raw CSV content
 * @returns {string} - HTML table representation
 */
function formatCSV(csvContent) {
    const lines = csvContent.trim().split('\n');
    const headers = lines[0].split(',');
    
    let tableHtml = '<table border="1" cellspacing="0" cellpadding="5" style="width:100%; border-collapse: collapse;">';
    
    // Add header row
    tableHtml += '<tr>';
    headers.forEach(header => {
        tableHtml += `<th style="background-color: #f0f4ff;">${header}</th>`;
    });
    tableHtml += '</tr>';
    
    // Add data rows
    for (let i = 1; i < lines.length; i++) {
        const cells = lines[i].split(',');
        tableHtml += '<tr>';
        cells.forEach(cell => {
            tableHtml += `<td>${cell}</td>`;
        });
        tableHtml += '</tr>';
    }
    
    tableHtml += '</table>';
    return tableHtml;
}







/**
 * report.js - Handles report generation and display
 * 
 * This file contains functions related to creating, formatting, and
 * displaying reports based on search results.
 */

/**
 * Generate a report based on the current search results
 * This would typically involve more complex operations in a real app,
 * potentially involving backend processing
 */
function generateReport() {
    if (state.searchResults.length === 0) {
        alert('No search results to generate a report from. Please perform a search first.');
        return;
    }
    
    // In a real application, this could trigger a backend process to generate a PDF or other report format
    // For this demo, we'll create a simple summary report modal
    
    // Create a modal dialog for the report
    const modal = document.createElement('div');
    modal.style.position = 'fixed';
    modal.style.top = '0';
    modal.style.left = '0';
    modal.style.width = '100%';
    modal.style.height = '100%';
    modal.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    modal.style.display = 'flex';
    modal.style.justifyContent = 'center';
    modal.style.alignItems = 'center';
    modal.style.zIndex = '1000';
    
    // Create the report content
    const reportCard = document.createElement('div');
    reportCard.style.backgroundColor = 'white';
    reportCard.style.padding = '2rem';
    reportCard.style.borderRadius = '8px';
    reportCard.style.maxWidth = '800px';
    reportCard.style.width = '90%';
    reportCard.style.maxHeight = '80vh';
    reportCard.style.overflow = 'auto';
    reportCard.style.position = 'relative';
    
    // Close button
    const closeButton = document.createElement('button');
    closeButton.style.position = 'absolute';
    closeButton.style.top = '1rem';
    closeButton.style.right = '1rem';
    closeButton.style.backgroundColor = 'transparent';
    closeButton.style.border = 'none';
    closeButton.style.fontSize = '1.5rem';
    closeButton.style.cursor = 'pointer';
    closeButton.innerHTML = '&times;';
    closeButton.addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    // Report title
    const title = document.createElement('h2');
    title.style.marginBottom = '1.5rem';
    title.style.borderBottom = '2px solid #f0f4ff';
    title.style.paddingBottom = '0.5rem';
    title.textContent = 'Network Policies Report';
    
    // Report summary
    const summary = document.createElement('div');
    summary.style.marginBottom = '1.5rem';
    summary.style.padding = '1rem';
    summary.style.backgroundColor = '#f8f9fa';
    summary.style.borderRadius = '4px';
    
    const patterns = state.selectedPatterns.length > 0 
        ? state.selectedPatterns.join(', ') 
        : 'All';
    
    summary.innerHTML = `
        <p><strong>Report Generated:</strong> ${new Date().toLocaleString()}</p>
        <p><strong>Patterns:</strong> ${patterns}</p>
        <p><strong>Results Found:</strong> ${state.searchResults.length}</p>
    `;
    
    // Results table
    const resultsTable = document.createElement('table');
    resultsTable.style.width = '100%';
    resultsTable.style.borderCollapse = 'collapse';
    resultsTable.style.marginBottom = '1rem';
    
    // Table header
    const tableHeader = document.createElement('thead');
    tableHeader.innerHTML = `
        <tr>
            <th style="text-align: left; padding: 0.75rem; border-bottom: 2px solid #dee2e6;">ID</th>
            <th style="text-align: left; padding: 0.75rem; border-bottom: 2px solid #dee2e6;">Title</th>
            <th style="text-align: left; padding: 0.75rem; border-bottom: 2px solid #dee2e6;">Type</th>
            <th style="text-align: left; padding: 0.75rem; border-bottom: 2px solid #dee2e6;">Date</th>
            <th style="text-align: left; padding: 0.75rem; border-bottom: 2px solid #dee2e6;">Status</th>
        </tr>
    `;
    
    // Table body
    const tableBody = document.createElement('tbody');
    state.searchResults.forEach(policy => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td style="padding: 0.75rem; border-bottom: 1px solid #dee2e6;">${policy.id}</td>
            <td style="padding: 0.75rem; border-bottom: 1px solid #dee2e6;">${policy.title}</td>
            <td style="padding: 0.75rem; border-bottom: 1px solid #dee2e6;">${policy.type}</td>
            <td style="padding: 0.75rem; border-bottom: 1px solid #dee2e6;">${formatDate(policy.date)}</td>
            <td style="padding: 0.75rem; border-bottom: 1px solid #dee2e6;">
                <span style="${getStatusStyle(policy.status)}">${policy.status}</span>
            </td>
        `;
        tableBody.appendChild(row);
    });
    
    // Assemble the table
    resultsTable.appendChild(tableHeader);
    resultsTable.appendChild(tableBody);
    
    // Export buttons
    const exportButtons = document.createElement('div');
    exportButtons.style.marginTop = '2rem';
    exportButtons.style.display = 'flex';
    exportButtons.style.justifyContent = 'flex-end';
    exportButtons.style.gap = '0.5rem';
    
    const exportPDF = document.createElement('button');
    exportPDF.style.backgroundColor = '#dc3545';
    exportPDF.style.color = 'white';
    exportPDF.style.border = 'none';
    exportPDF.style.padding = '0.5rem 1rem';
    exportPDF.style.borderRadius = '4px';
    exportPDF.style.cursor = 'pointer';
    exportPDF.textContent = 'Export as PDF';
    exportPDF.addEventListener('click', () => {
        alert('PDF export functionality would be implemented in a real application.');
    });
    
    const exportCSV = document.createElement('button');
    exportCSV.style.backgroundColor = '#28a745';
    exportCSV.style.color = 'white';
    exportCSV.style.border = 'none';
    exportCSV.style.padding = '0.5rem 1rem';
    exportCSV.style.borderRadius = '4px';
    exportCSV.style.cursor = 'pointer';
    exportCSV.textContent = 'Export as CSV';
    exportCSV.addEventListener('click', () => {
        alert('CSV export functionality would be implemented in a real application.');
    });
    
    exportButtons.appendChild(exportCSV);
    exportButtons.appendChild(exportPDF);
    
    // Assemble the report
    reportCard.appendChild(closeButton);
    reportCard.appendChild(title);
    reportCard.appendChild(summary);
    reportCard.appendChild(resultsTable);
    reportCard.appendChild(exportButtons);
    
    modal.appendChild(reportCard);
    document.body.appendChild(modal);
}

/**
 * Get CSS style string for status indicator
 * @param {string} status - The policy status
 * @returns {string} - CSS style string
 */
function getStatusStyle(status) {
    switch (status) {
        case 'active':
            return 'background-color: rgba(40, 167, 69, 0.1); color: #28a745; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.875rem;';
        case 'deprecated':
            return 'background-color: rgba(220, 53, 69, 0.1); color: #dc3545; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.875rem;';
        case 'draft':
            return 'background-color: rgba(255, 193, 7, 0.1); color: #ffc107; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.875rem;';
        default:
            return 'background-color: rgba(108, 117, 125, 0.1); color: #6c757d; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.875rem;';
    }
}

/**
 * Add a "Generate Report" button to the UI
 * This function would be called after initial DOM load
 */
function addReportButton() {
    const reportContainer = document.querySelector('.report-container');
    const reportInfo = document.querySelector('.report-info');
    
    const generateReportButton = document.createElement('button');
    generateReportButton.textContent = 'Generate Report';
    generateReportButton.style.marginLeft = '1rem';
    generateReportButton.addEventListener('click', generateReport);
    
    reportInfo.appendChild(generateReportButton);
}

// Add the report button after the document is loaded
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(addReportButton, 100); // Small delay to ensure the DOM elements exist
});







<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600">
  <!-- Background grid pattern -->
  <defs>
    <pattern id="smallGrid" width="20" height="20" patternUnits="userSpaceOnUse">
      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(58, 134, 255, 0.05)" stroke-width="0.5"/>
    </pattern>
    <pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse">
      <rect width="100" height="100" fill="url(#smallGrid)"/>
      <path d="M 100 0 L 0 0 0 100" fill="none" stroke="rgba(58, 134, 255, 0.1)" stroke-width="1"/>
    </pattern>
  </defs>
  
  <!-- Background rectangle with grid pattern -->
  <rect width="100%" height="100%" fill="url(#grid)" />
  
  <!-- Animated elements -->
  <g class="animated-elements">
    <!-- Decorative circles -->
    <circle cx="150" cy="150" r="50" fill="rgba(58, 134, 255, 0.05)">
      <animate attributeName="r" values="50;70;50" dur="10s" repeatCount="indefinite" />
    </circle>
    
    <circle cx="650" cy="450" r="80" fill="rgba(131, 56, 236, 0.05)">
      <animate attributeName="r" values="80;100;80" dur="15s" repeatCount="indefinite" />
    </circle>
    
    <!-- Decorative lines -->
    <line x1="100" y1="100" x2="300" y2="300" stroke="rgba(58, 134, 255, 0.1)" stroke-width="1">
      <animate attributeName="x2" values="300;350;300" dur="8s" repeatCount="indefinite" />
      <animate attributeName="y2" values="300;250;300" dur="8s" repeatCount="indefinite" />
    </line>
    
    <line x1="700" y1="200" x2="500" y2="400" stroke="rgba(131, 56, 236, 0.1)" stroke-width="1">
      <animate attributeName="x2" values="500;450;500" dur="12s" repeatCount="indefinite" />
      <animate attributeName="y2" values="400;450;400" dur="12s" repeatCount="indefinite" />
    </line>
    
    <!-- Network connections visualization -->
    <g stroke="rgba(58, 134, 255, 0.15)" stroke-width="1">
      <line x1="200" y1="300" x2="400" y2="200">
        <animate attributeName="y1" values="300;310;300" dur="5s" repeatCount="indefinite" />
      </line>
      <line x1="400" y1="200" x2="600" y2="300">
        <animate attributeName="y2" values="300;290;300" dur="8s" repeatCount="indefinite" />
      </line>
      <line x1="400" y1="200" x2="400" y2="400">
        <animate attributeName="y2" values="400;420;400" dur="6s" repeatCount="indefinite" />
      </line>
      
      <circle cx="200" cy="300" r="8" fill="rgba(58, 134, 255, 0.2)">
        <animate attributeName="cy" values="300;310;300" dur="5s" repeatCount="indefinite" />
      </circle>
      <circle cx="400" cy="200" r="10" fill="rgba(58, 134, 255, 0.3)">
        <animate attributeName="r" values="10;12;10" dur="4s" repeatCount="indefinite" />
      </circle>
      <circle cx="600" cy="300" r="8" fill="rgba(58, 134, 255, 0.2)">
        <animate attributeName="cy" values="300;290;300" dur="8s" repeatCount="indefinite" />
      </circle>
      <circle cx="400" cy="400" r="8" fill="rgba(58, 134, 255, 0.2)">
        <animate attributeName="cy" values="400;420;400" dur="6s" repeatCount="indefinite" />
      </circle>
    </g>
  </g>
  
  <!-- Subtle pulsing effect overlay -->
  <rect width="100%" height="100%" fill="url(#pulse-gradient)">
    <animate attributeName="opacity" values="0.5;0.7;0.5" dur="10s" repeatCount="indefinite" />
  </rect>
  
  <defs>
    <linearGradient id="pulse-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="rgba(58, 134, 255, 0)" />
      <stop offset="50%" stop-color="rgba(131, 56, 236, 0.03)" />
      <stop offset="100%" stop-color="rgba(58, 134, 255, 0)" />
      <animate attributeName="x1" values="0%;20%;0%" dur="20s" repeatCount="indefinite" />
      <animate attributeName="y1" values="0%;20%;0%" dur="20s" repeatCount="indefinite" />
      <animate attributeName="x2" values="100%;80%;100%" dur="20s" repeatCount="indefinite" />
      <animate attributeName="y2" values="100%;80%;100%" dur="20s" repeatCount="indefinite" />
    </linearGradient>
  </defs>
</svg>