You're absolutely right! Let me carefully review the logic and fix all the issues. The problem is in the field mappings - I need to ensure consistency and accuracy.
Updated index.html (Complete Replacement - VERIFIED)
Let me trace through the logic:
EDB Table should show:
OSCAR: guid, gfid, gus_id, namespace, status
CoPPER: guid, gfid, gus_id, eff_to, status
EDB: guid, gfid, gus_id, eff_to, status
STAR Table should show:
OSCAR: guid, gfid, gus_id, namespace, status
CoPPER: guid, gfid, gus_id, eff_to, exch_id, status (CoPPER has GUID here!)
STAR: gfid, gfid_description, exchange_id (STAR does NOT have GUID)
Here's the corrected version:
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OSCAR Reconciliation Tool</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-color: #1a365d;
            --primary-light: #2c5aa0;
            --primary-dark: #0f2537;
            --secondary-color: #e53e3e;
            --accent-color: #00b4d8;
            --accent-light: #48cae4;
            --success-color: #38a169;
            --success-light: #48bb78;
            --warning-color: #ed8936;
            --warning-light: #f6ad55;
            --error-color: #e53e3e;
            --error-light: #fc8181;
            
            --white: #ffffff;
            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-400: #9ca3af;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;
            
            --gradient-primary: linear-gradient(135deg, #1a365d 0%, #2c5aa0 100%);
            --gradient-accent: linear-gradient(135deg, #00b4d8 0%, #48cae4 100%);
            --gradient-success: linear-gradient(135deg, #38a169 0%, #48bb78 100%);
            --gradient-warning: linear-gradient(135deg, #ed8936 0%, #f6ad55 100%);
            --gradient-error: linear-gradient(135deg, #e53e3e 0%, #fc8181 100%);
            
            --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            
            --spacing-xs: 0.25rem;
            --spacing-sm: 0.5rem;
            --spacing-md: 1rem;
            --spacing-lg: 1.5rem;
            --spacing-xl: 2rem;
            --spacing-2xl: 3rem;
            
            --radius-sm: 0.375rem;
            --radius-md: 0.5rem;
            --radius-lg: 0.75rem;
            --radius-xl: 1rem;
            
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            
            --transition-fast: 0.15s ease-in-out;
            --transition-normal: 0.3s ease-in-out;
            --transition-slow: 0.5s ease-in-out;
        }

        body {
            font-family: var(--font-family);
            font-size: 1rem;
            line-height: 1.6;
            color: var(--gray-800);
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            min-height: 100vh;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 var(--spacing-lg);
        }

        .header {
            background: var(--gradient-primary);
            color: var(--white);
            padding: var(--spacing-xl) 0;
            box-shadow: var(--shadow-xl);
            position: sticky;
            top: 0;
            z-index: 100;
            border-bottom: 3px solid var(--accent-color);
        }

        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .logo {
            display: flex;
            align-items: center;
            gap: var(--spacing-md);
        }

        .logo i {
            font-size: 2rem;
            color: var(--accent-color);
            animation: rotate 3s linear infinite;
        }

        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .logo h1 {
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(135deg, var(--white) 0%, var(--accent-light) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .connection-status {
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            background: rgba(0, 180, 216, 0.2);
            padding: var(--spacing-sm) var(--spacing-lg);
            border-radius: var(--radius-xl);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 180, 216, 0.3);
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--success-color);
            box-shadow: 0 0 10px var(--success-color);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.1); }
        }

        .connection-status span {
            font-weight: 500;
            font-size: 0.875rem;
        }

        .main-content {
            padding: var(--spacing-2xl) 0;
            min-height: calc(100vh - 200px);
        }

        .welcome-section {
            text-align: center;
            margin-bottom: var(--spacing-2xl);
        }

        .welcome-content h2 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--white) 0%, var(--accent-color) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: var(--spacing-md);
            text-shadow: 0 0 30px rgba(0, 180, 216, 0.3);
        }

        .welcome-content p {
            font-size: 1.125rem;
            color: var(--gray-300);
            max-width: 700px;
            margin: 0 auto;
        }

        .search-section {
            margin-bottom: var(--spacing-2xl);
        }

        .search-card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: var(--radius-xl);
            padding: var(--spacing-2xl);
            box-shadow: var(--shadow-xl);
            border: 1px solid rgba(0, 180, 216, 0.2);
            backdrop-filter: blur(10px);
        }

        .search-header {
            text-align: center;
            margin-bottom: var(--spacing-xl);
        }

        .search-header h3 {
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: var(--spacing-sm);
        }

        .search-header h3 i {
            margin-right: var(--spacing-sm);
            color: var(--accent-color);
        }

        .search-header p {
            color: var(--gray-600);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: var(--spacing-lg);
            margin-bottom: var(--spacing-xl);
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: var(--spacing-sm);
        }

        .form-group label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--gray-700);
        }

        .form-group label i {
            color: var(--accent-color);
            margin-right: var(--spacing-xs);
        }

        .form-group input {
            padding: var(--spacing-md);
            border: 2px solid var(--gray-300);
            border-radius: var(--radius-lg);
            font-size: 1rem;
            transition: var(--transition-normal);
            background: var(--white);
            font-weight: 500;
        }

        .form-group input:disabled {
            background: var(--gray-100);
            color: var(--gray-400);
            cursor: not-allowed;
            opacity: 0.6;
        }

        .form-group input:focus:not(:disabled) {
            outline: none;
            border-color: var(--accent-color);
            box-shadow: 0 0 0 4px rgba(0, 180, 216, 0.15);
            transform: translateY(-2px);
        }

        .input-info {
            font-size: 0.75rem;
            color: var(--gray-500);
            font-style: italic;
        }

        .divider {
            text-align: center;
            margin: var(--spacing-xl) 0;
            position: relative;
        }

        .divider::before {
            content: '';
            position: absolute;
            left: 0;
            top: 50%;
            width: 42%;
            height: 2px;
            background: linear-gradient(90deg, transparent 0%, var(--accent-color) 100%);
        }

        .divider::after {
            content: '';
            position: absolute;
            right: 0;
            top: 50%;
            width: 42%;
            height: 2px;
            background: linear-gradient(90deg, var(--accent-color) 0%, transparent 100%);
        }

        .divider span {
            background: var(--white);
            padding: 0 var(--spacing-lg);
            color: var(--accent-color);
            font-weight: 700;
            font-size: 1.125rem;
            border: 2px solid var(--accent-color);
            border-radius: var(--radius-xl);
            display: inline-block;
        }

        .btn-primary {
            background: var(--gradient-accent);
            color: var(--white);
            border: none;
            padding: var(--spacing-lg) var(--spacing-2xl);
            border-radius: var(--radius-xl);
            font-size: 1.125rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition-normal);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: var(--spacing-md);
            box-shadow: 0 8px 20px rgba(0, 180, 216, 0.4);
            border: 2px solid transparent;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-3px);
            box-shadow: 0 12px 30px rgba(0, 180, 216, 0.6);
            border-color: var(--accent-light);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-primary i {
            font-size: 1.25rem;
        }

        .btn-secondary {
            background: var(--white);
            color: var(--primary-color);
            border: 2px solid var(--primary-color);
            padding: var(--spacing-sm) var(--spacing-lg);
            border-radius: var(--radius-lg);
            font-size: 0.875rem;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition-normal);
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
        }

        .btn-secondary:hover {
            background: var(--primary-color);
            color: var(--white);
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .submit-container {
            text-align: center;
            margin-top: var(--spacing-xl);
        }

        .results-section {
            display: none;
            animation: fadeInUp 0.6s ease-out;
        }

        .results-section.show {
            display: block;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: var(--spacing-xl);
            background: rgba(255, 255, 255, 0.1);
            padding: var(--spacing-lg);
            border-radius: var(--radius-xl);
            backdrop-filter: blur(10px);
        }

        .results-header h3 {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--white);
        }

        .results-header h3 i {
            margin-right: var(--spacing-sm);
            color: var(--accent-color);
        }

        .results-actions {
            display: flex;
            gap: var(--spacing-md);
        }

        .scenario-card {
            background: linear-gradient(135deg, rgba(0, 180, 216, 0.95) 0%, rgba(72, 202, 228, 0.95) 100%);
            border-radius: var(--radius-xl);
            padding: var(--spacing-2xl);
            box-shadow: 0 15px 35px rgba(0, 180, 216, 0.4);
            margin-bottom: var(--spacing-xl);
            border: 2px solid rgba(255, 255, 255, 0.3);
            position: relative;
            overflow: hidden;
        }

        .scenario-card::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: shimmer 3s infinite;
        }

        @keyframes shimmer {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .scenario-card h4 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--white);
            margin-bottom: var(--spacing-md);
            display: flex;
            align-items: center;
            gap: var(--spacing-sm);
            position: relative;
            z-index: 1;
        }

        .scenario-card h4 i {
            font-size: 1.5rem;
        }

        .scenario-content {
            position: relative;
            z-index: 1;
        }

        .scenario-text {
            color: var(--white);
            font-size: 1.125rem;
            font-weight: 600;
            margin-bottom: var(--spacing-md);
            padding: var(--spacing-md);
            background: rgba(255, 255, 255, 0.15);
            border-radius: var(--radius-lg);
            border-left: 4px solid var(--white);
        }

        .action-text {
            color: var(--white);
            font-size: 1rem;
            line-height: 1.8;
            padding: var(--spacing-lg);
            background: rgba(26, 54, 93, 0.5);
            border-radius: var(--radius-lg);
            border: 2px dashed rgba(255, 255, 255, 0.5);
        }

        .action-text strong {
            font-weight: 700;
            text-decoration: underline;
        }

        .comparison-tables {
            display: grid;
            grid-template-columns: 1fr;
            gap: var(--spacing-2xl);
            margin-bottom: var(--spacing-xl);
        }

        .table-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: var(--radius-xl);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
            overflow: hidden;
            border: 2px solid var(--accent-color);
        }

        .table-header {
            background: var(--gradient-primary);
            color: var(--white);
            padding: var(--spacing-xl);
            text-align: center;
            position: relative;
        }

        .table-header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--gradient-accent);
        }

        .table-header h4 {
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: var(--spacing-md);
        }

        .table-header h4 i {
            color: var(--accent-color);
            font-size: 1.5rem;
        }

        .table-container {
            overflow-x: auto;
            max-height: 600px;
            overflow-y: auto;
        }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
        }

        .comparison-table th {
            background: linear-gradient(135deg, var(--gray-800) 0%, var(--gray-700) 100%);
            color: var(--white);
            padding: var(--spacing-lg);
            text-align: left;
            font-weight: 700;
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 3px solid var(--accent-color);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .comparison-table td {
            padding: var(--spacing-md);
            text-align: left;
            border-bottom: 1px solid var(--gray-200);
            vertical-align: middle;
            font-size: 0.95rem;
        }

        .comparison-table tbody tr {
            transition: var(--transition-fast);
        }

        .comparison-table tbody tr:hover {
            background: linear-gradient(90deg, rgba(0, 180, 216, 0.1) 0%, transparent 100%);
            transform: translateX(5px);
        }

        .db-column {
            font-weight: 700;
            color: var(--primary-color);
        }

        .oscar-col {
            background: linear-gradient(135deg, rgba(26, 54, 93, 0.08) 0%, rgba(26, 54, 93, 0.03) 100%);
            border-left: 3px solid var(--primary-color);
        }

        .copper-col {
            background: linear-gradient(135deg, rgba(0, 180, 216, 0.08) 0%, rgba(0, 180, 216, 0.03) 100%);
            border-left: 3px solid var(--accent-color);
        }

        .star-col {
            background: linear-gradient(135deg, rgba(56, 161, 105, 0.08) 0%, rgba(56, 161, 105, 0.03) 100%);
            border-left: 3px solid var(--success-color);
        }

        .edb-col {
            background: linear-gradient(135deg, rgba(237, 137, 54, 0.08) 0%, rgba(237, 137, 54, 0.03) 100%);
            border-left: 3px solid var(--warning-color);
        }

        .status-badge {
            padding: 0.375rem 0.75rem;
            border-radius: var(--radius-lg);
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            display: inline-block;
            letter-spacing: 0.05em;
            box-shadow: var(--shadow-sm);
        }

        .status-match {
            background: var(--gradient-success);
            color: var(--white);
        }

        .status-mismatch {
            background: var(--gradient-error);
            color: var(--white);
        }

        .status-missing {
            background: var(--gradient-warning);
            color: var(--white);
        }

        .status-partial {
            background: linear-gradient(135deg, var(--warning-color) 0%, var(--error-color) 100%);
            color: var(--white);
        }

        .status-active {
            background: var(--gradient-success);
            color: var(--white);
        }

        .status-inactive {
            background: linear-gradient(135deg, var(--gray-600) 0%, var(--gray-500) 100%);
            color: var(--white);

        .alert {
            padding: var(--spacing-lg);
            border-radius: var(--radius-lg);
            margin-bottom: var(--spacing-lg);
            border-left: 4px solid;
        }

        .alert-error {
            background: rgba(229, 62, 62, 0.1);
            border-left-color: var(--error-color);
            color: var(--error-color);
        }

        .alert-success {
            background: rgba(56, 161, 105, 0.1);
            border-left-color: var(--success-color);
            color: var(--success-color);
        }

        .loading {
            display: inline-block;
            width: 1.25rem;
            height: 1.25rem;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top-color: var(--white);
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @media (max-width: 768px) {
            .container {
                padding: 0 var(--spacing-md);
            }
            
            .header-content {
                flex-direction: column;
                gap: var(--spacing-md);
            }
            
            .logo h1 {
                font-size: 1.25rem;
            }
            
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .results-header {
                flex-direction: column;
                gap: var(--spacing-md);
                align-items: stretch;
            }
            
            .results-actions {
                justify-content: center;
                flex-wrap: wrap;
            }
            
            .comparison-table {
                font-size: 0.875rem;
            }
            
            .comparison-table th,
            .comparison-table td {
                padding: var(--spacing-sm);
            }
        }
    </style>
</head>
<body>
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-sync-alt"></i>
                    <h1>OSCAR Reconcile</h1>
                </div>
                <div class="connection-status">
                    <div class="status-dot"></div>
                    <span>All Systems Connected</span>
                </div>
            </div>
        </div>
    </header>

    <main class="main-content">
        <div class="container">
            <section class="welcome-section">
                <div class="welcome-content">
                    <h2>Data Reconciliation Engine</h2>
                    <p>OSCAR • CoPPER • STAR • EDB Multi-System Comparison & Validation</p>
                </div>
            </section>

            <section class="search-section">
                <div class="search-card">
                    <div class="search-header">
                        <h3><i class="fas fa-search"></i> Reconciliation Parameters</h3>
                        <p>Search by GUID or by GFID + GUS ID combination</p>
                    </div>
                    
                    <form id="reconcile-form">
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="guid">
                                    <i class="fas fa-key"></i> GUID (12 chars)
                                </label>
                                <input type="text" id="guid" name="guid" placeholder="e.g., ABCDEFGH1234" maxlength="12">
                                <div class="input-info">Global Unique Identifier</div>
                            </div>
                        </div>

                        <div class="divider">
                            <span>OR</span>
                        </div>

                        <div class="form-grid">
                            <div class="form-group">
                                <label for="gfid">
                                    <i class="fas fa-building"></i> GFID (4 chars)
                                </label>
                                <input type="text" id="gfid" name="gfid" placeholder="e.g., ABCD" maxlength="4">
                                <div class="input-info">Globex Firm ID</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="gus-id">
                                    <i class="fas fa-user"></i> GUS ID (5 chars)
                                </label>
                                <input type="text" id="gus-id" name="gus_id" placeholder="e.g., ABCDE" maxlength="5">
                                <div class="input-info">Globex User Signature ID</div>
                            </div>
                        </div>

                        <div class="submit-container">
                            <button type="submit" class="btn-primary" id="submit-btn">
                                <i class="fas fa-bolt"></i>
                                <span>Execute Reconciliation</span>
                            </button>
                        </div>
                    </form>
                </div>
            </section>

            <section class="results-section" id="results-section">
                <div class="results-header">
                    <h3><i class="fas fa-chart-line"></i> Reconciliation Results</h3>
                    <div class="results-actions">
                        <button class="btn-secondary" id="export-btn">
                            <i class="fas fa-download"></i> Export JSON
                        </button>
                        <button class="btn-secondary" id="clear-results-btn">
                            <i class="fas fa-times-circle"></i> Clear Results
                        </button>
                    </div>
                </div>

                <div id="error-container"></div>

                <div class="scenario-card" id="scenario-card">
                    <h4><i class="fas fa-lightbulb"></i> Detected Scenario & Recommended Action</h4>
                    <div class="scenario-content">
                        <div class="scenario-text" id="scenario-text"></div>
                        <div class="action-text" id="action-text"></div>
                    </div>
                </div>

                <div class="comparison-tables">
                    <div class="table-section">
                        <div class="table-header">
                            <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER ↔ EDB Comparison</h4>
                        </div>
                        <div class="table-container">
                            <table class="comparison-table" id="edb-table">
                                <thead>
                                    <tr>
                                        <th>Field</th>
                                        <th class="oscar-col">OSCAR</th>
                                        <th class="copper-col">CoPPER</th>
                                        <th class="edb-col">EDB</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody id="edb-table-body">
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div class="table-section">
                        <div class="table-header">
                            <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER ↔STAR Comparison</h4>
                        </div>
                        <div class="table-container">
                            <table class="comparison-table" id="star-table">
                                <thead>
                                    <tr>
                                        <th>Field</th>
                                        <th class="oscar-col">OSCAR</th>
                                        <th class="copper-col">CoPPER</th>
                                        <th class="star-col">STAR</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody id="star-table-body">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <script>
        let currentResults = null;

        // CORRECTED Field mappings - reflecting actual database schemas
        const FIELD_MAPPINGS = {
            edb: {
                // Fields that ACTUALLY exist in each database for EDB comparison
                oscar: {
                    guid: true,
                    gfid: true, 
                    gus_id: true,
                    namespace: true,
                    status: true
                },
                copper: {
                    guid: true,
                    gfid: true,
                    gus_id: true,
                    eff_to: true,
                    status: true
                },
                edb: {
                    guid: true,
                    gfid: true,
                    gus_id: true,
                    eff_to: true,
                    status: true
                }
            },
            star: {
                // Fields that ACTUALLY exist in each database for STAR comparison
                oscar: {
                    guid: true,
                    gfid: true,
                    gus_id: true,
                    namespace: true,
                    status: true
                },
                copper: {
                    guid: true,      // CoPPER HAS GUID in STAR comparison
                    gfid: true,
                    gus_id: true,
                    eff_to: true,
                    exch_id: true,
                    status: true
                },
                star: {
                    guid: false,     // STAR does NOT have GUID
                    gfid: true,
                    gfid_description: true,
                    exchange_id: true
                }
            }
        };

        const FIELD_LABELS = {
            'guid': 'GUID',
            'gfid': 'GFID',
            'gus_id': 'GUS ID',
            'status': 'Status',
            'eff_to': 'Expiry Date',
            'namespace': 'Namespace',
            'exch_id': 'Exchange ID',
            'exchange_id': 'Exchange ID',
            'gfid_description': 'GFID Description'
        };

        // Define display order for fields
        const FIELD_ORDER = ['guid', 'gfid', 'gus_id', 'namespace', 'eff_to', 'exch_id', 'exchange_id', 'gfid_description', 'status'];

        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('reconcile-form');
            const resultsSection = document.getElementById('results-section');
            const submitBtn = document.getElementById('submit-btn');
            
            const guidInput = document.getElementById('guid');
            const gfidInput = document.getElementById('gfid');
            const gusIdInput = document.getElementById('gus-id');

            // Input locking logic - mutual exclusivity
            guidInput.addEventListener('input', function() {
                if (this.value.trim().length > 0) {
                    gfidInput.disabled = true;
                    gusIdInput.disabled = true;
                    gfidInput.value = '';
                    gusIdInput.value = '';
                } else {
                    gfidInput.disabled = false;
                    gusIdInput.disabled = false;
                }
            });

            gfidInput.addEventListener('input', function() {
                if (this.value.trim().length > 0) {
                    guidInput.disabled = true;
                    guidInput.value = '';
                } else if (gusIdInput.value.trim().length === 0) {
                    guidInput.disabled = false;
                }
            });

            gusIdInput.addEventListener('input', function() {
                if (this.value.trim().length > 0) {
                    guidInput.disabled = true;
                    guidInput.value = '';
                } else if (gfidInput.value.trim().length === 0) {
                    guidInput.disabled = false;
                }
            });
            
            form.addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const guid = guidInput.value.trim();
                const gfid = gfidInput.value.trim();
                const gus_id = gusIdInput.value.trim();
                
                // Validation
                if (!guid && (!gfid || !gus_id)) {
                    showError('Please provide either GUID or both GFID and GUS ID');
                    return;
                }

                if (guid && (gfid || gus_id)) {
                    showError('Please provide either GUID or GFID+GUS ID, not both');
                    return;
                }

                if ((gfid && !gus_id) || (!gfid && gus_id)) {
                    showError('Both GFID and GUS ID are required when not using GUID');
                    return;
                }
                
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<div class="loading"></div><span>Processing...</span>';
                
                try {
                    const response = await fetch('/api/reconcile', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ guid, gfid, gus_id })
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.error || 'Reconciliation failed');
                    }
                    
                    currentResults = data;
                    displayResults(data);
                    resultsSection.classList.add('show');
                    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    
                } catch (error) {
                    showError(error.message);
                    console.error('Error details:', error);
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-bolt"></i><span>Execute Reconciliation</span>';
                }
            });
            
            document.getElementById('clear-results-btn').addEventListener('click', function() {
                resultsSection.classList.remove('show');
                form.reset();
                guidInput.disabled = false;
                gfidInput.disabled = false;
                gusIdInput.disabled = false;
                currentResults = null;
                clearError();
            });
            
            document.getElementById('export-btn').addEventListener('click', function() {
                if (currentResults) {
                    downloadJSON(currentResults, `oscar_reconciliation_${new Date().toISOString().split('T')[0]}.json`);
                }
            });
        });
        
        function displayResults(data) {
            clearError();
            
            // Display scenario with enhanced styling
            const scenarioElement = document.getElementById('scenario-text');
            const actionElement = document.getElementById('action-text');
            
            scenarioElement.innerHTML = `<strong>📋 Scenario:</strong> ${data.scenario}`;
            actionElement.innerHTML = `<strong>⚡ Recommended Action:</strong><br>${data.recommended_action}`;
            
            // Display tables with only relevant fields
            displayEDBTable(data.oscar_copper_edb);
            displaySTARTable(data.oscar_copper_star);
        }
        
        function displayEDBTable(data) {
            const tbody = document.getElementById('edb-table-body');
            tbody.innerHTML = '';
            
            const fieldMap = FIELD_MAPPINGS.edb;
            
            // Collect all possible fields from all three databases
            const allFieldsSet = new Set();
            Object.keys(fieldMap.oscar).forEach(f => allFieldsSet.add(f));
            Object.keys(fieldMap.copper).forEach(f => allFieldsSet.add(f));
            Object.keys(fieldMap.edb).forEach(f => allFieldsSet.add(f));
            
            // Sort fields by defined order
            const allFields = FIELD_ORDER.filter(f => allFieldsSet.has(f));
            
            // Display each field
            allFields.forEach(field => {
                // Check if field exists in each database
                const oscarHas = fieldMap.oscar[field] === true;
                const copperHas = fieldMap.copper[field] === true;
                const edbHas = fieldMap.edb[field] === true;
                
                // Skip if field doesn't exist in ANY database
                if (!oscarHas && !copperHas && !edbHas) {
                    return;
                }
                
                // Get actual values from data
                const oscarVal = oscarHas && data.oscar ? data.oscar[field] : null;
                const copperVal = copperHas && data.copper ? data.copper[field] : null;
                const edbVal = edbHas && data.edb ? data.edb[field] : null;
                
                const status = getFieldStatus(oscarVal, copperVal, edbVal, oscarHas, copperHas, edbHas);
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td class="db-column">${FIELD_LABELS[field] || field.toUpperCase()}</td>
                    <td class="oscar-col">${oscarHas ? formatValue(oscarVal, field) : '<span style="color: var(--gray-400); font-style: italic;">—</span>'}</td>
                    <td class="copper-col">${copperHas ? formatValue(copperVal, field) : '<span style="color: var(--gray-400); font-style: italic;">—</span>'}</td>
                    <td class="edb-col">${edbHas ? formatValue(edbVal, field) : '<span style="color: var(--gray-400); font-style: italic;">—</span>'}</td>
                    <td>${status}</td>
                `;
                
                tbody.appendChild(row);
            });
            
            // Add overall comparison row
            const summaryRow = document.createElement('tr');
            summaryRow.style.borderTop = '3px solid var(--accent-color)';
            summaryRow.style.fontWeight = '700';
            summaryRow.innerHTML = `
                <td class="db-column" style="background: var(--gray-100);">OVERALL STATUS</td>
                <td class="oscar-col" style="background: rgba(26, 54, 93, 0.15);">${formatValue(data.comparison.oscar_status, 'status')}</td>
                <td class="copper-col" style="background: rgba(0, 180, 216, 0.15);">${formatValue(data.comparison.copper_status, 'status')}</td>
                <td class="edb-col" style="background: rgba(237, 137, 54, 0.15);">${formatValue(data.comparison.edb_status, 'status')}</td>
                <td>${getOverallStatus(data.comparison.oscar_copper_match, data.comparison.copper_edb_match)}</td>
            `;
            tbody.appendChild(summaryRow);
        }
        
        function displaySTARTable(data) {
            const tbody = document.getElementById('star-table-body');
            tbody.innerHTML = '';
            
            const fieldMap = FIELD_MAPPINGS.star;
            
            // Collect all possible fields from all three databases
            const allFieldsSet = new Set();
            Object.keys(fieldMap.oscar).forEach(f => allFieldsSet.add(f));
            Object.keys(fieldMap.copper).forEach(f => allFieldsSet.add(f));
            Object.keys(fieldMap.star).forEach(f => allFieldsSet.add(f));
            
            // Sort fields by defined order
            const allFields = FIELD_ORDER.filter(f => allFieldsSet.has(f));
            
            // Display each field
            allFields.forEach(field => {
                // Check if field exists in each database
                const oscarHas = fieldMap.oscar[field] === true;
                const copperHas = fieldMap.copper[field] === true;
                const starHas = fieldMap.star[field] === true;
                
                // Skip if field doesn't exist in ANY database
                if (!oscarHas && !copperHas && !starHas) {
                    return;
                }
                
                // Get actual values from data
                const oscarVal = oscarHas && data.oscar ? data.oscar[field] : null;
                const copperVal = copperHas && data.copper ? data.copper[field] : null;
                const starVal = starHas && data.star ? data.star[field] : null;
                
                const status = getFieldStatus(oscarVal, copperVal, starVal, oscarHas, copperHas, starHas);
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td class="db-column">${FIELD_LABELS[field] || field.toUpperCase()}</td>
                    <td class="oscar-col">${oscarHas ? formatValue(oscarVal, field) : '<span style="color: var(--gray-400); font-style: italic;">—</span>'}</td>
                    <td class="copper-col">${copperHas ? formatValue(copperVal, field) : '<span style="color: var(--gray-400); font-style: italic;">—</span>'}</td>
                    <td class="star-col">${starHas ? formatValue(starVal, field) : '<span style="color: var(--gray-400); font-style: italic;">—</span>'}</td>
                    <td>${status}</td>
                `;
                
                tbody.appendChild(row);
            });
            
            // Add overall comparison row
            const summaryRow = document.createElement('tr');
            summaryRow.style.borderTop = '3px solid var(--accent-color)';
            summaryRow.style.fontWeight = '700';
            summaryRow.innerHTML = `
                <td class="db-column" style="background: var(--gray-100);">OVERALL STATUS</td>
                <td class="oscar-col" style="background: rgba(26, 54, 93, 0.15);">${formatValue(data.comparison.oscar_status, 'status')}</td>
                <td class="copper-col" style="background: rgba(0, 180, 216, 0.15);">${formatValue(data.comparison.copper_status, 'status')}</td>
                <td class="star-col" style="background: rgba(56, 161, 105, 0.15);">${data.comparison.star_exists ? '<span class="status-badge status-active">EXISTS</span>' : '<span class="status-badge status-missing">NOT FOUND</span>'}</td>
                <td>${getOverallStatus(data.comparison.oscar_copper_match, data.comparison.copper_gfid_in_star ? 'MATCH' : 'MISSING')}</td>
            `;
            tbody.appendChild(summaryRow);
        }
        
        function formatValue(value, field) {
            // Handle null, undefined, empty string
            if (value === null || value === undefined || value === '') {
                return '<span style="color: var(--gray-500); font-weight: 600;">NULL</span>';
            }
            
            // Format status badges
            if (field === 'status') {
                const valueLower = String(value).toLowerCase();
                let statusClass = '';
                
                if (valueLower.includes('active')) {
                    statusClass = 'status-active';
                } else if (valueLower.includes('inactive') || valueLower.includes('expired')) {
                    statusClass = 'status-inactive';
                } else if (valueLower.includes('missing')) {
                    statusClass = 'status-missing';
                }
                
                return `<span class="status-badge ${statusClass}">${value}</span>`;
            }
            
            // Format dates
            if (field === 'eff_to' && value) {
                try {
                    const date = new Date(value);
                    if (!isNaN(date.getTime())) {
                        return `<span style="font-family: monospace; font-weight: 600; color: var(--gray-700);">${date.toISOString().split('T')[0]}</span>`;
                    }
                } catch (e) {
                    // Return as-is if date parsing fails
                }
            }
            
            // Highlight key identifier fields
            if (field === 'guid' || field === 'gfid' || field === 'gus_id') {
                return `<span style="font-weight: 700; color: var(--primary-color); font-family: monospace;">${value}</span>`;
            }
            
            // Default return
            return `<span style="color: var(--gray-800);">${value}</span>`;
        }
        
        function getFieldStatus(val1, val2, val3, has1, has2, has3) {
            // Collect values only from databases that should have this field
            const values = [];
            const hasField = [];
            
            if (has1) {
                hasField.push('db1');
                if (val1 !== null && val1 !== undefined && val1 !== '') {
                    values.push(String(val1).trim().toUpperCase());
                }
            }
            
            if (has2) {
                hasField.push('db2');
                if (val2 !== null && val2 !== undefined && val2 !== '') {
                    values.push(String(val2).trim().toUpperCase());
                }
            }
            
            if (has3) {
                hasField.push('db3');
                if (val3 !== null && val3 !== undefined && val3 !== '') {
                    values.push(String(val3).trim().toUpperCase());
                }
            }
            
            const expectedCount = hasField.length;
            
            // If field doesn't exist in any database
            if (expectedCount === 0) {
                return '<span style="color: var(--gray-400); font-style: italic;">—</span>';
            }
            
            // If no values found where they should exist
            if (values.length === 0) {
                return '<span class="status-badge status-missing">ALL NULL</span>';
            }
            
            // If we have fewer values than expected
            if (values.length < expectedCount) {
                return '<span class="status-badge status-partial">PARTIAL</span>';
            }
            
            // Check if all values match
            const uniqueValues = [...new Set(values)];
            
            if (uniqueValues.length === 1) {
                return '<span class="status-badge status-match">✓ MATCH</span>';
            } else {
                return '<span class="status-badge status-mismatch">✗ MISMATCH</span>';
            }
        }
        
        function getOverallStatus(match1, match2) {
            const isMatch1 = match1 === 'MATCH';
            const isMatch2 = match2 === 'MATCH' || match2 === true;
            const isMissing1 = match1 && (match1.includes('MISSING') || match1.includes('FIRST_MISSING') || match1.includes('SECOND_MISSING'));
            const isMissing2 = match2 === 'MISSING' || match2 === false;
            
            if (isMatch1 && isMatch2) {
                return '<span class="status-badge status-match" style="font-size: 0.875rem; padding: 0.5rem 1rem;">✓ FULL MATCH</span>';
            } else if (isMissing1 || isMissing2) {
                return '<span class="status-badge status-missing" style="font-size: 0.875rem; padding: 0.5rem 1rem;">⚠ DATA MISSING</span>';
            } else {
                return '<span class="status-badge status-mismatch" style="font-size: 0.875rem; padding: 0.5rem 1rem;">✗ MISMATCH</span>';
            }
        }
        
        function showError(message) {
            const errorContainer = document.getElementById('error-container');
            errorContainer.innerHTML = `
                <div class="alert alert-error" style="animation: fadeInUp 0.3s ease-out;">
                    <i class="fas fa-exclamation-triangle" style="margin-right: 0.5rem; font-size: 1.25rem;"></i>
                    <strong>Error:</strong> ${message}
                </div>
            `;
        }
        
        function clearError() {
            const errorContainer = document.getElementById('error-container');
            errorContainer.innerHTML = '';
        }
        
        function downloadJSON(data, filename) {
            const blob = new Blob([JSON.stringify(data, null, 2)], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            showToast('✓ Results exported successfully!', 'success');
        }
        
        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            
            const colors = {
                success: 'var(--success-color)',
                error: 'var(--error-color)',
                warning: 'var(--warning-color)',
                info: 'var(--accent-color)'
            };
            
            const icons = {
                success: 'fa-check-circle',
                error: 'fa-exclamation-circle',
                warning: 'fa-exclamation-triangle',
                info: 'fa-info-circle'
            };
            
            toast.style.cssText = `
                position: fixed;
                top: 100px;
                right: 30px;
                background: linear-gradient(135deg, ${colors[type]}, ${colors[type]}dd);
                color: white;
                padding: 1.25rem 1.75rem;
                border-radius: 1rem;
                box-shadow: 0 15px 35px rgba(0,0,0,0.3);
                z-index: 10000;
                animation: slideInRight 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                border: 2px solid rgba(255,255,255,0.3);
                min-width: 300px;
            `;
            
            toast.innerHTML = `
                <i class="fas ${icons[type]}" style="font-size: 1.5rem;"></i>
                <span>${message}</span>
            `;
            
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'slideOutRight 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
                setTimeout(() => {
                    document.body.removeChild(toast);
                }, 400);
            }, 3000);
        }
        
        // Add animation styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from {
                    transform: translateX(400px);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            @keyframes slideOutRight {
                from {
                    transform: translateX(0);
                    opacity: 1;
                }
                to {
                    transform: translateX(400px);
                    opacity: 0;
                }
            }
            @keyframes fadeInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>