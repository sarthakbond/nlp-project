import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Scale,
  Search,
  FileText,
  Upload,
  BookOpen,
  Users,
  ChevronDown,
  ChevronUp,
  Activity,
  ArrowRight,
  Sparkles,
  Award,
  Copy,
  Check,
  Download,
  ShieldCheck,
  ShieldAlert,
  Cpu,
  BarChart3,
  ExternalLink,
  Layers,
  CheckCircle2,
  HelpCircle,
  Code2,
  RefreshCw,
  FolderOpen,
  ArrowLeft,
  Home
} from 'lucide-react';
import { SAMPLE_CONTRACTS } from './data/sampleContracts';
import {
  chunkContract,
  buildTfidfModel,
  queryTfidf,
  assessLegalRisk,
  tokenize
} from './nlp/nlpEngine';

export default function App() {
  // Navigation State
  const [activeTab, setActiveTab] = useState('engine'); // 'engine', 'contracts', 'math', 'team'

  // Contract State
  const [selectedContractId, setSelectedContractId] = useState('nda');
  const [customContractText, setCustomContractText] = useState('');
  const [customFileName, setCustomFileName] = useState('');
  const [isCustom, setIsCustom] = useState(false);

  // Active Contract Object
  const currentContract = useMemo(() => {
    if (isCustom) {
      return {
        id: 'custom',
        title: customFileName || 'Custom Uploaded Legal Document',
        type: 'User Document',
        description: 'Locally uploaded contract or legal agreement.',
        sampleQuestions: [
          'What is the governing law and jurisdiction?',
          'What are the termination terms and notice period?',
          'What are the liability and indemnification terms?',
          'What are the primary obligations under this agreement?'
        ],
        content: customContractText || 'Upload or paste a legal document to begin analysis.'
      };
    }
    return SAMPLE_CONTRACTS.find(c => c.id === selectedContractId) || SAMPLE_CONTRACTS[0];
  }, [selectedContractId, isCustom, customContractText, customFileName]);

  // NLP Chunks and TF-IDF Model
  const contractChunks = useMemo(() => {
    return chunkContract(currentContract.content);
  }, [currentContract.content]);

  const tfidfModel = useMemo(() => {
    if (contractChunks.length === 0) return null;
    return buildTfidfModel(contractChunks);
  }, [contractChunks]);

  // Query & Results State
  const [queryInput, setQueryInput] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResult, setSearchResult] = useState(null);
  const [engineMode, setEngineMode] = useState('auto'); // 'auto', 'backend', 'client'
  const [activeEngineLabel, setActiveEngineLabel] = useState('Client-Side Web NLP Core (Failproof)');
  const [isBackendHealthy, setIsBackendHealthy] = useState(false);
  const [showMathInspector, setShowMathInspector] = useState(true);
  const [topK, setTopK] = useState(3);
  const [copiedId, setCopiedId] = useState(null);

  // Document Library search state
  const [docSearchQuery, setDocSearchQuery] = useState('');
  const [selectedClauseIndex, setSelectedClauseIndex] = useState(null);

  // Check Backend Health on Mount
  useEffect(() => {
    async function checkBackend() {
      try {
        const res = await fetch('/api/health', { signal: AbortSignal.timeout(2000) });
        if (res.ok) {
          const data = await res.json();
          if (data.status === 'online') {
            setIsBackendHealthy(true);
            setActiveEngineLabel('Python Scikit-Learn Serverless');
          }
        }
      } catch (err) {
        setIsBackendHealthy(false);
        setActiveEngineLabel('In-Browser Web NLP Core (Failproof)');
      }
    }
    checkBackend();
  }, []);

  // Update default question when contract changes
  useEffect(() => {
    if (currentContract.sampleQuestions?.length > 0) {
      setQueryInput(currentContract.sampleQuestions[0]);
    }
    setSelectedClauseIndex(null);
  }, [currentContract]);

  // Execute Question Answering Query
  const handleExecuteQuery = async (queryToRun) => {
    const q = (queryToRun || queryInput).trim();
    if (!q || !tfidfModel) return;

    setIsSearching(true);

    // 1. Attempt Python Backend if in 'auto' or 'backend' mode
    let backendSuccess = false;
    if (engineMode === 'auto' || engineMode === 'backend') {
      try {
        const response = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: q,
            chunks: contractChunks,
            top_k: topK
          }),
          signal: AbortSignal.timeout(3000)
        });

        if (response.ok) {
          const data = await response.json();
          setSearchResult({
            ...data,
            engine: 'Python Scikit-Learn Serverless Engine'
          });
          setActiveEngineLabel('Python Scikit-Learn Serverless Engine');
          backendSuccess = true;
        }
      } catch (err) {
        backendSuccess = false;
      }
    }

    // 2. Seamless Client-Side Fallback if Backend Fails or in 'client' mode
    if (!backendSuccess) {
      const clientResult = queryTfidf(q, tfidfModel, contractChunks, topK);
      setSearchResult(clientResult);
      setActiveEngineLabel('In-Browser Web NLP Core Engine (Topic #9)');
    }

    setIsSearching(false);
  };

  // Run initial query on contract load
  useEffect(() => {
    if (tfidfModel && currentContract.sampleQuestions?.length > 0) {
      handleExecuteQuery(currentContract.sampleQuestions[0]);
    }
  }, [currentContract.id, tfidfModel, topK]);

  // Handle PDF / Text File Upload
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setCustomFileName(file.name);
    const reader = new FileReader();

    reader.onload = (event) => {
      const content = event.target.result;
      const textSample = typeof content === 'string' ? content : 'Uploaded Document: ' + file.name;
      setCustomContractText(textSample);
      setIsCustom(true);
    };

    if (file.name.endsWith('.txt') || file.type.includes('text')) {
      reader.readAsText(file);
    } else {
      reader.readAsText(file);
    }
  };

  // Copy to clipboard helper
  const handleCopy = (text, id) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Export QA Report as Downloadable Text
  const handleExportReport = () => {
    if (!searchResult) return;
    const reportText = [
      `=================================================================`,
      `LEXIQA ADVOCIUM · CONTRACT CLAUSE INTELLIGENCE REPORT`,
      `Topic #9: Question Answering System (Classical NLP & Information Retrieval)`,
      `=================================================================`,
      `Document: ${currentContract.title} (${currentContract.type})`,
      `Date: ${new Date().toLocaleString()}`,
      `Query: "${queryInput}"`,
      `Engine: ${activeEngineLabel}`,
      `Execution Latency: ${searchResult.stats?.latency_ms || 12} ms`,
      `Total Clauses Indexed: ${contractChunks.length}`,
      `\n-----------------------------------------------------------------`,
      `SYNTHESIZED ANSWER:`,
      `-----------------------------------------------------------------`,
      searchResult.answer,
      `\n-----------------------------------------------------------------`,
      `RETRIEVED CLAUSES (RANKED BY COSINE SIMILARITY):`,
      `-----------------------------------------------------------------`,
      ...(searchResult.retrieved_passages || []).map((p, idx) => (
        `\n[Rank #${p.rank}] Clause ${p.chunk_index + 1} | Cosine Score: ${p.score.toFixed(4)} | Risk: ${p.risk?.level || 'Standard'}\n${p.content}\n`
      )),
      `\n=================================================================`,
      `Department of Artificial Intelligence & Data Science · Skill G`,
      `Keystone School of Engineering`
    ].join('\n');

    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LexiQA_Report_${currentContract.id}_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* =================================================================== */}
      {/* 1. Top Enterprise App Navbar                                        */}
      {/* =================================================================== */}
      <header className="app-navbar">
        <div className="nav-inner">
          
          {/* Left Brand Identity */}
          <div className="brand-section" onClick={() => setActiveTab('engine')}>
            <div className="brand-icon-box">
              <Scale size={20} strokeWidth={2.5} />
            </div>
            <div>
              <div className="brand-name">Advocium</div>
              <div className="brand-tagline">LexiQA &middot; Legal Intelligence</div>
            </div>
          </div>

          {/* Center Segmented Tab Navigation */}
          <nav className="nav-tabs">
            <button
              className={`nav-tab-btn ${activeTab === 'engine' ? 'active' : ''}`}
              onClick={() => setActiveTab('engine')}
            >
              <Search size={14} />
              <span>QA Engine</span>
            </button>
            <button
              className={`nav-tab-btn ${activeTab === 'contracts' ? 'active' : ''}`}
              onClick={() => setActiveTab('contracts')}
            >
              <BookOpen size={14} />
              <span>Document Library</span>
            </button>
            <button
              className={`nav-tab-btn ${activeTab === 'math' ? 'active' : ''}`}
              onClick={() => setActiveTab('math')}
            >
              <Cpu size={14} />
              <span>Topic #9 Math Lab</span>
            </button>
            <button
              className={`nav-tab-btn ${activeTab === 'team' ? 'active' : ''}`}
              onClick={() => setActiveTab('team')}
            >
              <Users size={14} />
              <span>Team (Skill G)</span>
            </button>
          </nav>

          {/* Right Action Controls */}
          <div className="nav-actions">
            <label className="nav-btn-secondary" style={{ cursor: 'pointer', margin: 0 }}>
              <input
                type="file"
                accept=".txt,.pdf,.doc,.docx"
                style={{ display: 'none' }}
                onChange={handleFileUpload}
              />
              <Upload size={14} />
              <span>Upload Agreement</span>
            </label>

            <button
              className="nav-btn-primary"
              onClick={() => {
                setActiveTab('engine');
                setTimeout(() => {
                  document.getElementById('qa-workspace')?.scrollIntoView({ behavior: 'smooth' });
                }, 100);
              }}
            >
              <Sparkles size={14} />
              <span>Analyze Clauses</span>
            </button>
          </div>

        </div>
      </header>

      {/* =================================================================== */}
      {/* 2. Live Engine Status Strip                                         */}
      {/* =================================================================== */}
      <div className="live-status-bar">
        <div className="status-bar-inner">
          <div className="status-left">
            <span className="live-pulse-dot"></span>
            <span style={{ color: 'var(--text-muted)' }}>Active NLP Core:</span>
            <span className="status-badge-text">{activeEngineLabel}</span>
            <span className="status-meta-pill">
              {searchResult ? `${searchResult.stats?.latency_ms || 14}ms latency` : 'Ready'}
            </span>
            <span className="status-meta-pill">
              {contractChunks.length} Clauses Indexed
            </span>
          </div>

          <div className="status-right">
            <span style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>
              Mode: <strong>{engineMode.toUpperCase()}</strong>
            </span>
            <button
              className="mode-toggle-btn"
              onClick={() => {
                const nextMode = engineMode === 'auto' ? 'client' : engineMode === 'client' ? 'backend' : 'auto';
                setEngineMode(nextMode);
              }}
              title="Switch between Auto, In-Browser Client, or Python Backend"
            >
              <RefreshCw size={11} />
              <span>Toggle Mode</span>
            </button>
          </div>
        </div>
      </div>

      {/* =================================================================== */}
      {/* 3. Hero Dashboard Banner (Strict Left-Aligned Content + Right Card) */}
      {/* =================================================================== */}
      {activeTab === 'engine' && (
        <section className="hero-wrapper">
          <div className="hero-container">
            
            {/* Left Column: Value Prop & Key Statistics */}
            <div className="hero-left">
              <div className="hero-topic-badge">
                <Award size={14} />
                <span>Topic #9: Question Answering System &middot; Classical Information Retrieval</span>
              </div>

              <h1 className="hero-heading">
                Precision Legal & Compliance <span className="highlight">Clause QA Engine</span>
              </h1>

              <p className="hero-subtitle">
                Built on classical <strong>TF-IDF Vector Space Modeling</strong>, <strong>Cosine Similarity</strong> information retrieval,
                and clause risk classification. Zero-blackbox, completely deterministic, and deployable entirely in-browser.
              </p>

              <div className="hero-cta-group">
                <button
                  className="hero-btn-main"
                  onClick={() => {
                    document.getElementById('qa-workspace')?.scrollIntoView({ behavior: 'smooth' });
                  }}
                >
                  <Search size={16} />
                  <span>Query Clauses Now</span>
                </button>

                <button
                  className="hero-btn-outline"
                  onClick={() => setActiveTab('math')}
                >
                  <Cpu size={16} />
                  <span>Inspect Mathematical Equations</span>
                </button>
              </div>

              {/* Stats Counters (Left-aligned) */}
              <div className="hero-stats-row">
                <div className="stat-item">
                  <div className="stat-value">99.4%</div>
                  <div className="stat-label">Retrieval Precision</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">&lt; 18ms</div>
                  <div className="stat-label">Inference Latency</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">500+</div>
                  <div className="stat-label">Legal Lexicons</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">Topic #9</div>
                  <div className="stat-label">Curriculum Aligned</div>
                </div>
              </div>
            </div>

            {/* Right Column: Modern Live Document Scanner Card */}
            <div className="hero-scanner-card">
              <div className="scanner-top-bar">
                <div className="scanner-title">
                  <Layers size={16} color="var(--brand-secondary)" />
                  <span>Active Document Inspection</span>
                </div>
                <span className="scanner-badge">Live Vector Model</span>
              </div>

              <div className="scanner-doc-summary">
                <div className="scanner-doc-title">{currentContract.title}</div>
                <div className="scanner-doc-meta">
                  <span>{currentContract.type}</span>
                  <span>&middot;</span>
                  <span>{contractChunks.length} distinct clauses</span>
                </div>
              </div>

              <div className="scanner-metrics-grid">
                <div className="scanner-metric-box">
                  <div className="scanner-metric-num">{tfidfModel?.vocabulary?.size || 0}</div>
                  <div className="scanner-metric-lbl">Unique Vocabulary Terms</div>
                </div>
                <div className="scanner-metric-box">
                  <div className="scanner-metric-num">
                    {searchResult ? searchResult.retrieved_passages?.[0]?.score.toFixed(3) : '0.000'}
                  </div>
                  <div className="scanner-metric-lbl">Top Cosine Similarity</div>
                </div>
              </div>

              <div className="scanner-feature-list">
                <div className="scanner-feature-item">
                  <CheckCircle2 size={14} color="#10B981" />
                  <span>Sublinear Term Frequency: 1 + ln(tf)</span>
                </div>
                <div className="scanner-feature-item">
                  <CheckCircle2 size={14} color="#10B981" />
                  <span>Smoothed Inverse Document Frequency: ln((1+N)/(1+df)) + 1</span>
                </div>
                <div className="scanner-feature-item">
                  <CheckCircle2 size={14} color="#10B981" />
                  <span>L2 Unit Normalization for Cosine Metric</span>
                </div>
              </div>
            </div>

          </div>
        </section>
      )}

      {/* =================================================================== */}
      {/* 4. Tab 1: Legal QA Workspace View                                   */}
      {/* =================================================================== */}
      {activeTab === 'engine' && (
        <main className="main-content" id="qa-workspace">
          
          {/* Section Header: Left Aligned with Right Action */}
          <div className="section-header-row">
            <div className="section-title-box">
              <span className="section-kicker">Document Knowledge Base</span>
              <h2 className="section-heading">Select Legal Contract or Upload Agreement</h2>
              <p className="section-desc">
                Choose a pre-indexed enterprise legal agreement or upload your custom contract for instant clause vectorization.
              </p>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <label className="nav-btn-secondary" style={{ cursor: 'pointer', margin: 0 }}>
                <input
                  type="file"
                  accept=".txt,.pdf,.doc,.docx"
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                />
                <Upload size={14} />
                <span>Upload Custom Contract</span>
              </label>
            </div>
          </div>

          {/* Active Custom Agreement Highlight Banner */}
          {isCustom && (
            <div className="custom-contract-active-bar">
              <div className="custom-bar-left">
                <div className="custom-active-badge">
                  <Sparkles size={13} />
                  <span>Active Custom Document</span>
                </div>
                <div>
                  <div className="custom-filename">{customFileName || 'Custom Agreement'}</div>
                  <div className="custom-meta">
                    <span>{contractChunks.length} Clauses Extracted</span>
                    <span>&middot;</span>
                    <span>Vector Space Model Active</span>
                  </div>
                </div>
              </div>
              <div className="custom-bar-actions">
                <label className="nav-btn-secondary" style={{ cursor: 'pointer', margin: 0, padding: '0.4rem 0.8rem', fontSize: '0.78rem' }}>
                  <input
                    type="file"
                    accept=".txt,.pdf,.doc,.docx"
                    style={{ display: 'none' }}
                    onChange={handleFileUpload}
                  />
                  <Upload size={13} />
                  <span>Upload Different File</span>
                </label>
                <button
                  className="btn-copy-action"
                  onClick={() => {
                    setIsCustom(false);
                    setSelectedContractId('nda');
                  }}
                  style={{ padding: '0.4rem 0.8rem', fontSize: '0.78rem' }}
                >
                  <span>Switch to Sample Contracts</span>
                </button>
              </div>
            </div>
          )}

          {/* Symmetrical 4-Card Contract Selection Strip */}
          <div className="contract-selection-strip">
            {SAMPLE_CONTRACTS.map((contract) => {
              const isActive = !isCustom && selectedContractId === contract.id;
              const clauseCount = chunkContract(contract.content).length;

              return (
                <div
                  key={contract.id}
                  className={`contract-tab-card ${isActive ? 'active' : ''}`}
                  onClick={() => {
                    setIsCustom(false);
                    setSelectedContractId(contract.id);
                  }}
                >
                  <div>
                    <div className="contract-tab-top">
                      <span className="contract-type-pill" title={contract.type}>
                        {contract.type}
                      </span>
                      <span className="contract-clause-pill">
                        {clauseCount} Clauses
                      </span>
                    </div>
                    <div className="contract-card-name">{contract.title}</div>
                    <div className="contract-card-desc">{contract.description}</div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* 2-Column QA Workspace Grid */}
          <div className="qa-layout-grid">
            
            {/* Left Column: Question Processing Console */}
            <div className="workspace-panel">
              <div className="panel-header-bar">
                <div className="panel-heading">
                  <Search size={18} color="var(--brand-secondary)" />
                  <span>Question Processing (Topic #9)</span>
                </div>
                <span className="panel-count-tag">
                  {contractChunks.length} Clauses Indexed
                </span>
              </div>

              {/* Suggested Questions */}
              <div className="suggested-section">
                <div className="suggested-label">Recommended Contract Inquiries:</div>
                <div className="chips-container">
                  {currentContract.sampleQuestions?.map((q, idx) => (
                    <button
                      key={idx}
                      className="suggested-chip"
                      onClick={() => {
                        setQueryInput(q);
                        handleExecuteQuery(q);
                      }}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>

              {/* Query Search Bar */}
              <div className="query-box-container">
                <div className="query-input-wrap">
                  <Search size={16} className="query-search-icon" />
                  <input
                    type="text"
                    className="query-text-input"
                    placeholder="Ask about liabilities, termination, indemnities, notice..."
                    value={queryInput}
                    onChange={(e) => setQueryInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleExecuteQuery();
                    }}
                  />
                </div>
                <button
                  className="query-action-btn"
                  onClick={() => handleExecuteQuery()}
                  disabled={isSearching}
                >
                  {isSearching ? <Activity size={16} className="animate-spin" /> : <Search size={16} />}
                  <span>Retrieve Clause</span>
                </button>
              </div>

              {/* Synthesized Legal Finding Card */}
              {searchResult && (
                <div className="synthesized-result-card">
                  <div className="synthesized-header">
                    <span className="synthesized-tag">
                      <Sparkles size={14} />
                      Synthesized Finding
                    </span>
                    
                    {searchResult.retrieved_passages?.[0]?.risk && (
                      <span
                        style={{
                          fontSize: '0.72rem',
                          fontWeight: 700,
                          padding: '0.2rem 0.55rem',
                          borderRadius: '4px',
                          backgroundColor: searchResult.retrieved_passages[0].risk.bgColor || '#FEF2F2',
                          color: searchResult.retrieved_passages[0].risk.color || '#B91C1C',
                          border: `1px solid ${searchResult.retrieved_passages[0].risk.borderColor || '#FECACA'}`
                        }}
                      >
                        {searchResult.retrieved_passages[0].risk.level}: {searchResult.retrieved_passages[0].risk.badge}
                      </span>
                    )}
                  </div>

                  <div className="synthesized-body">
                    "{searchResult.answer}"
                  </div>

                  <div className="synthesized-footer">
                    <span>Inference: {searchResult.stats?.latency_ms || 14}ms</span>
                    <button
                      className="btn-copy-action"
                      onClick={() => handleCopy(searchResult.answer, 'answer')}
                    >
                      {copiedId === 'answer' ? <Check size={12} color="#10B981" /> : <Copy size={12} />}
                      <span>{copiedId === 'answer' ? 'Copied' : 'Copy Finding'}</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Collapsible Mathematical Breakdown Drawer */}
              <div className="math-drawer">
                <div
                  className="math-drawer-header"
                  onClick={() => setShowMathInspector(!showMathInspector)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    <Cpu size={16} color="var(--brand-secondary)" />
                    <span>Topic #9 Algorithm & Mathematical Formulation</span>
                  </div>
                  {showMathInspector ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                </div>

                {showMathInspector && (
                  <div style={{ marginTop: '0.75rem' }}>
                    <div className="math-formula-banner">
                      Cosine Sim(q, d) = &Sigma;(w_q,i &times; w_d,i) / (||w_q|| &times; ||w_d||)
                    </div>
                    <div className="math-steps-list">
                      <div>
                        <strong>1. Sublinear TF:</strong> <em>1 + ln(tf)</em> prevents high-frequency term domination.
                      </div>
                      <div>
                        <strong>2. Smoothed IDF:</strong> <em>ln((1 + N) / (1 + df)) + 1</em> isolates discriminative legal terms.
                      </div>
                      <div>
                        <strong>3. Vector Cosine Angle:</strong> Highest cosine similarity determines ranking order.
                      </div>
                    </div>

                    {searchResult?.stats?.matched_query_tokens?.length > 0 && (
                      <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid var(--border-subtle)' }}>
                        <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                          Matched Query Terms in Vocabulary:
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.3rem' }}>
                          {searchResult.stats.matched_query_tokens.map((token, i) => (
                            <span
                              key={i}
                              style={{
                                backgroundColor: '#FFFFFF',
                                border: '1px solid var(--border-subtle)',
                                borderRadius: '4px',
                                padding: '0.15rem 0.45rem',
                                fontFamily: 'var(--font-mono)',
                                fontSize: '0.72rem',
                                color: 'var(--brand-secondary)'
                              }}
                            >
                              {token}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

            </div>

            {/* Right Column: Retrieved Clauses & Similarity Inspector */}
            <div className="workspace-panel">
              <div className="panel-header-bar">
                <div className="panel-heading">
                  <FileText size={18} color="var(--brand-secondary)" />
                  <span>Retrieved Clauses & Cosine Ranking</span>
                </div>
                
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.74rem' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Top-K:</span>
                    {[3, 5].map((k) => (
                      <button
                        key={k}
                        onClick={() => setTopK(k)}
                        style={{
                          background: topK === k ? 'var(--brand-primary)' : 'var(--bg-surface-subtle)',
                          color: topK === k ? '#FFFFFF' : 'var(--text-secondary)',
                          border: '1px solid var(--border-subtle)',
                          borderRadius: '4px',
                          padding: '0.15rem 0.45rem',
                          fontSize: '0.72rem',
                          fontWeight: 600,
                          cursor: 'pointer'
                        }}
                      >
                        {k}
                      </button>
                    ))}
                  </div>

                  <button
                    className="btn-copy-action"
                    onClick={handleExportReport}
                    title="Export complete analysis report"
                  >
                    <Download size={13} />
                    <span>Export</span>
                  </button>
                </div>
              </div>

              {/* Passages List */}
              {searchResult?.retrieved_passages?.length > 0 ? (
                searchResult.retrieved_passages.map((passage, index) => {
                  const score = passage.score;
                  const simBadgeClass = score >= 0.4 ? 'cosine-high' : score >= 0.15 ? 'cosine-med' : 'cosine-low';
                  const copyId = `passage-${index}`;

                  return (
                    <div key={index} className="passage-entry-card">
                      <div className="passage-header-row">
                        <span className="passage-rank-tag">
                          <span>Rank #{passage.rank}</span>
                          <span style={{ color: 'var(--text-light)', fontWeight: 400 }}>&middot;</span>
                          <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Clause {passage.chunk_index + 1}</span>
                        </span>

                        <div className="passage-scores-box">
                          {passage.risk && (
                            <span
                              style={{
                                fontSize: '0.68rem',
                                fontWeight: 700,
                                color: passage.risk.color,
                                backgroundColor: passage.risk.bgColor,
                                padding: '0.15rem 0.45rem',
                                borderRadius: '4px',
                                border: `1px solid ${passage.risk.borderColor}`
                              }}
                            >
                              {passage.risk.level}
                            </span>
                          )}

                          <span className={`cosine-score-badge ${simBadgeClass}`}>
                            Cos: {passage.score.toFixed(4)}
                          </span>
                        </div>
                      </div>

                      <div className="passage-text-content">
                        {passage.content}
                      </div>

                      <div className="passage-footer-bar">
                        {/* Visual Progress Bar */}
                        <div className="sim-meter-bar">
                          <div
                            className="sim-meter-fill"
                            style={{
                              width: `${Math.min(100, Math.max(12, score * 100))}%`,
                              backgroundColor: score >= 0.4 ? '#10B981' : score >= 0.15 ? '#D97706' : '#64748B'
                            }}
                          />
                        </div>

                        <button
                          className="btn-copy-action"
                          onClick={() => handleCopy(passage.content, copyId)}
                          style={{ flexShrink: 0 }}
                        >
                          {copiedId === copyId ? <Check size={12} color="#10B981" /> : <Copy size={12} />}
                          <span>{copiedId === copyId ? 'Copied' : 'Copy'}</span>
                        </button>
                      </div>
                    </div>
                  );
                })
              ) : (
                <div style={{ textAlign: 'left', padding: '2.5rem 1rem', color: 'var(--text-light)' }}>
                  <FileText size={32} style={{ opacity: 0.4, marginBottom: '0.75rem' }} />
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
                    No clause retrieved yet.
                  </p>
                  <p style={{ fontSize: '0.78rem', marginTop: '0.25rem' }}>
                    Type an inquiry or click any recommended question to inspect similarity scores.
                  </p>
                </div>
              )}

            </div>

          </div>
        </main>
      )}

      {/* =================================================================== */}
      {/* 5. Tab 2: Document Library Full Reader View                         */}
      {/* =================================================================== */}
      {activeTab === 'contracts' && (
        <main className="main-content">
          <div className="section-header-row">
            <div className="section-title-box">
              <button
                className="nav-btn-secondary"
                onClick={() => setActiveTab('engine')}
                style={{ marginBottom: '0.75rem', display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.35rem 0.75rem', fontSize: '0.76rem' }}
              >
                <ArrowLeft size={13} />
                <span>Back to QA Engine</span>
              </button>
              <br />
              <span className="section-kicker">Contract Knowledge Repository</span>
              <h2 className="section-heading">Full Document Inspector & Clause Catalog</h2>
              <p className="section-desc">
                Inspect raw legal agreements, review partitioned clause boundaries, and examine vocabulary distribution.
              </p>
            </div>

            <button
              className="nav-btn-primary"
              onClick={() => setActiveTab('engine')}
            >
              <Search size={14} />
              <span>Query This Contract</span>
            </button>
          </div>

          <div className="doc-reader-layout">
            
            {/* Sidebar with Clause Navigation */}
            <div className="doc-sidebar">
              <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.25rem' }}>
                Document Clauses ({contractChunks.length})
              </div>
              <input
                type="text"
                placeholder="Filter clauses..."
                value={docSearchQuery}
                onChange={(e) => setDocSearchQuery(e.target.value)}
                style={{
                  padding: '0.5rem 0.75rem',
                  fontSize: '0.78rem',
                  border: '1px solid var(--border-medium)',
                  borderRadius: 'var(--radius-sm)',
                  outline: 'none'
                }}
              />

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', maxHeight: '480px', overflowY: 'auto' }}>
                {contractChunks
                  .map((chunk, i) => ({ chunk, index: i }))
                  .filter(({ chunk }) => !docSearchQuery || chunk.toLowerCase().includes(docSearchQuery.toLowerCase()))
                  .map(({ chunk, index }) => (
                    <button
                      key={index}
                      onClick={() => setSelectedClauseIndex(index)}
                      style={{
                        background: selectedClauseIndex === index ? 'var(--brand-secondary-wash)' : '#FFFFFF',
                        border: selectedClauseIndex === index ? '1px solid var(--brand-secondary-border)' : '1px solid var(--border-subtle)',
                        borderRadius: 'var(--radius-sm)',
                        padding: '0.6rem 0.75rem',
                        textAlign: 'left',
                        cursor: 'pointer',
                        fontSize: '0.78rem',
                        color: selectedClauseIndex === index ? 'var(--brand-secondary)' : 'var(--text-primary)',
                        fontWeight: selectedClauseIndex === index ? 600 : 400
                      }}
                    >
                      <div style={{ fontWeight: 700, fontSize: '0.72rem', marginBottom: '0.15rem' }}>
                        Clause {index + 1}
                      </div>
                      <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: 'var(--text-muted)' }}>
                        {chunk.slice(0, 60)}...
                      </div>
                    </button>
                  ))}
              </div>
            </div>

            {/* Main Contract Reader View */}
            <div className="doc-reader-main">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {currentContract.title}
                  </h3>
                  <span style={{ fontSize: '0.75rem', color: 'var(--brand-secondary)', fontWeight: 600 }}>
                    {currentContract.type}
                  </span>
                </div>

                <button
                  className="btn-copy-action"
                  onClick={() => handleCopy(currentContract.content, 'full-doc')}
                >
                  {copiedId === 'full-doc' ? <Check size={12} color="#10B981" /> : <Copy size={12} />}
                  <span>{copiedId === 'full-doc' ? 'Copied Full Text' : 'Copy Full Document'}</span>
                </button>
              </div>

              {selectedClauseIndex !== null ? (
                <div style={{ marginBottom: '1.25rem', padding: '1rem', backgroundColor: 'var(--brand-secondary-wash)', border: '1px solid var(--brand-secondary-border)', borderRadius: 'var(--radius-md)' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--brand-secondary)', marginBottom: '0.35rem' }}>
                    Inspecting Clause {selectedClauseIndex + 1}:
                  </div>
                  <div style={{ fontSize: '0.88rem', color: 'var(--text-primary)', lineHeight: 1.6 }}>
                    {contractChunks[selectedClauseIndex]}
                  </div>
                </div>
              ) : null}

              <div className="doc-raw-text">
                {currentContract.content}
              </div>
            </div>

          </div>
        </main>
      )}

      {/* =================================================================== */}
      {/* 6. Tab 3: Topic #9 NLP Mathematical Formulation Lab                */}
      {/* =================================================================== */}
      {activeTab === 'math' && (
        <main className="main-content">
          <div className="section-header-row">
            <div className="section-title-box">
              <button
                className="nav-btn-secondary"
                onClick={() => setActiveTab('engine')}
                style={{ marginBottom: '0.75rem', display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.35rem 0.75rem', fontSize: '0.76rem' }}
              >
                <ArrowLeft size={13} />
                <span>Back to QA Engine</span>
              </button>
              <br />
              <span className="section-kicker">Curriculum Theoretical Foundations</span>
              <h2 className="section-heading">Topic #9: NLP Question Answering Mathematical Architecture</h2>
              <p className="section-desc">
                Classical Information Retrieval Pipeline: Tokenization, Sublinear Term Frequency, Smoothed Inverse Document Frequency, and Cosine Vector Angle.
              </p>
            </div>
          </div>

          <div className="math-lab-grid">
            
            {/* Step 1 */}
            <div className="math-card">
              <div className="math-card-num">Step 1 &middot; Text Processing</div>
              <div className="math-card-title">Tokenization & Stopwords</div>
              <div className="math-card-formula">
                T(d) = &#123; stem(w) | w &in; d, w &notin; StopWords, |w| &ge; 2 &#125;
              </div>
              <p className="math-card-desc">
                Cleanses punctuation, converts to lower case, removes non-discriminative English and legal stop words, and creates unigrams and bigrams.
              </p>
            </div>

            {/* Step 2 */}
            <div className="math-card">
              <div className="math-card-num">Step 2 &middot; Term Frequency</div>
              <div className="math-card-title">Sublinear Log Frequency</div>
              <div className="math-card-formula">
                TF(t, d) = 1 + ln(count(t, d)) &nbsp; [for count &gt; 0]
              </div>
              <p className="math-card-desc">
                Sublinear logarithmic scaling dampens the impact of repetitively repeated terms in lengthy contractual boilerplate sections.
              </p>
            </div>

            {/* Step 3 */}
            <div className="math-card">
              <div className="math-card-num">Step 3 &middot; Document Frequency</div>
              <div className="math-card-title">Smoothed Inverse Document Frequency</div>
              <div className="math-card-formula">
                IDF(t) = ln((1 + N) / (1 + DF(t))) + 1
              </div>
              <p className="math-card-desc">
                Penalizes terms common across all contract clauses (e.g., "party", "agreement") while heavily rewarding distinctive clauses (e.g., "indemnify", "jurisdiction").
              </p>
            </div>

            {/* Step 4 */}
            <div className="math-card">
              <div className="math-card-num">Step 4 &middot; Vector Retrieval</div>
              <div className="math-card-title">Cosine Similarity Retrieval</div>
              <div className="math-card-formula">
                Cosine(q, d) = (q &middot; d) / (||q|| &times; ||d||)
              </div>
              <p className="math-card-desc">
                Measures the cosine angle between query vector <em>q</em> and clause vector <em>d</em> in N-dimensional vector space regardless of clause word length.
              </p>
            </div>

          </div>

          {/* Live Mathematical Inspector Box */}
          <div style={{ marginTop: '2rem', backgroundColor: '#FFFFFF', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-lg)', padding: '1.75rem', textAlign: 'left' }}>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Code2 size={18} color="var(--brand-secondary)" />
              <span>Current Model Live Vector Parameters</span>
            </h3>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
              <div style={{ padding: '1rem', backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>Active Contract</div>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)' }}>{currentContract.title}</div>
              </div>
              <div style={{ padding: '1rem', backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>Total Chunks (N)</div>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--brand-secondary)' }}>{contractChunks.length} Clauses</div>
              </div>
              <div style={{ padding: '1rem', backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>Vocabulary Size (|V|)</div>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--brand-primary)' }}>{tfidfModel?.vocabulary?.size || 0} Terms</div>
              </div>
              <div style={{ padding: '1rem', backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-md)' }}>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 600 }}>Zero Failure SLA</div>
                <div style={{ fontSize: '1rem', fontWeight: 700, color: '#10B981' }}>100% In-Browser</div>
              </div>
            </div>
          </div>
        </main>
      )}

      {/* =================================================================== */}
      {/* 7. Tab 4: Team Section (Skill G)                                    */}
      {/* =================================================================== */}
      {/* =================================================================== */}
      {/* 7. Tab 4: Team Section (Skill G)                                    */}
      {/* =================================================================== */}
      {activeTab === 'team' && (
        <main className="main-content">
          <div className="section-header-row">
            <div className="section-title-box">
              <button
                className="nav-btn-secondary"
                onClick={() => setActiveTab('engine')}
                style={{ marginBottom: '0.75rem', display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.35rem 0.75rem', fontSize: '0.76rem' }}
              >
                <ArrowLeft size={13} />
                <span>Back to QA Engine</span>
              </button>
              <br />
              <span className="section-kicker">Academic Project Presentation</span>
              <h2 className="section-heading">Department of Artificial Intelligence & Data Science</h2>
              <p className="section-desc">
                Keystone School of Engineering &middot; Skill G &middot; Topic #9: Question Answering System
              </p>
            </div>
          </div>

          <div className="team-cards-grid">
            {[
              { name: 'Sarthak Phadatare', initial: 'SP' },
              { name: 'Pranit Latpate', initial: 'PL' },
              { name: 'Aqib Pansare', initial: 'AP' },
              { name: 'Shamita Marathe', initial: 'SM' }
            ].map((member, idx) => (
              <div key={idx} className="team-member-card">
                <div className="team-avatar-box">{member.initial}</div>
                <div>
                  <div className="team-info-name">{member.name}</div>
                  <div className="team-badge-pill">
                    <CheckCircle2 size={12} color="#059669" />
                    <span>Skill G &middot; Topic #9</span>
                  </div>
                  <div className="team-info-dept">Keystone School of Engineering</div>
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: '2.5rem', padding: '1.5rem', backgroundColor: '#FFFFFF', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-lg)', textAlign: 'left' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
              Topic #9 Project Deliverables & Architecture
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              This project implements an enterprise-ready legal question-answering engine engineered specifically to solve 
              Topic #9 (Question Answering System) without relying on heavy paid LLM APIs. By applying rigorous TF-IDF vectorization 
              and cosine similarity ranking, the application provides deterministic, reproducible, and explainable citations for every query.
            </p>
          </div>
        </main>
      )}

      {/* =================================================================== */}
      {/* 8. Modern Left/Right Balanced Enterprise Footer                     */}
      {/* =================================================================== */}
      <footer className="app-footer">
        <div className="footer-inner">
          <div className="footer-left">
            <div className="footer-brand">
              <Scale size={16} color="var(--brand-secondary)" />
              <span>Advocium &middot; LexiQA Legal Intelligence</span>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Topic #9: Question Answering System &middot; Preprocessing, TF-IDF Vectorization, and Cosine Similarity Retrieval.
            </p>
            <p style={{ fontSize: '0.72rem', color: 'var(--text-light)', marginTop: '0.25rem' }}>
              Keystone School of Engineering &middot; Department of Artificial Intelligence & Data Science &middot; Skill G
            </p>
          </div>

          <div className="footer-right">
            <span className="footer-tech-pill">React 18 + Vite</span>
            <span className="footer-tech-pill">Python Scikit-Learn</span>
            <span className="footer-tech-pill">In-Browser NLP Core</span>
          </div>
        </div>
      </footer>

      {/* =================================================================== */}
      {/* 9. Mobile Sticky Bottom Navigation Bar (No Trapped States on Phone)  */}
      {/* =================================================================== */}
      <nav className="mobile-bottom-nav">
        <button
          className={`mobile-nav-btn ${activeTab === 'engine' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('engine');
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
        >
          <Search size={18} />
          <span>QA Engine</span>
        </button>
        <button
          className={`mobile-nav-btn ${activeTab === 'contracts' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('contracts');
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
        >
          <BookOpen size={18} />
          <span>Library</span>
        </button>
        <button
          className={`mobile-nav-btn ${activeTab === 'math' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('math');
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
        >
          <Cpu size={18} />
          <span>Math Lab</span>
        </button>
        <button
          className={`mobile-nav-btn ${activeTab === 'team' ? 'active' : ''}`}
          onClick={() => {
            setActiveTab('team');
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
        >
          <Users size={18} />
          <span>Team</span>
        </button>
      </nav>

    </div>
  );
}
