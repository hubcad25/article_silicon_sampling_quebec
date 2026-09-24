import './styles.css';
import { useState, useEffect, useMemo, useCallback } from 'react';

    const DatasetViewer = () => {
      const [files, setFiles] = useState([]);
      const [selectedFile, setSelectedFile] = useState('');
      const [data, setData] = useState([]);
      const [loading, setLoading] = useState(false);
      const [error, setError] = useState('');
      const [currentPage, setCurrentPage] = useState(0);
      const [filters, setFilters] = useState({
        language: '',
        year: '',
        hasContext: '',
        searchText: ''
      });

      const ITEMS_PER_PAGE = 20;

      // Load available JSONL files on mount
      useEffect(() => {
        const loadFileList = async () => {
          try {
            const res = await fetch('/api/files');
            const links = await res.json();
            setFiles(links);
            if (links.length > 0) {
              setSelectedFile(links[0]);
            }
          } catch (err) {
            setError('Could not load file list. Use drag-and-drop or file picker instead.');
          }
        };
        loadFileList();
      }, []);

      // Load selected file
      useEffect(() => {
        if (!selectedFile) return;

        setLoading(true);
        setError('');
        setCurrentPage(0);

        const loadFile = async () => {
          try {
            const res = await fetch(`/data/datasets/${selectedFile}`);
            const text = await res.text();
            const lines = text.trim().split('\n');
            const parsed = lines.map((line, idx) => {
              try {
                return JSON.parse(line);
              } catch (e) {
                console.error(`Error parsing line ${idx}:`, e);
                return null;
              }
            }).filter(Boolean);
            setData(parsed);
            setLoading(false);
          } catch (err) {
            setError(`Failed to load file: ${err.message}`);
            setLoading(false);
          }
        };

        loadFile();
      }, [selectedFile]);

      // Parse message structure
      const parseMessages = (item) => {
        const messages = item.messages || [];
        const system = messages.find(m => m.role === 'system')?.content || '';
        const user = messages.find(m => m.role === 'user')?.content || '';
        const assistant = messages.find(m => m.role === 'assistant')?.content || '';
        return { system, user, assistant };
      };

      // Detect language
      const detectLanguage = (systemText) => {
        return systemText.startsWith('Tu es') ? 'FR' : 'EN';
      };

      // Extract year from system message
      const extractYear = (systemText) => {
        const match = systemText.match(/(?:conducted|mené)\s+(?:in|en)\s+(\d{4})/i);
        return match ? match[1] : null;
      };

      // Check if has context
      const hasContext = (userText) => {
        return userText.includes('→') || userText.includes('Tes réponses');
      };

      // Filter data
      const filteredData = useMemo(() => {
        return data.filter(item => {
          const { system, user } = parseMessages(item);
          const lang = detectLanguage(system);
          const year = extractYear(system);
          const context = hasContext(user);

          if (filters.language && lang !== filters.language) return false;
          if (filters.year && year !== filters.year) return false;
          if (filters.hasContext === 'yes' && !context) return false;
          if (filters.hasContext === 'no' && context) return false;
          if (filters.searchText) {
            const text = `${system} ${user} ${parseMessages(item).assistant}`.toLowerCase();
            if (!text.includes(filters.searchText.toLowerCase())) return false;
          }
          return true;
        });
      }, [data, filters]);

      // Pagination
      const paginatedData = useMemo(() => {
        const start = currentPage * ITEMS_PER_PAGE;
        return filteredData.slice(start, start + ITEMS_PER_PAGE);
      }, [filteredData, currentPage]);

      // Summary stats
      const stats = useMemo(() => {
        const langs = { FR: 0, EN: 0 };
        const years = {};
        let totalContext = 0;
        const answers = {};
        let totalContextLines = 0;

        filteredData.forEach(item => {
          const { system, user, assistant } = parseMessages(item);
          const lang = detectLanguage(system);
          const year = extractYear(system);
          const context = hasContext(user);

          langs[lang]++;
          if (year) years[year] = (years[year] || 0) + 1;
          if (context) {
            totalContext++;
            const lines = user.split('\n').filter(l => l.includes('→')).length;
            totalContextLines += lines;
          }
          answers[assistant] = (answers[assistant] || 0) + 1;
        });

        const topAnswers = Object.entries(answers)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 10)
          .map(([ans, count]) => ({ answer: ans, count }));

        return {
          total: filteredData.length,
          langs,
          years,
          contextExamples: totalContext,
          avgContextLines: totalContext > 0 ? (totalContextLines / totalContext).toFixed(1) : 0,
          topAnswers
        };
      }, [filteredData]);

      // Get unique years for filter
      const availableYears = useMemo(() => {
        const years = new Set();
        data.forEach(item => {
          const { system } = parseMessages(item);
          const year = extractYear(system);
          if (year) years.add(year);
        });
        return Array.from(years).sort();
      }, [data]);

      // Handlers
      const handleFilterChange = (key, value) => {
        setFilters(prev => ({ ...prev, [key]: value }));
        setCurrentPage(0);
      };

      const handleFileUpload = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        loadJsonlFile(file);
      };

      const handleDragDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        const file = e.dataTransfer.files?.[0];
        if (file) loadJsonlFile(file);
      };

      const loadJsonlFile = (file) => {
        setLoading(true);
        setError('');
        setCurrentPage(0);
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const text = e.target.result;
            const lines = text.trim().split('\n');
            const parsed = lines.map((line, idx) => {
              try {
                return JSON.parse(line);
              } catch (err) {
                console.error(`Error parsing line ${idx}:`, err);
                return null;
              }
            }).filter(Boolean);
            setData(parsed);
            setSelectedFile(`[Uploaded: ${file.name}]`);
            setLoading(false);
          } catch (err) {
            setError(`Failed to parse file: ${err.message}`);
            setLoading(false);
          }
        };
        reader.readAsText(file);
      };

      return (
        <div className="container">
          <h1>Dataset Viewer</h1>

          {error && <div className="error-message">{error}</div>}

          <div className="file-section">
            <div className="file-controls">
              <div className="control-group">
                <label>Select Dataset File</label>
                <select value={selectedFile} onChange={(e) => setSelectedFile(e.target.value)}>
                  <option value="">-- Choose a file --</option>
                  {files.map(f => (
                    <option key={f} value={f}>{f}</option>
                  ))}
                </select>
              </div>
              <div className="control-group">
                <label>Or Upload JSONL File</label>
                <input type="file" accept=".jsonl,.json" onChange={handleFileUpload} />
              </div>
            </div>

            {selectedFile && (
              <div className="file-info">
                <div className="file-info-item">
                  <span className="file-info-label">File:</span>
                  <span>{selectedFile}</span>
                </div>
                <div className="file-info-item">
                  <span className="file-info-label">Examples:</span>
                  <span>{loading ? 'Loading...' : data.length}</span>
                </div>
              </div>
            )}
          </div>

          {data.length > 0 && (
            <>
              <div className="filters-section">
                <div className="filters-grid">
                  <div className="control-group">
                    <label>Language</label>
                    <select value={filters.language} onChange={(e) => handleFilterChange('language', e.target.value)}>
                      <option value="">All</option>
                      <option value="FR">French</option>
                      <option value="EN">English</option>
                    </select>
                  </div>
                  <div className="control-group">
                    <label>Survey Year</label>
                    <select value={filters.year} onChange={(e) => handleFilterChange('year', e.target.value)}>
                      <option value="">All</option>
                      {availableYears.map(y => (
                        <option key={y} value={y}>{y}</option>
                      ))}
                    </select>
                  </div>
                  <div className="control-group">
                    <label>Context</label>
                    <select value={filters.hasContext} onChange={(e) => handleFilterChange('hasContext', e.target.value)}>
                      <option value="">All</option>
                      <option value="yes">Has Context</option>
                      <option value="no">No Context</option>
                    </select>
                  </div>
                </div>
                <div className="control-group">
                  <label>Search All Messages</label>
                  <input
                    type="text"
                    className="search-input"
                    placeholder="Search..."
                    value={filters.searchText}
                    onChange={(e) => handleFilterChange('searchText', e.target.value)}
                  />
                </div>
              </div>

              <div className="summary-panel">
                <div className="summary-grid">
                  <div className="summary-item">
                    <div className="summary-label">Total Examples</div>
                    <div className="summary-value">{stats.total}</div>
                  </div>
                  <div className="summary-item">
                    <div className="summary-label">Language Split</div>
                    <div className="summary-stat">French: {stats.langs.FR}</div>
                    <div className="summary-stat">English: {stats.langs.EN}</div>
                  </div>
                  <div className="summary-item">
                    <div className="summary-label">With Context</div>
                    <div className="summary-value">{stats.contextExamples}</div>
                    <div className="summary-stat">Avg context lines: {stats.avgContextLines}</div>
                  </div>
                </div>

                {Object.keys(stats.years).length > 0 && (
                  <div className="summary-stat">
                    <strong>By year:</strong> {Object.entries(stats.years).map(([y, c]) => `${y}: ${c}`).join(', ')}
                  </div>
                )}

                {stats.topAnswers.length > 0 && (
                  <div className="top-answers">
                    <div className="top-answers-title">Top 10 Most Frequent Answers</div>
                    <ul className="top-answers-list">
                      {stats.topAnswers.map((item, idx) => (
                        <li key={idx} className="top-answers-item">
                          "{item.answer}" — {item.count} times
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div className="chat-section">
                {filteredData.length === 0 ? (
                  <div className="no-results">No examples match the current filters.</div>
                ) : (
                  <>
                    <div className="examples-list">
                      {paginatedData.map((item, idx) => {
                        const { system, user, assistant } = parseMessages(item);
                        const globalIndex = currentPage * ITEMS_PER_PAGE + idx;
                        return (
                          <div key={globalIndex} className="example-card">
                            <div className="example-index">Example #{globalIndex + 1} (in filtered set)</div>
                            <div className="message system">
                              <div className="message-label">System (Persona)</div>
                              <div className="message-content">{system}</div>
                            </div>
                            <div className="message user">
                              <div className="message-label">User (Question)</div>
                              <div className="message-content">{user}</div>
                            </div>
                            <div className="message assistant">
                              <div className="message-label">Assistant (Answer)</div>
                              <div className="message-content">{assistant}</div>
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="pagination">
                      <button
                        onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                        disabled={currentPage === 0}
                      >
                        ← Previous
                      </button>
                      <div className="pagination-info">
                        Page {currentPage + 1} of {Math.ceil(filteredData.length / ITEMS_PER_PAGE)}
                        ({filteredData.length} results)
                      </div>
                      <button
                        onClick={() => setCurrentPage(p => p + 1)}
                        disabled={(currentPage + 1) * ITEMS_PER_PAGE >= filteredData.length}
                      >
                        Next →
                      </button>
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </div>
      );
    };

export default DatasetViewer;
