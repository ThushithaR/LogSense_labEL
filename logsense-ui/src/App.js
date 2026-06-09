import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  AlertCircle, 
  CheckCircle, 
  Terminal, 
  Search, 
  Database, 
  Cpu, 
  ShieldCheck, 
  History,
  Info
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell 
} from 'recharts';

const API_BASE = 'http://localhost:8000';

const App = () => {
  const [logLine, setLogLine] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState({ status: 'checking', model: '...' });

  useEffect(() => {
    checkHealth();
    const saved = localStorage.getItem('logsense_history');
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setHealth(data);
    } catch (e) {
      setHealth({ status: 'offline', model: 'none' });
    }
  };

  const analyzeLog = async () => {
    if (!logLine.trim()) return;
    setLoading(true);
    setResult(null);

    try {
      // 1. Get Prediction
      const predictRes = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ log_line: logLine })
      });
      const predictData = await predictRes.json();

      // 2. Get Explanation
      const explainRes = await fetch(`${API_BASE}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ log_line: logLine })
      });
      const explainData = await explainRes.json();

      const combined = {
        ...predictData,
        top_tokens: explainData.top_tokens
      };

      setResult(combined);
      
      const newHistory = [combined, ...history].slice(0, 10);
      setHistory(newHistory);
      localStorage.setItem('logsense_history', JSON.stringify(newHistory));
    } catch (e) {
      console.error(e);
      alert("Failed to connect to API. Is uvicorn running?");
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (label) => {
    return label === 'ANOMALY' ? 'text-rose-500' : 'text-emerald-500';
  };

  const getBgColor = (label) => {
    return label === 'ANOMALY' ? 'bg-rose-500/10 border-rose-500/20' : 'bg-emerald-500/10 border-emerald-500/20';
  };

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 font-sans p-6 md:p-12">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        body { font-family: 'Inter', sans-serif; margin: 0; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); }
        .gradient-text { background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .btn-primary { background: linear-gradient(135deg, #3b82f6, #2563eb); transition: all 0.2s; border: none; cursor: pointer; color: white; }
        .btn-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3); }
        .animate-pulse-slow { animation: pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .7; } }
        
        /* Utility classes for the layout since we aren't using Tailwind's full build */
        .flex { display: flex; }
        .flex-col { flex-direction: column; }
        .items-center { align-items: center; }
        .justify-between { justify-content: space-between; }
        .gap-2 { gap: 0.5rem; }
        .gap-3 { gap: 0.75rem; }
        .gap-4 { gap: 1rem; }
        .gap-6 { gap: 1.5rem; }
        .gap-8 { gap: 2rem; }
        .mb-12 { margin-bottom: 3rem; }
        .mb-6 { margin-bottom: 1.5rem; }
        .mb-4 { margin-bottom: 1rem; }
        .mt-6 { margin-top: 1.5rem; }
        .mt-12 { margin-top: 3rem; }
        .p-6 { padding: 1.5rem; }
        .p-8 { padding: 2rem; }
        .p-5 { padding: 1.25rem; }
        .px-6 { padding-left: 1.5rem; padding-right: 1.5rem; }
        .py-2.5 { padding-top: 0.625rem; padding-bottom: 0.625rem; }
        .rounded-xl { border-radius: 0.75rem; }
        .rounded-2xl { border-radius: 1rem; }
        .rounded-3xl { border-radius: 1.5rem; }
        .rounded-full { border-radius: 9999px; }
        .w-full { width: 100%; }
        .max-w-6xl { max-width: 72rem; margin-left: auto; margin-right: auto; }
        .grid { display: grid; }
        .grid-cols-1 { grid-template-columns: repeat(1, minmax(0, 1fr)); }
        @media (min-width: 1024px) { .lg\\:grid-cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); } .lg\\:col-span-2 { grid-column: span 2 / span 2; } }
        .bg-slate-800 { background-color: #1e293b; }
        .bg-slate-900 { background-color: #0f172a; }
        .text-rose-500 { color: #f43f5e; }
        .text-emerald-500 { color: #10b981; }
        .text-blue-400 { color: #60a5fa; }
        .text-slate-400 { color: #94a3b8; }
        .text-slate-500 { color: #64748b; }
        .text-slate-200 { color: #e2e8f0; }
        .text-white { color: #ffffff; }
        .bg-blue-600 { background-color: #2563eb; }
        .bg-emerald-500 { background-color: #10b981; }
        .bg-rose-500 { background-color: #f43f5e; }
        .bg-emerald-500\/10 { background-color: rgba(16, 185, 129, 0.1); }
        .border-emerald-500\/20 { border-color: rgba(16, 185, 129, 0.2); }
        .bg-rose-500\/10 { background-color: rgba(244, 63, 94, 0.1); }
        .border-rose-500\/20 { border-color: rgba(244, 63, 94, 0.2); }
        .font-bold { font-weight: 700; }
        .font-black { font-weight: 900; }
        .uppercase { text-transform: uppercase; }
        .tracking-tight { letter-spacing: -0.025em; }
        .tracking-wider { letter-spacing: 0.05em; }
        .tracking-widest { letter-spacing: 0.1em; }
        .text-xs { font-size: 0.75rem; }
        .text-sm { font-size: 0.875rem; }
        .text-2xl { font-size: 1.5rem; }
        .text-3xl { font-size: 1.875rem; }
        .text-4xl { font-size: 2.25rem; }
        .shadow-xl { box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04); }
        .border { border: 1px solid #1e293b; }
        .border-slate-700 { border-color: #334155; }
        .border-slate-800 { border-color: #1e293b; }
        .relative { position: relative; }
        .absolute { position: absolute; }
        .bottom-4 { bottom: 1rem; }
        .right-4 { right: 1rem; }
      `}</style>

      {/* Header */}
      <header className="max-w-6xl flex flex-col items-center justify-between mb-12 gap-4" style={{flexDirection: 'row'}}>
        <div className="flex items-center gap-3">
          <div className="p-6 bg-blue-600 rounded-xl shadow-xl" style={{padding: '0.75rem'}}>
            <ShieldCheck size={32} className="text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Log<span className="gradient-text">Sense</span></h1>
            <p className="text-slate-400 text-sm flex items-center gap-2">
              <Activity size={14} className="text-emerald-500" />
              Real-time Log Anomaly Engine
            </p>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${health.status === 'ok' ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} style={{width: '0.5rem', height: '0.5rem'}} />
            <span className="uppercase tracking-wider text-xs font-semibold">{health.status}</span>
          </div>
        </div>
      </header>

      <main className="max-w-6xl grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Column: Input */}
        <div className="lg:col-span-2" style={{gridColumn: 'span 2'}}>
          <section className="glass rounded-3xl p-8 shadow-xl mb-6">
            <div className="flex items-center gap-2 mb-6 text-slate-400 uppercase tracking-widest text-xs font-bold">
              <Terminal size={16} />
              <span>Log Entry Analysis</span>
            </div>
            
            <div className="relative">
              <textarea
                className="w-full bg-slate-900 border border-slate-700 rounded-2xl p-6 text-white mono text-sm focus:outline-none"
                style={{minHeight: '140px', boxSizing: 'border-box'}}
                placeholder="Paste log line here... e.g., sshd[1234]: Failed password for root from 192.168.1.5"
                value={logLine}
                onChange={(e) => setLogLine(e.target.value)}
              />
              <button 
                onClick={analyzeLog}
                disabled={loading || !logLine.trim()}
                className={`absolute bottom-4 right-4 flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold btn-primary`}
                style={{opacity: loading || !logLine.trim() ? 0.5 : 1}}
              >
                {loading ? '...' : <Search size={18} />}
                {loading ? 'Analyzing...' : 'Detect Anomaly'}
              </button>
            </div>

            <div className="mt-6 flex gap-2" style={{flexWrap: 'wrap'}}>
              <span className="text-xs text-slate-500" style={{alignSelf: 'center', marginRight: '0.5rem'}}>Presets:</span>
              {[
                { name: 'SSH Success', log: 'May 10 10:30:05 server sshd[4821]: Accepted publickey for user1 from 10.0.0.5' },
                { name: 'Brute Force', log: 'May 10 10:31:12 server sshd[1234]: Failed password for root from 192.168.1.5' },
                { name: 'Kernel OOM', log: 'kernel[0]: Out of memory: Kill process 14532 (java) score 892' },
                { name: 'Priv Esc', log: 'sudo: alice : USER=root ; COMMAND=/bin/bash' },
              ].map(p => (
                <button 
                  key={p.name}
                  onClick={() => setLogLine(p.log)}
                  className="text-xs bg-slate-800 text-slate-400 px-6 py-2.5 rounded-xl border border-slate-700"
                  style={{padding: '0.375rem 0.75rem', cursor: 'pointer'}}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </section>

          {result && (
            <section className={`glass rounded-3xl p-8 shadow-xl`} style={{borderLeft: `4px solid ${result.label === 'ANOMALY' ? '#f43f5e' : '#10b981'}`}}>
              <div className="flex justify-between gap-8" style={{flexDirection: 'row'}}>
                <div style={{flex: 1}}>
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <h2 className="text-sm font-bold text-slate-400 uppercase tracking-widest mb-1">Inference Results</h2>
                      <div className={`text-4xl font-black flex items-center gap-3 ${getStatusColor(result.label)}`}>
                        {result.label === 'ANOMALY' ? <AlertCircle size={36} /> : <CheckCircle size={36} />}
                        {result.label}
                      </div>
                    </div>
                    <div style={{textAlign: 'right'}}>
                      <div className="text-sm text-slate-500 font-medium">Confidence Score</div>
                      <div className="text-2xl font-bold text-white">{(result.confidence * 100).toFixed(1)}%</div>
                    </div>
                  </div>

                  <div className="bg-slate-900 rounded-2xl p-5 border border-slate-800">
                    <div className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase mb-4">
                      <Database size={14} />
                      <span>Normalized Pattern</span>
                    </div>
                    <div className="mono text-xs text-blue-400 p-6 rounded-xl bg-slate-800" style={{padding: '0.75rem'}}>
                      {result.normalized_log}
                    </div>
                    <div className="text-xs text-slate-600 mt-6">
                      Latency: {result.latency_ms}ms
                    </div>
                  </div>
                </div>

                <div style={{flex: 1, minHeight: '250px'}}>
                  <div className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Token Attribution (SHAP)</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={result.top_tokens} layout="vertical" margin={{ left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
                      <XAxis type="number" hide />
                      <YAxis 
                        dataKey="token" 
                        type="category" 
                        stroke="#94a3b8" 
                        fontSize={12} 
                        tickLine={false} 
                        axisLine={false}
                        width={80}
                      />
                      <Tooltip 
                        cursor={{ fill: '#1e293b' }}
                        contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }}
                      />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]} barSize={20}>
                        {result.top_tokens.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.importance > 0 ? '#f43f5e' : '#3b82f6'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </section>
          )}
        </div>

        {/* Right Column: History */}
        <div className="space-y-6">
          <section className="glass rounded-3xl p-6 shadow-xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2 text-slate-400 uppercase tracking-widest text-xs font-bold">
                <History size={16} />
                <span>Recent History</span>
              </div>
            </div>

            <div className="space-y-3" style={{maxHeight: '600px', overflowY: 'auto'}}>
              {history.length === 0 ? (
                <div style={{textAlign: 'center', padding: '3rem', color: '#475569'}}>
                  <p className="text-sm">No analysis history yet</p>
                </div>
              ) : (
                history.map((item, i) => (
                  <div key={i} className="p-6 rounded-2xl border mb-3" style={{backgroundColor: item.label === 'ANOMALY' ? 'rgba(244, 63, 94, 0.1)' : 'rgba(16, 185, 129, 0.1)', cursor: 'pointer'}} onClick={() => setResult(item)}>
                    <div className="flex justify-between items-start mb-2">
                      <span className={`text-xs font-bold uppercase ${getStatusColor(item.label)}`}>{item.label}</span>
                      <span className="text-xs text-slate-500">{(item.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="text-xs text-slate-400 mono" style={{overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'}}>
                      {item.normalized_log}
                    </div>
                  </div>
                ))
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default App;
