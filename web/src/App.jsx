import React, { useState, useEffect } from 'react';
import ReactJson from 'react-json-view';
import {
  FolderSearch,
  FileText,
  Play,
  Settings,
  CheckCircle,
  AlertCircle,
  Loader2,
  FileJson,
  Sparkles,
  Zap,
  Layout,
  Clock,
  FolderOpen
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from './components/ui/card';
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Progress } from './components/ui/progress';
import { cn } from './lib/utils';

function App() {
  const [path, setPath] = useState('');
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [resultContent, setResultContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [progressValue, setProgressValue] = useState(0);
  const [activeTab, setActiveTab] = useState('process'); // 'process' | 'results'
  const [processedStatus, setProcessedStatus] = useState({}); // { filename: 'success' | 'error' | 'pending' }

  // Load results on mount and when tab changes
  useEffect(() => {
    if (activeTab === 'results') {
      fetchResults();
    }
  }, [activeTab]);

  // Simulate progress when processing
  useEffect(() => {
    let interval;
    if (processing) {
      setProgressValue(0);
      interval = setInterval(() => {
        setProgressValue((prev) => {
          if (prev >= 90) return 90; // Stall at 90% until complete
          return prev + Math.random() * 5;
        });
      }, 500);
    } else {
      setProgressValue(100);
    }
    return () => clearInterval(interval);
  }, [processing]);

  const handleBrowse = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/choose-path');
      const data = await response.json();
      if (data.path) {
        setPath(data.path);
      }
    } catch (error) {
      console.error('Error opening dialog:', error);
      alert('打开文件夹选择器失败，请检查后端服务是否正常运行');
    }
  };

  const handleScan = async () => {
    if (!path.trim()) {
      alert('请先选择或输入 PDF 文件夹路径');
      return;
    }
    setLoading(true);
    setProcessedStatus({}); // Reset status on new scan
    try {
      const response = await fetch('http://localhost:8000/api/scan', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ path }),
      });
      const data = await response.json();
      if (data.files && data.files.length === 0) {
        alert('在该目录下未找到 PDF 文件');
      }
      setFiles(data.files || []);
    } catch (error) {
      console.error('Error scanning:', error);
      alert('扫描目录失败，请检查路径是否正确');
    } finally {
      setLoading(false);
    }
  };

  const handleProcess = async () => {
    const filesToProcess = files.filter(f => selectedFile === null || f === selectedFile);
    if (filesToProcess.length === 0) return;

    setProcessing(true);
    setProgressValue(0);

    // Mark selected as processing locally just for UI feedback if needed, 
    // but we'll wait for result to mark success/error

    try {
      const response = await fetch('http://localhost:8000/api/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          files: filesToProcess,
          model: 'deepseek-chat',
          max_pages: 10,
          max_tokens: 20000
        }),
      });
      const data = await response.json();
      console.log('Processed:', data);

      // Update status for processed files
      const newStatus = { ...processedStatus };
      if (data.results) {
        data.results.forEach(res => {
          // res is roughly { file: path, status: 'success' | 'error', error: ... }
          // We need to map full path back to filename or keep full path mapping
          // The files state stores full paths.
          newStatus[res.file] = res.status;
        });
      }
      setProcessedStatus(newStatus);

      // Delay slightly to show 100% progress but DO NOT switch tabs
      setTimeout(() => {
        setProcessing(false);
        // Alert completion or show success toast - simple alert for now or just rely on UI update
        // alert('处理完成！请到结果管理页面查看详情。'); 
      }, 600);

    } catch (error) {
      console.error('Error processing:', error);
      alert('文件处理失败');
      setProcessing(false);
    }
  };

  const fetchResults = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/results');
      const data = await response.json();
      setResults(data.files || []);
    } catch (error) {
      console.error('Error fetching results:', error);
    }
  };

  const viewResult = async (filename) => {
    try {
      const response = await fetch(`http://localhost:8000/api/results/${filename}`);
      const data = await response.json();
      setResultContent(data);
    } catch (error) {
      console.error('Error fetching result content:', error);
    }
  };

  // Enhanced Variants for "Senior" Feel
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08,
        delayChildren: 0.1,
        ease: [0.22, 1, 0.36, 1]
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 80,
        damping: 15,
        mass: 0.8
      }
    }
  };

  const scaleHover = {
    scale: 1.02,
    transition: { type: 'spring', stiffness: 400, damping: 10 }
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 flex font-sans selection:bg-indigo-500/30 overflow-hidden relative">
      {/* Dynamic Background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute top-[-20%] left-[-10%] w-[800px] h-[800px] bg-indigo-600/10 rounded-full blur-[120px] animate-pulse-slow" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-cyan-600/10 rounded-full blur-[100px] animate-pulse-slow delay-1000" />
        <div className="absolute top-[40%] left-[20%] w-[400px] h-[400px] bg-purple-600/5 rounded-full blur-[80px] animate-pulse-slow delay-500" />
      </div>

      {/* Sidebar */}
      <motion.div
        initial={{ x: -100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ type: 'spring', stiffness: 100, damping: 20 }}
        className="w-72 border-r border-white/5 bg-slate-900/40 backdrop-blur-2xl p-6 flex flex-col gap-8 z-20 relative shadow-2xl shadow-black/20"
      >
        <div className="flex items-center gap-3 px-2 group cursor-pointer">
          <motion.div
            whileHover={{ rotate: 15, scale: 1.1 }}
            className="h-10 w-10 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/30 ring-1 ring-white/20"
          >
            <FileText className="h-5 w-5 text-white" />
          </motion.div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent group-hover:to-white transition-all">
              PageIndex
            </h1>
            <p className="text-[10px] text-slate-500 font-mono tracking-wider uppercase">Enterprise Edition</p>
          </div>
        </div>

        <nav className="flex flex-col gap-2">
          <NavItem
            active={activeTab === 'process'}
            onClick={() => setActiveTab('process')}
            icon={<Zap className="h-4 w-4" />}
            label="智能解析"
          />
          <NavItem
            active={activeTab === 'results'}
            onClick={() => setActiveTab('results')}
            icon={<Layout className="h-4 w-4" />}
            label="结果管理"
          />
        </nav>

        <div className="mt-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <Card className="bg-gradient-to-br from-slate-900/80 to-slate-900/40 border-white/5 p-4 backdrop-blur-md relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              <div className="flex items-center gap-2 mb-2 relative z-10">
                <Sparkles className="h-3.5 w-3.5 text-amber-300 animate-pulse" />
                <span className="text-xs font-semibold text-amber-200/90 tracking-wide">AI MODEL</span>
              </div>
              <p className="text-xs text-slate-400 leading-relaxed relative z-10">
                当前运行 <strong>DeepSeek V3</strong> <br />深度文档理解模型
              </p>
            </Card>
          </motion.div>
        </div>
      </motion.div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden relative z-10 flex flex-col">
        {/* Top Header area (optional, for breadcrumbs or user profile if expanded later) */}

        <AnimatePresence mode="wait">
          {activeTab === 'process' && (
            <motion.div
              key="process"
              variants={containerVariants}
              initial="hidden"
              animate="visible"
              exit={{ opacity: 0, x: -20, transition: { duration: 0.2 } }}
              className="flex-1 p-10 overflow-auto flex flex-col items-center justify-center min-h-0"
            >
              <motion.div variants={itemVariants} className="text-center space-y-6 mb-16 relative">
                {/* Decorative Glow behind title */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[300px] h-[100px] bg-indigo-500/20 blur-[60px] rounded-full pointer-events-none" />

                <h1 className="text-5xl font-bold tracking-tight">
                  <span className="bg-gradient-to-b from-white to-slate-400 bg-clip-text text-transparent">文档智能</span>
                  <span className="bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent ml-3">结构化解析</span>
                </h1>
                <p className="text-slate-400 text-lg max-w-2xl mx-auto leading-relaxed font-light">
                  构建于新一代大语言模型之上，精准提取 PDF 文档的多层级目录树、  <br className="hidden sm:block" />
                  章节摘要与关键实体信息，重塑知识获取体验。
                </p>
              </motion.div>

              <motion.div variants={itemVariants} className="w-full max-w-4xl relative z-10">
                {/* Main Action Card */}
                <Card className="bg-slate-900/40 backdrop-blur-xl border border-white/10 shadow-2xl overflow-hidden relative">
                  <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-cyan-500 opacity-50" />

                  <div className="p-8 space-y-8">
                    {/* Input Area */}
                    <div className="flex gap-4 items-stretch">
                      <div className="relative flex-1 group">
                        <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                          <FolderSearch className="h-5 w-5 text-slate-500 group-focus-within:text-indigo-400 transition-colors duration-300" />
                        </div>
                        <Input
                          placeholder="选择或输入本地 PDF 文件夹路径..."
                          value={path}
                          onChange={(e) => setPath(e.target.value)}
                          className="pl-12 h-14 bg-slate-950/50 border-white/10 focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/20 transition-all font-mono text-sm shadow-inner rounded-xl"
                        />
                      </div>

                      <motion.div whileHover={scaleHover} whileTap={{ scale: 0.95 }}>
                        <Button
                          title="选择文件夹"
                          onClick={handleBrowse}
                          variant="secondary"
                          className="h-14 w-14 px-0 rounded-xl bg-slate-800/80 hover:bg-slate-700 border border-white/5"
                        >
                          <FolderOpen className="h-6 w-6 text-slate-300" />
                        </Button>
                      </motion.div>

                      <motion.div whileHover={scaleHover} whileTap={{ scale: 0.95 }}>
                        <Button
                          onClick={handleScan}
                          disabled={loading}
                          className="h-14 px-10 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-lg font-medium shadow-lg shadow-indigo-600/25 border border-white/10"
                        >
                          {loading ? <Loader2 className="h-6 w-6 animate-spin" /> : "扫描目录"}
                        </Button>
                      </motion.div>
                    </div>

                    {/* Pending/Processing Area */}
                    <AnimatePresence>
                      {(processing || files.length > 0) && (
                        <motion.div
                          initial={{ opacity: 0, height: 0, marginTop: 0 }}
                          animate={{ opacity: 1, height: 'auto', marginTop: 24 }}
                          exit={{ opacity: 0, height: 0, marginTop: 0 }}
                          className="overflow-hidden"
                        >
                          {/* Progress Bar */}
                          {processing && (
                            <div className="mb-6 space-y-3 bg-slate-950/30 p-5 rounded-2xl border border-white/5 relative overflow-hidden">
                              <div className="absolute inset-0 bg-indigo-500/5 animate-pulse" />
                              <div className="flex justify-between text-sm text-indigo-200/80 font-medium relative z-10">
                                <span className="flex items-center gap-2">
                                  <Loader2 className="h-4 w-4 animate-spin text-indigo-400" /> 正在深度解析文档结构...
                                </span>
                                <span className="font-mono">{Math.round(progressValue)}%</span>
                              </div>
                              <Progress value={progressValue} className="h-1.5 bg-slate-800" indicatorClassName="bg-gradient-to-r from-indigo-500 to-cyan-500" />
                            </div>
                          )}

                          {files.length > 0 && (
                            <div className="bg-slate-950/30 rounded-2xl border border-white/5 overflow-hidden">
                              <div className="p-4 border-b border-white/5 flex justify-between items-center bg-white/[0.02]">
                                <div className="flex items-center gap-2">
                                  <div className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
                                  <span className="text-sm font-medium text-slate-300">已发现 {files.length} 个文档</span>
                                </div>
                                <div className="flex gap-2">
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => setFiles([])}
                                    className="h-8 text-xs text-slate-500 hover:text-slate-300 hover:bg-white/5"
                                  >
                                    清空
                                  </Button>
                                  <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
                                    <Button
                                      onClick={handleProcess}
                                      disabled={processing}
                                      className="h-8 text-xs bg-emerald-600/90 hover:bg-emerald-500 px-4 space-x-2"
                                    >
                                      <Play className="h-3 w-3 fill-current" />
                                      <span>开始解析</span>
                                    </Button>
                                  </motion.div>
                                </div>
                              </div>
                              <div className="max-h-[300px] overflow-y-auto p-2 custom-scrollbar">
                                {files.map((file, i) => (
                                  <FileListItem
                                    key={file}
                                    file={file}
                                    status={processedStatus[file]}
                                    index={i}
                                    onViewResults={() => setActiveTab('results')}
                                  />
                                ))}
                              </div>
                            </div>
                          )}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </Card>
              </motion.div>
            </motion.div>
          )}

          {activeTab === 'results' && (
            <motion.div
              key="results"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
              className="flex-1 p-6 flex gap-6 min-h-0"
            >
              {/* Result List */}
              <div className="w-80 flex flex-col gap-4">
                <h2 className="text-lg font-bold text-slate-200 flex items-center gap-2 px-2">
                  <Clock className="h-5 w-5 text-indigo-400" /> 解析历史
                </h2>
                <div className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar pb-10">
                  {results.map((res, i) => (
                    <motion.button
                      key={i}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.05 }}
                      onClick={() => viewResult(res.name)}
                      className={cn(
                        "w-full text-left p-4 rounded-xl border transition-all duration-300 group relative overflow-hidden",
                        resultContent?.name === res.name
                          ? "bg-white/5 border-indigo-500/50 shadow-lg shadow-indigo-500/10"
                          : "bg-slate-900/40 border-white/5 hover:bg-white/5 hover:border-white/10"
                      )}
                    >
                      <div className="flex items-center gap-3 relative z-10">
                        <div className={cn(
                          "h-9 w-9 rounded-lg flex items-center justify-center transition-all duration-300",
                          resultContent?.name === res.name
                            ? "bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-md transform scale-105"
                            : "bg-slate-800 text-slate-500 group-hover:text-slate-300 group-hover:bg-slate-700"
                        )}>
                          <FileJson className="h-4.5 w-4.5" />
                        </div>
                        <div className="overflow-hidden flex-1">
                          <div className={cn(
                            "font-medium text-sm truncate transition-colors",
                            resultContent?.name === res.name ? "text-indigo-200" : "text-slate-300 group-hover:text-slate-100"
                          )}>{res.name}</div>
                          <div className="text-[10px] text-slate-500 mt-1 flex items-center justify-between">
                            <span>JSON 结构数据</span>
                            <span className="opacity-0 group-hover:opacity-100 transition-opacity text-indigo-400">查看 &rarr;</span>
                          </div>
                        </div>
                      </div>
                      {resultContent?.name === res.name && (
                        <div className="absolute inset-0 bg-gradient-to-r from-indigo-500/10 to-transparent pointer-events-none" />
                      )}
                    </motion.button>
                  ))}
                </div>
              </div>

              {/* Preview Area */}
              <div className="flex-1 flex flex-col gap-4 min-h-0">
                <h2 className="text-lg font-bold text-slate-200 flex items-center gap-2 px-2">
                  <Layout className="h-5 w-5 text-cyan-400" /> 结构化内容预览
                </h2>
                <Card className="flex-1 bg-[#1e293b]/50 border-white/5 overflow-hidden relative backdrop-blur-xl shadow-2xl rounded-2xl group">
                  {/* Glass reflection */}
                  <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none z-10" />

                  {resultContent ? (
                    <div className="absolute inset-0 overflow-auto p-8 custom-scrollbar relative z-20">
                      <ReactJson
                        src={resultContent}
                        theme="ocean"
                        displayDataTypes={false}
                        displayObjectSize={false}
                        style={{ backgroundColor: 'transparent', fontSize: '13px', fontFamily: '"JetBrains Mono", monospace' }}
                        enableClipboard={true}
                        indentWidth={4}
                        collapsed={2}
                        name={false}
                        iconStyle="triangle"
                      />
                    </div>
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-6 z-20">
                      <div className="h-32 w-32 rounded-full bg-slate-800/30 flex items-center justify-center relative overlow-hidden">
                        <div className="absolute inset-0 bg-indigo-500/10 rounded-full animate-ping-slow" />
                        <FileJson className="h-12 w-12 opacity-50" />
                      </div>
                      <div className="text-center space-y-2">
                        <p className="text-xl font-medium text-slate-400">暂无预览内容</p>
                        <p className="text-sm text-slate-600">请从左侧列表中选择一个解析结果</p>
                      </div>
                    </div>
                  )}
                </Card>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

function NavItem({ active, onClick, icon, label }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 px-4 py-3.5 rounded-xl text-sm font-medium transition-all duration-300 relative group overflow-hidden",
        active
          ? "text-white shadow-lg shadow-indigo-900/20"
          : "text-slate-400 hover:text-slate-200 hover:bg-white/5"
      )}
    >
      {active && (
        <motion.div
          layoutId="activeTab"
          className="absolute inset-0 bg-gradient-to-r from-indigo-600 to-indigo-500"
          initial={false}
          transition={{ type: "spring", stiffness: 500, damping: 30 }}
        />
      )}

      {/* Icon Background Glow for Active State */}
      {active && <div className="absolute left-8 top-1/2 -translate-y-1/2 w-8 h-8 bg-white/20 blur-xl rounded-full" />}

      <div className={cn("relative z-10 p-0.5 transition-transform duration-300 group-hover:scale-110", active ? "text-white" : "text-slate-500 group-hover:text-slate-300")}>
        {icon}
      </div>
      <span className="relative z-10">{label}</span>
    </button>
  )
}

function FileListItem({ file, status, index, onViewResults }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.05 }}
      className="flex items-center justify-between p-3 rounded-lg hover:bg-white/5 transition-colors group border border-transparent hover:border-white/5"
    >
      <div className="flex items-center gap-3 overflow-hidden">
        <div className={cn(
          "h-8 w-8 rounded-lg flex items-center justify-center transition-all duration-300",
          status === 'success' ? "bg-emerald-500/20 text-emerald-400 ring-1 ring-emerald-500/30" :
            status === 'error' ? "bg-red-500/20 text-red-400 ring-1 ring-red-500/30" :
              "bg-slate-800 text-slate-500 group-hover:bg-indigo-500/20 group-hover:text-indigo-400"
        )}>
          {status === 'success' ? <CheckCircle className="h-4 w-4" /> :
            status === 'error' ? <AlertCircle className="h-4 w-4" /> :
              <FileText className="h-4 w-4" />}
        </div>
        <div className="flex flex-col overflow-hidden">
          <span className="text-sm text-slate-300 truncate font-medium group-hover:text-white transition-colors">{file}</span>
          <span className="text-[10px] text-slate-600 truncate group-hover:text-slate-500">PDF 文档</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {status === 'success' ? (
          <div className="flex items-center gap-3">
            <span className="text-[10px] px-2 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium whitespace-nowrap">已完成</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-slate-500 hover:text-indigo-400 hover:bg-indigo-500/10 rounded-full opacity-0 group-hover:opacity-100 transition-all scale-90 group-hover:scale-100"
              onClick={onViewResults}
              title="查看结果"
            >
              <FileJson className="h-3.5 w-3.5" />
            </Button>
          </div>
        ) : status === 'error' ? (
          <span className="text-xs px-2 py-1 rounded bg-red-500/10 text-red-400 border border-red-500/20 whitespace-nowrap">失败</span>
        ) : (
          <span className="text-[10px] uppercase tracking-wider text-slate-600 font-medium whitespace-nowrap">Pending</span>
        )}
      </div>
    </motion.div>
  )
}

export default App;
